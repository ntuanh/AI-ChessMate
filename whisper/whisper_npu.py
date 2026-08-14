#!/usr/bin/env python3
"""Run Whisper-Small-Quantized on the QCS8550 NPU, end to end.

Drives the two QNN context binaries through ``qnn-net-run``: the encoder once per
utterance, then the decoder once per generated token.  Everything outside the NPU --
the log-mel frontend, the greedy search, the byte-level token decoding -- is plain
numpy here, so the whole path can be checked without a C++ build.

The per-token ``qnn-net-run`` invocation is the slow part, and deliberately so: this
script exists to prove the model runs and transcribes correctly.  A persistent worker
holding both graphs in one process is the optimisation that follows, and it can be
validated against this script's output.

Two things make the chaining work byte-for-byte, both read out of metadata.json:

* ``k/v_cache_self_*_out`` carry the *same* scale and zero point as ``..._in``, so a
  step's output cache is fed straight back as the next step's input with no
  requantisation.
* The encoder's ``k/v_cache_cross_*`` outputs match the decoder's inputs of the same
  name, so the cross-attention cache is written once and reused for every token.

Usage:
    python3 whisper_npu.py audio.wav [--lang vi] [--max-tokens 40]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = 3000
CHUNK = SR * 30  # Whisper always sees exactly 30 seconds

N_LAYERS = 12
CACHE_LEN = 199        # self-attention cache slots
MASK_LEN = 200         # cache slots + the current token

SOT = 50258
TRANSCRIBE = 50359
NOTIMESTAMPS = 50363
EOT = 50257
LANG_BASE = 50259      # <|en|>; every other language is an offset from here


# --------------------------------------------------------------------------
# log-mel frontend
# --------------------------------------------------------------------------

def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale -- the one Whisper uses (librosa default, htk=False)."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    mel = freq / f_sp
    high = freq >= min_log_hz
    mel[high] = min_log_mel + np.log(freq[high] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    freq = mel * f_sp
    high = mel >= min_log_mel
    freq[high] = min_log_hz * np.exp(logstep * (mel[high] - min_log_mel))
    return freq


def mel_filterbank() -> np.ndarray:
    """80 triangular filters with Slaney area normalisation, shape (80, 201)."""
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    lo, hi = _hz_to_mel(np.array([0.0])), _hz_to_mel(np.array([SR / 2.0]))
    mel_points = np.linspace(lo[0], hi[0], N_MELS + 2)
    freqs = _mel_to_hz(mel_points)

    fdiff = np.diff(freqs)
    ramps = freqs[:, None] - fft_freqs[None, :]

    weights = np.zeros((N_MELS, len(fft_freqs)))
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    weights *= (2.0 / (freqs[2:N_MELS + 2] - freqs[:N_MELS]))[:, None]
    return weights


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper's log-mel spectrogram, shape (80, 3000), values roughly in [-1, 1]."""
    if len(audio) < CHUNK:
        audio = np.pad(audio, (0, CHUNK - len(audio)))
    audio = audio[:CHUNK]

    # torch.stft(center=True) reflects at both ends before framing.
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1]  # periodic, matches torch.hann_window(400)

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    spec = np.fft.rfft(frames * window, axis=-1)

    # Whisper drops the final frame, leaving exactly 3000.
    magnitudes = np.abs(spec[:-1].T) ** 2
    mel = mel_filterbank() @ magnitudes

    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def read_wav(path: str) -> np.ndarray:
    """Read a mono 16 kHz PCM16 wav into float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: can dung PCM 16-bit")
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
        if w.getframerate() != SR:
            # Linear resample is good enough for speech commands.
            n = int(round(len(audio) * SR / w.getframerate()))
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                              np.arange(len(audio)), audio).astype(np.float32)
    return audio


# --------------------------------------------------------------------------
# quantisation helpers
# --------------------------------------------------------------------------

def quantize(x: np.ndarray, scale: float, zero_point: int, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    q = np.rint(x / scale) + zero_point
    return np.clip(q, info.min, info.max).astype(dtype)


class Spec:
    """Tensor names, shapes and quantisation parameters, straight from metadata."""

    def __init__(self, metadata_path: str) -> None:
        meta = json.load(open(metadata_path))
        self.enc = meta["model_files"]["encoder.bin"]
        self.dec = meta["model_files"]["decoder.bin"]

    def q(self, section: Dict, io: str, name: str) -> Tuple[float, int]:
        p = section[io][name]["quantization_parameters"]
        return p["scale"], p["zero_point"]


# --------------------------------------------------------------------------
# qnn-net-run driver
# --------------------------------------------------------------------------

class NetRun:
    def __init__(self, binary: str, backend: str, workdir: str) -> None:
        self.binary = binary
        self.backend = backend
        self.workdir = workdir

    def run(self, context: str, inputs: Dict[str, str], out_dir: str,
            tag: str) -> Dict[str, str]:
        """Execute one inference and return {tensor_name: output_path}."""
        list_path = os.path.join(self.workdir, f"{tag}_list.txt")
        with open(list_path, "w") as fh:
            fh.write(" ".join(f"{name}:={path}" for name, path in inputs.items()) + "\n")

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        cmd = [
            self.binary,
            "--backend", self.backend,
            "--retrieve_context", context,
            "--input_list", list_path,
            "--output_dir", out_dir,
            "--use_native_input_files",
            "--use_native_output_files",
            "--perf_profile", "burst",
            # The decoder context binary is 225 MB and qnn-net-run re-reads it for
            # every token.  Memory-mapping lets the page cache serve the repeats and
            # roughly halves the per-step cost (measured 950 ms -> 500 ms).
            "--use_mmap",
        ]
        proc = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        result_dir = os.path.join(out_dir, "Result_0")
        if proc.returncode != 0 or not os.path.isdir(result_dir):
            sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
            raise SystemExit(f"qnn-net-run that bai ({tag})")

        # --use_native_output_files appends "_native" to every filename; strip it so
        # callers can look tensors up by their graph names.
        out: Dict[str, str] = {}
        for fname in os.listdir(result_dir):
            if not fname.endswith(".raw"):
                continue
            stem = os.path.splitext(fname)[0]
            if stem.endswith("_native"):
                stem = stem[: -len("_native")]
            out[stem] = os.path.join(result_dir, fname)
        return out


# --------------------------------------------------------------------------
# token decoding
# --------------------------------------------------------------------------

def byte_decoder() -> Dict[str, int]:
    """Inverse of GPT-2's byte-to-unicode map, needed to turn tokens back into bytes."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Tokenizer:
    def __init__(self, vocab_path: str, added_path: str) -> None:
        vocab = json.load(open(vocab_path, encoding="utf-8"))
        added = json.load(open(added_path, encoding="utf-8"))
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.id_to_token.update({i: t for t, i in added.items()})
        self.token_to_id = {t: i for i, t in self.id_to_token.items()}
        self.byte_dec = byte_decoder()

    def language_token(self, code: str) -> int:
        tok = f"<|{code}|>"
        if tok not in self.token_to_id:
            raise SystemExit(f"khong co token ngon ngu {tok}")
        return self.token_to_id[tok]

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.id_to_token.get(i, "") for i in ids
                       if i < 50257)  # drop every special token
        return bytearray(self.byte_dec.get(ch, 0) for ch in text).decode("utf-8", "replace")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--models", default=HERE)
    ap.add_argument("--workdir", default=os.path.join(HERE, "run"))
    ap.add_argument("--keep-mel", action="store_true")
    args = ap.parse_args()

    sdk = os.environ.get("SDK")
    flavor = os.environ.get("FLAVOR")
    if not sdk or not flavor:
        raise SystemExit("hay source env.sh truoc (can SDK va FLAVOR)")
    net_run = os.path.join(sdk, "bin", flavor, "qnn-net-run")
    backend = os.path.join(sdk, "lib", flavor, "libQnnHtp.so")

    os.makedirs(args.workdir, exist_ok=True)
    spec = Spec(os.path.join(args.models, "metadata.json"))
    tok = Tokenizer(os.path.join(args.models, "vocab.json"),
                    os.path.join(args.models, "added_tokens.json"))
    runner = NetRun(net_run, backend, args.workdir)

    # -- 1. audio -> quantised log-mel ------------------------------------
    t0 = time.time()
    audio = read_wav(args.audio)
    speech_s = len(audio) / SR
    mel = log_mel(audio)
    scale, zp = spec.q(spec.enc, "inputs", "input_features")
    mel_q = quantize(mel[None], scale, zp, np.uint16)
    mel_path = os.path.join(args.workdir, "mel.raw")
    mel_q.tofile(mel_path)
    t_mel = time.time() - t0
    print(f"[1] log-mel {mel.shape} tu {speech_s:.2f}s tieng noi     {t_mel*1000:7.0f} ms")

    # -- 2. encoder, once -------------------------------------------------
    t0 = time.time()
    cross = runner.run(os.path.join(args.models, "encoder.bin"),
                       {"input_features": mel_path},
                       os.path.join(args.workdir, "enc_out"), "enc")
    t_enc = time.time() - t0
    print(f"[2] encoder -> {len(cross)} tensor cross-attention        {t_enc*1000:7.0f} ms")

    # -- 3. greedy decode -------------------------------------------------
    prompt = [SOT, tok.language_token(args.lang), TRANSCRIBE, NOTIMESTAMPS]
    generated: List[int] = []

    # Caches start at the value that dequantises to zero, which is the zero point --
    # not literal zero bytes.
    self_cache: Dict[str, str] = {}
    for layer in range(N_LAYERS):
        for kind, shape in (("k", (N_LAYERS, 1, 64, CACHE_LEN)),
                            ("v", (N_LAYERS, 1, CACHE_LEN, 64))):
            name = f"{kind}_cache_self_{layer}_in"
            _, czp = spec.q(spec.dec, "inputs", name)
            path = os.path.join(args.workdir, f"{name}.raw")
            np.full(shape, czp, dtype=np.uint8).tofile(path)
            self_cache[name] = path

    mask_scale, mask_zp = spec.q(spec.dec, "inputs", "attention_mask")
    ids_path = os.path.join(args.workdir, "input_ids.raw")
    pos_path = os.path.join(args.workdir, "position_ids.raw")
    mask_path = os.path.join(args.workdir, "attention_mask.raw")

    logits_scale, logits_zp = spec.q(spec.dec, "outputs", "logits")
    decoder = os.path.join(args.models, "decoder.bin")

    t0 = time.time()
    steps = 0
    for step in range(len(prompt) + args.max_tokens):
        token = prompt[step] if step < len(prompt) else generated[-1]
        np.array([[token]], dtype=np.int32).tofile(ids_path)
        np.array([step], dtype=np.int32).tofile(pos_path)

        # The cache fills from the right, so the newest `step` slots are the valid
        # ones; everything before them must be masked out.
        mask = np.full((1, 1, 1, MASK_LEN), -100.0, dtype=np.float32)
        mask[..., CACHE_LEN - step:] = 0.0
        quantize(mask, mask_scale, mask_zp, np.uint16).tofile(mask_path)

        inputs = {"input_ids": ids_path, "position_ids": pos_path,
                  "attention_mask": mask_path}
        inputs.update(self_cache)
        for layer in range(N_LAYERS):
            for kind in ("k", "v"):
                name = f"{kind}_cache_cross_{layer}"
                inputs[name] = cross[name]

        out = runner.run(decoder, inputs,
                         os.path.join(args.workdir, "dec_out"), f"dec{step}")
        steps += 1

        # Output caches become the next step's input caches, byte for byte.
        for layer in range(N_LAYERS):
            for kind in ("k", "v"):
                src = out[f"{kind}_cache_self_{layer}_out"]
                dst = os.path.join(args.workdir, f"{kind}_cache_self_{layer}_in.raw")
                shutil.copyfile(src, dst)
                self_cache[f"{kind}_cache_self_{layer}_in"] = dst

        if step < len(prompt) - 1:
            continue

        logits = np.fromfile(out["logits"], dtype=np.uint16).astype(np.int32)
        nxt = int(np.argmax(logits))          # argmax is unaffected by the affine scale
        if nxt == EOT:
            break
        generated.append(nxt)
        print(f"    token {len(generated):2d}: {nxt:<6} "
              f"{tok.decode([nxt])!r}", flush=True)

    t_dec = time.time() - t0
    per_step = t_dec / max(steps, 1)
    print(f"[3] decoder {steps} buoc                          {t_dec*1000:7.0f} ms "
          f"({per_step*1000:.0f} ms/buoc)")

    text = tok.decode(generated).strip()
    print("\n" + "=" * 60)
    print(f"KET QUA: {text}")
    print("=" * 60)
    print(f"tong: {(t_mel + t_enc + t_dec):.2f}s cho {speech_s:.2f}s tieng noi")

    if not args.keep_mel and os.path.exists(mel_path):
        os.remove(mel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
