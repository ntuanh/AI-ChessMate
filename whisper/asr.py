#!/usr/bin/env python3
"""Speech to text on the NPU, fast enough to actually use.

Keeps one `whisper_worker` process alive with both context binaries resident, so a
request costs an encoder pass plus a few milliseconds per token instead of reloading
225 MB from disk for every single token.

    from asr import Recognizer
    with Recognizer() as asr:
        print(asr.listen(4))          # record from the C270 and transcribe
        print(asr.transcribe_wav(p))  # or transcribe a file

Command line:
    python3 asr.py            # record 4 s, print the text, exit
    python3 asr.py --loop     # keep listening; the worker is started once
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import sys
import time
import wave
from typing import List, Optional

import numpy as np

import whisper_npu as W

HERE = os.path.dirname(os.path.abspath(__file__))
MIC_DEVICE = "plughw:0,0"

# OpenAI Whisper's own hallucination thresholds.
NO_SPEECH_MAX = 0.6
LOGPROB_MIN = -1.0
COMPRESSION_MAX = 2.4

# Phrases Whisper emits when it hears nothing it can transcribe: YouTube subtitle
# boilerplate straight out of the training data.  Observed repeatedly on this board
# from room noise alone, at signal levels well above the silence threshold.
BOILERPLATE = (
    "subscribe", "ghien mi go", "dang ky kenh", "don xem video",
    "cam on cac ban da", "hen gap lai", "chuc cac ban", "phu de",
)


def strip_diacritics_simple(text: str) -> str:
    import unicodedata
    text = text.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def _compression_ratio(text: str) -> float:
    """Repetitive text compresses well; Whisper loops ('tên tên tên') score high."""
    import zlib
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))

# Falling back to these means the script runs without sourcing env.sh first.
# The flavor is not a free choice: aarch64-ubuntu-gcc9.4 ships only libQnnHtpV68Stub.so
# and this chip is Hexagon V73, which fails as a misleading "Unsupported SoC model".
DEFAULT_SDK = "/opt/qcom/qairt-new/qairt/2.48.0.260626"
DEFAULT_FLAVOR = "aarch64-oe-linux-gcc11.2"
HEXAGON_SKELS = "lib/hexagon-v73/unsigned"


def _resolve_sdk() -> tuple:
    """SDK path and flavor, from the environment or the known-good defaults."""
    sdk = os.environ.get("SDK") or DEFAULT_SDK
    flavor = os.environ.get("FLAVOR") or DEFAULT_FLAVOR
    if not os.path.isdir(os.path.join(sdk, "lib", flavor)):
        raise WorkerError(
            f"khong thay {sdk}/lib/{flavor} — hay source env.sh hoac dat SDK/FLAVOR")

    # ADSP_LIBRARY_PATH is *overwritten*, never defaulted.  Inheriting someone else's
    # value is worse than having none: root's .bashrc sources /opt/qcom/qirp-sdk, whose
    # Hexagon skeletons are QAIRT 2.28 and cannot serve these 2.45 context binaries --
    # the graph then fails with a bare "contextCreateFromBinary failed".
    skels = os.path.join(sdk, HEXAGON_SKELS)
    stale = os.environ.get("ADSP_LIBRARY_PATH")
    if stale and stale != skels:
        print(f"asr: bo qua ADSP_LIBRARY_PATH cu ({stale.split(':')[0]}...), "
              f"dung {skels}", file=sys.stderr)
    os.environ["ADSP_LIBRARY_PATH"] = skels

    # Our libraries must resolve first, ahead of any other SDK already on the path.
    libdir = os.path.join(sdk, "lib", flavor)
    others = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
              if p and p != libdir]
    os.environ["LD_LIBRARY_PATH"] = ":".join([libdir] + others)
    return sdk, flavor


class WorkerError(RuntimeError):
    pass


class Recognizer:
    """A persistent whisper_worker, plus the audio and tokenizer bits around it."""

    def __init__(self, models: str = HERE, lang: str = "vi",
                 worker: Optional[str] = None) -> None:
        sdk, flavor = _resolve_sdk()

        self.models = models
        # Scratch files go to a private temp directory, never next to the models.
        # Whoever runs this first would otherwise leave root-owned mel.raw / mic.wav
        # behind and the next user fails with Permission denied.
        self._scratch = tempfile.TemporaryDirectory(prefix="whisper-")
        self.mel_path = os.path.join(self._scratch.name, "mel.raw")
        self.wav_path = os.path.join(self._scratch.name, "mic.wav")
        self.tok = W.Tokenizer(os.path.join(models, "vocab.json"),
                               os.path.join(models, "added_tokens.json"))
        self.lang_token = self.tok.language_token(lang)

        meta = json.load(open(os.path.join(models, "metadata.json")))
        q = meta["model_files"]["encoder.bin"]["inputs"]["input_features"][
            "quantization_parameters"]
        self.mel_scale, self.mel_zp = q["scale"], q["zero_point"]

        cmd = [worker or os.path.join(models, "whisper_worker"),
               os.path.join(models, "encoder.bin"),
               os.path.join(models, "decoder.bin"),
               os.path.join(sdk, "lib", flavor)]

        if not os.access(cmd[0], os.X_OK):
            raise WorkerError(f"khong thay {cmd[0]} — chay ./build.sh truoc")

        # Keep the worker's stderr: when it dies during startup, that text is the only
        # thing that says why, and discarding it turns every failure into the same
        # useless "worker exited" message.  A temp file rather than a fixed path --
        # a log left in the model directory by one user blocks the next one with
        # Permission denied, which is a confusing way to learn about file ownership.
        self._log = tempfile.TemporaryFile(mode="w+")

        t0 = time.time()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1, env=os.environ.copy())
        # The fastRPC layer prints a banner to stdout before the worker says anything,
        # so read past the noise rather than treating the first line as the answer.
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise WorkerError("worker thoat khi dang khoi dong:\n" + self._tail())
            if line.strip() == "READY":
                break
        self.startup_s = time.time() - t0
        self.last_stats = {}

    def _tail(self, lines: int = 12) -> str:
        """Last few lines the worker wrote to stderr, minus the library's chatter."""
        try:
            self._log.flush()
            self._log.seek(0)
            noise = ("m_mutex", "rpcmem", "BufferAllocator")
            kept = [l.rstrip() for l in self._log
                    if l.strip() and not any(n in l for n in noise)]
            return "\n".join("  " + l for l in kept[-lines:]) or "  (khong co log)"
        except Exception:
            return "  (khong doc duoc worker.log)"

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def __enter__(self) -> "Recognizer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- recognition ------------------------------------------------------

    def transcribe_wav(self, path: str, max_tokens: int = 32) -> str:
        audio = W.read_wav(path)
        return self.transcribe_audio(audio, max_tokens)

    def transcribe_audio(self, audio: np.ndarray, max_tokens: int = 32) -> str:
        t0 = time.time()
        mel = W.log_mel(audio)
        W.quantize(mel[None], self.mel_scale, self.mel_zp, np.uint16).tofile(self.mel_path)
        mel_ms = (time.time() - t0) * 1000

        self.proc.stdin.write(f"{self.mel_path} {self.lang_token} {max_tokens}\n")
        self.proc.stdin.flush()

        tokens: List[int] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise WorkerError("worker thoat giua chung")
            line = line.strip()
            if line.startswith("TOK "):
                tokens.append(int(line[4:]))
            elif line.startswith("DONE "):
                parts = line[5:].split()
                enc_ms, dec_ms, steps = (int(x) for x in parts[:3])
                no_speech = float(parts[3]) if len(parts) > 3 else 0.0
                avg_logprob = float(parts[4]) if len(parts) > 4 else 0.0
                self.last_stats = {"mel_ms": mel_ms, "encoder_ms": enc_ms,
                                   "decoder_ms": dec_ms, "steps": steps,
                                   "total_ms": mel_ms + enc_ms + dec_ms,
                                   "no_speech": no_speech,
                                   "avg_logprob": avg_logprob}
                break
            elif line.startswith("ERR "):
                raise WorkerError(line[4:])

        text = self.tok.decode(tokens).strip()
        self.last_stats["compression"] = _compression_ratio(text)
        self.last_stats["reject"] = self._reject_reason(text)
        return "" if self.last_stats["reject"] else text

    def _reject_reason(self, text: str) -> str:
        """Why this transcription should be treated as 'no speech heard'.

        Thresholds are OpenAI Whisper's own defaults.  Rejecting is the whole point:
        a hallucinated sentence that reaches the chess layer gets acted on, whereas
        an empty string just asks the user to repeat themselves.
        """
        s = self.last_stats
        if not text:
            return "khong co chu nao"
        if s["no_speech"] > NO_SPEECH_MAX and s["avg_logprob"] < LOGPROB_MIN:
            return (f"khong phai tieng noi (no_speech={s['no_speech']:.2f}, "
                    f"logprob={s['avg_logprob']:.2f})")
        if s["avg_logprob"] < LOGPROB_MIN:
            return f"do tin cay qua thap (logprob={s['avg_logprob']:.2f})"
        if s["compression"] > COMPRESSION_MAX:
            return f"lap di lap lai (nen={s['compression']:.2f})"
        low = strip_diacritics_simple(text)
        if any(p in low for p in BOILERPLATE):
            return "cau mau phu de YouTube - dau hieu ao giac"
        return ""

    def listen(self, seconds: float = 4.0, max_tokens: int = 32) -> str:
        """Record from the C270 and transcribe.  Returns '' on a silent capture."""
        wav = self.wav_path
        subprocess.run(
            ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000",
             "-c", "1", "-d", str(int(seconds)), wav],
            check=True, capture_output=True)

        with wave.open(wav, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        peak = int(np.abs(pcm).max()) if len(pcm) else 0
        self.last_peak = peak
        if peak < 500:
            # Whisper answers silence with a fluent invented sentence, which reads
            # like a model failure rather than a missing signal.  Say nothing instead.
            return ""

        return self.transcribe_audio(pcm.astype(np.float32) / 32768.0, max_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--loop", action="store_true", help="nghe lien tuc")
    ap.add_argument("--wav", help="phien am mot file thay vi thu tu micro")
    args = ap.parse_args()

    with Recognizer(lang=args.lang) as asr:
        print(f"worker san sang sau {asr.startup_s:.2f}s", file=sys.stderr)

        def once() -> None:
            if args.wav:
                text = asr.transcribe_wav(args.wav, args.max_tokens)
            else:
                print(f">>> NOI DI ({args.seconds:.0f}s)...", file=sys.stderr)
                text = asr.listen(args.seconds, args.max_tokens)
            s = asr.last_stats
            if not text:
                print(f"(im lang, peak={getattr(asr, 'last_peak', 0)})", file=sys.stderr)
                return
            print(text)
            print(f"    mel {s['mel_ms']:.0f} ms · encoder {s['encoder_ms']} ms · "
                  f"decoder {s['decoder_ms']} ms / {s['steps']} buoc · "
                  f"TONG {s['total_ms']/1000:.2f}s", file=sys.stderr)

        once()
        while args.loop:
            try:
                input("\n[Enter de noi tiep, Ctrl-C de thoat]")
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
