"""NGHE (ASR offline) — nhận lệnh giọng nói tiếng Việt bằng Vosk.
Thu âm từ mic (arecord) rồi chuyển thành chữ, đưa cho trợ lý/coach xử lý.
Dùng để ĐỐI THOẠI RẢNH TAY với chuyên gia cờ vua.

Lưu ý: cần test với GIỌNG NGƯỜI THẬT (tự test bằng giọng espeak máy móc có thể ra sai)."""
from __future__ import annotations
import audioop
import json
import os
import subprocess
import tempfile
import wave

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    _HAS_VOSK = True
except Exception:
    _HAS_VOSK = False

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..",
                             "models_asr", "vosk-model-small-vn-0.4")


class Listener:
    def __init__(self, model_path: str | None = None, samplerate: int = 16000):
        if not _HAS_VOSK:
            raise RuntimeError("Chưa cài vosk. Chạy: pip3 install vosk")
        self.samplerate = samplerate
        path = os.path.abspath(model_path or DEFAULT_MODEL)
        if not os.path.isdir(path):
            raise RuntimeError(f"Không thấy model ASR ở {path}")
        self.model = Model(path)

    def _read_wav_16k_mono(self, path: str) -> bytes:
        wf = wave.open(path, "rb")
        data = wf.readframes(wf.getnframes())
        sw, ch, rate = wf.getsampwidth(), wf.getnchannels(), wf.getframerate()
        wf.close()
        if sw != 2:
            data = audioop.lin2lin(data, sw, 2)
        if ch == 2:
            data = audioop.tomono(data, 2, 0.5, 0.5)
        if rate != self.samplerate:
            data, _ = audioop.ratecv(data, 2, 1, rate, self.samplerate, None)
        return data

    def transcribe_wav(self, path: str) -> str:
        rec = KaldiRecognizer(self.model, self.samplerate)
        data = self._read_wav_16k_mono(path)
        rec.AcceptWaveform(data)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    def record(self, seconds: float = 4.0, path: str | None = None) -> str:
        path = path or os.path.join(tempfile.gettempdir(), "asr_in.wav")
        subprocess.run(
            ["arecord", "-q", "-f", "S16_LE", "-r", str(self.samplerate),
             "-c", "1", "-d", str(seconds), path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return path

    def listen_once(self, seconds: float = 4.0) -> str:
        """Thu âm `seconds` giây từ mic rồi trả về câu nhận diện."""
        return self.transcribe_wav(self.record(seconds))


def _selftest():
    """Tự test: espeak đọc vài câu lệnh -> Vosk nhận lại (giọng máy nên có thể sai)."""
    import shutil
    l = Listener()
    tmp = tempfile.gettempdir()
    phrases = ["đánh giá thế cờ", "nên đi nước nào", "có gì nguy hiểm không"]
    print("Tự test ASR (đầu vào là giọng espeak máy móc):")
    for p in phrases:
        wav = os.path.join(tmp, "tts_probe.wav")
        subprocess.run(["espeak-ng", "-v", "vi", "-s", "130", p, "-w", wav],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = l.transcribe_wav(wav)
        print(f"   nói: '{p}'  ->  nghe: '{got}'")


if __name__ == "__main__":
    _selftest()
