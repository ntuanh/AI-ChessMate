"""Đọc to (Text-to-Speech).
Ưu tiên PIPER (giọng neural tự nhiên); nếu không có thì lùi về espeak-ng (nhiều giọng).
Có chế độ chỉ tạo file WAV để tự test không cần loa."""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile

from . import config


class Speaker:
    def __init__(self, settings):
        self.settings = settings
        self.has_espeak = shutil.which("espeak-ng") is not None
        self.has_aplay = shutil.which("aplay") is not None
        self.has_piper = (os.path.exists(config.PIPER_BIN)
                          and os.path.exists(config.PIPER_MODEL))

    def _engine(self) -> str:
        e = self.settings.tts_engine
        if e == "piper":
            return "piper" if self.has_piper else "espeak"
        if e == "espeak":
            return "espeak"
        # auto
        return "piper" if self.has_piper else "espeak"

    # ---------- Piper ----------
    def _synth_piper(self, text: str, path: str) -> bool:
        try:
            env = dict(os.environ, LD_LIBRARY_PATH=config.PIPER_LIB)
            subprocess.run(
                [config.PIPER_BIN, "--model", config.PIPER_MODEL,
                 "--length_scale", str(self.settings.piper_length),
                 "--output_file", path],
                input=text.encode("utf-8"), env=env, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False

    # ---------- espeak ----------
    def _voice_string(self, voice_name: str | None = None) -> str:
        name = voice_name or self.settings.voice
        return config.VOICES.get(name, config.VOICES.get("nam_chuan", "vi"))

    def _synth_espeak(self, text: str, path: str, voice_name: str | None = None) -> bool:
        if not self.has_espeak:
            return False
        try:
            subprocess.run(
                ["espeak-ng", "-v", self._voice_string(voice_name),
                 "-s", str(self.settings.speed), text, "-w", path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False

    # ---------- chung ----------
    def synth_to_wav(self, text: str, path: str, voice_name: str | None = None) -> bool:
        if self._engine() == "piper" and voice_name is None:
            if self._synth_piper(text, path):
                return True
        return self._synth_espeak(text, path, voice_name)

    def play_wav(self, path: str) -> bool:
        if not self.has_aplay:
            return False
        try:
            cmd = ["aplay", "-q"]
            if config.APLAY_DEVICE:
                cmd += ["-D", config.APLAY_DEVICE]
            cmd.append(path)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def say(self, text: str, save_wav: str | None = None, voice_name: str | None = None) -> None:
        if not self.settings.speak and not save_wav:
            return
        path = save_wav or os.path.join(tempfile.gettempdir(), "chess_tts.wav")
        if self.synth_to_wav(text, path, voice_name):
            if self.settings.speak:
                self.play_wav(path)

    def demo_all_voices(self, text: str, out_dir: str) -> dict:
        """Tạo file mẫu: 1 giọng Piper + các giọng espeak, để nghe thử."""
        os.makedirs(out_dir, exist_ok=True)
        made = {}
        if self.has_piper:
            p = os.path.join(out_dir, "voice_PIPER.wav")
            if self._synth_piper(text, p):
                made["PIPER"] = p
        for name in config.VOICES:
            p = os.path.join(out_dir, f"voice_{name}.wav")
            if self._synth_espeak(text, p, voice_name=name):
                made[name] = p
        return made
