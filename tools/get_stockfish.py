"""Fetch a Stockfish binary into ``engine/``.

The binary is not committed: it is ~80 MB, platform-specific, and GPL-3 with its
own source obligations.  This script downloads the right official build for the
machine it runs on, or copies one you already have.

    python tools/get_stockfish.py                 # download for this platform
    python tools/get_stockfish.py --from PATH     # copy a binary you already have
    python tools/get_stockfish.py --check         # just say what would be used

Assets are resolved through the GitHub releases API rather than a hardcoded URL,
because the tag moves (sf_17 -> sf_18 -> ...) and a pinned link rots silently.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engine"
API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
RELEASES = "https://github.com/official-stockfish/Stockfish/releases"

#: Asset name fragments per platform, best instruction set first.  A CPU without
#: AVX2 will not run the avx2 build at all, so the safe sse41-popcnt build is the
#: last entry and every 64-bit x86 chip since ~2008 can execute it.
PREFERENCES = {
    "win32": [
        "stockfish-windows-x86-64-avx2.zip",
        "stockfish-windows-x86-64-sse41-popcnt.zip",
        "stockfish-windows-x86-64.zip",
    ],
    "linux": [
        "stockfish-ubuntu-x86-64-avx2.tar",
        "stockfish-ubuntu-x86-64-sse41-popcnt.tar",
        "stockfish-ubuntu-x86-64.tar",
    ],
    "darwin": [
        "stockfish-macos-m1-apple-silicon.tar",
        "stockfish-macos-x86-64-avx2.tar",
        "stockfish-macos-x86-64-sse41-popcnt.tar",
    ],
}


def platform_key() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return "win32"


def wanted_assets() -> list:
    key = platform_key()
    names = list(PREFERENCES[key])
    if key == "darwin" and platform.machine() not in ("arm64", "aarch64"):
        names = [n for n in names if "m1-apple" not in n]
    if key == "linux" and platform.machine() in ("aarch64", "arm64"):
        # No official ARM Linux build ships in the release assets.
        return []
    return names


def latest_release() -> dict:
    req = urllib.request.Request(API, headers={"User-Agent": "AI-ChessMate"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AI-ChessMate"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunks, seen = [], 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            seen += len(chunk)
            if total:
                pct = 100 * seen / total
                print(f"\r  {seen >> 20} / {total >> 20} MB ({pct:.0f}%)", end="")
        print()
        return b"".join(chunks)


def extract(blob: bytes, asset: str) -> Path:
    """Pull the executable out of the archive into ``engine/``.

    The archives contain a ``stockfish/`` directory with the binary plus source
    and docs; only the executable is kept.
    """
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    def is_binary(name: str) -> bool:
        base = os.path.basename(name)
        if not base.lower().startswith("stockfish"):
            return False
        return base.lower().endswith(".exe") or "." not in base

    if asset.endswith(".zip"):
        archive = zipfile.ZipFile(io.BytesIO(blob))
        members = [m for m in archive.namelist() if is_binary(m)]
        if not members:
            raise RuntimeError(f"no executable inside {asset}")
        target = ENGINE_DIR / os.path.basename(members[0])
        with archive.open(members[0]) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        archive = tarfile.open(fileobj=io.BytesIO(blob))
        members = [m for m in archive.getmembers() if m.isfile() and is_binary(m.name)]
        if not members:
            raise RuntimeError(f"no executable inside {asset}")
        target = ENGINE_DIR / os.path.basename(members[0].name)
        src = archive.extractfile(members[0])
        if src is None:
            raise RuntimeError(f"could not read {members[0].name}")
        with open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)

    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def verify(path: Path) -> str:
    """Start it and read the UCI id, so a bad download fails here, not in the UI."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    try:
        return engine.id.get("name", "unknown")
    finally:
        engine.quit()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Stockfish into engine/")
    parser.add_argument("--from", dest="source", help="copy this binary instead of downloading")
    parser.add_argument("--check", action="store_true", help="report what would be used and exit")
    parser.add_argument("--force", action="store_true", help="download even if one is present")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from digital_board.engine import find_binary  # noqa: E402 - needs ROOT on the path

    existing = find_binary()
    if args.check:
        print(existing or "no Stockfish found")
        return 0 if existing else 1

    if args.source:
        src = Path(args.source).expanduser()
        if not src.is_file():
            print(f"no such file: {src}", file=sys.stderr)
            return 1
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        dst = ENGINE_DIR / src.name
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR)
        print(f"copied -> {dst}")
        print(f"verified: {verify(dst)}")
        return 0

    if existing and not args.force:
        print(f"already available: {existing}")
        print("pass --force to download a fresh copy anyway")
        return 0

    names = wanted_assets()
    if not names:
        print(
            f"no official build for {sys.platform}/{platform.machine()}.\n"
            f"Install one from your package manager (apt install stockfish) or\n"
            f"build from source, then set STOCKFISH_PATH.",
            file=sys.stderr,
        )
        return 1

    try:
        release = latest_release()
    except urllib.error.URLError as exc:
        print(f"could not reach GitHub ({exc}).", file=sys.stderr)
        print(f"Download manually from {RELEASES} and unzip into engine/", file=sys.stderr)
        return 1

    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    choice = next((n for n in names if n in assets), None)
    if choice is None:
        print(f"none of {names} in release {release.get('tag_name')}", file=sys.stderr)
        return 1

    print(f"Stockfish {release.get('tag_name')} -- {choice}")
    blob = download(assets[choice])
    target = extract(blob, choice)
    print(f"installed -> {target}")
    print(f"verified: {verify(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
