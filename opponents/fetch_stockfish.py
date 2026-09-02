"""Download the official Stockfish Windows binary for LOCAL calibration only.

    uv run python opponents/fetch_stockfish.py

Stockfish is a measuring stick here, never part of the submission: it lives under opponents/,
which the packager does not include (only root *.py and weights/ ship). Using an engine to
measure or annotate is allowed; shipping or running one inside agent.py is an instant DQ.
"""

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
# Preference order: broad compatibility first, then faster micro-arch builds.
ARCH_PREFERENCE = ["x86-64-sse41-popcnt", "x86-64-avx2", "x86-64-ssse3", "x86-64.zip"]
DEST = Path(__file__).resolve().parent / "engines"


def pick_asset(assets: list[dict]) -> dict:
    windows = [a for a in assets if "windows" in a["name"] and a["name"].endswith(".zip")]
    for arch in ARCH_PREFERENCE:
        for asset in windows:
            if arch in asset["name"]:
                return asset
    if windows:
        return windows[0]
    raise SystemExit("no Windows Stockfish asset found in the latest release")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(API, headers={"User-Agent": "chessathon"})
    with urllib.request.urlopen(req) as resp:
        release = json.load(resp)
    asset = pick_asset(release["assets"])
    print(f"release {release['tag_name']}  asset {asset['name']} "
          f"({asset['size'] / 1e6:.0f} MB)")

    zip_path = DEST / "stockfish.zip"
    req = urllib.request.Request(
        asset["browser_download_url"], headers={"User-Agent": "chessathon"}
    )
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
        out.write(resp.read())

    with zipfile.ZipFile(zip_path) as zf:
        exe_name = next((n for n in zf.namelist() if n.endswith(".exe")), None)
        if exe_name is None:
            raise SystemExit("no .exe inside the Stockfish zip")
        zf.extract(exe_name, DEST)

    exe_path = DEST / exe_name
    final = DEST / "stockfish.exe"
    if final.exists():
        final.unlink()
    exe_path.replace(final)
    zip_path.unlink()
    print(f"stockfish binary -> {final}")

    # Confirm it speaks UCI.
    import chess.engine
    with chess.engine.SimpleEngine.popen_uci(str(final)) as engine:
        print(f"handshake OK: {engine.id.get('name', '?')}")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
