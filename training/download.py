"""Stream the Lichess eval dump and save a compact (codes, score) sample.

The raw file is ~22 GB compressed and never touches disk: we stream, decompress, sample
the first --limit usable positions, and write only the packed npz. Scores are centipawns
from White's point of view (the dataset's convention), clamped to +/- --clamp. Mates map
to the clamp ceiling. A material-vs-score correlation is printed as a sign-convention check.

Run from the repo root:  uv run python training/download.py --limit 1000000
"""

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import chess
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import board_to_codes, white_cp_to_mover

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
_MATERIAL = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int64)  # by piece_type


def score_from_eval(pvs: list[dict], clamp: int) -> int | None:
    """Centipawns from White's POV for the principal variation, or None if unusable."""
    pv = pvs[0]
    if "cp" in pv:
        return int(np.clip(pv["cp"], -clamp, clamp))
    if "mate" in pv:
        mate = pv["mate"]
        if mate == 0:
            return None
        return clamp if mate > 0 else -clamp
    return None


def material(codes: np.ndarray) -> int:
    total = 0
    for c in codes:
        if c == 0:
            continue
        pt = c if c <= 6 else c - 6
        total += _MATERIAL[pt] if c <= 6 else -_MATERIAL[pt]
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the Lichess eval dump.")
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--clamp", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("training/data.npz"))
    args = parser.parse_args()

    codes = np.zeros((args.limit, 64), dtype=np.int8)
    scores = np.zeros(args.limit, dtype=np.int16)
    mat_sum = np.zeros(args.limit, dtype=np.int32)
    n = 0
    seen = 0
    t0 = time.monotonic()

    req = urllib.request.Request(URL, headers={"User-Agent": "chessathon"})
    with urllib.request.urlopen(req) as resp:
        reader = zstandard.ZstdDecompressor().stream_reader(resp)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            seen += 1
            try:
                obj = json.loads(line)
                evals = obj["evals"]
                best = max(evals, key=lambda e: e.get("depth", 0))
                if best.get("depth", 0) < args.min_depth:
                    continue
                white_cp = score_from_eval(best["pvs"], args.clamp)
                if white_cp is None:
                    continue
                fen = obj["fen"]
                if fen.count(" ") == 3:  # dataset omits the move counters
                    fen += " 0 1"
                board = chess.Board(fen)
            except (KeyError, ValueError, IndexError):
                continue

            c = board_to_codes(board)
            codes[n] = c
            scores[n] = white_cp_to_mover(white_cp, board)
            mat_sum[n] = material(c)
            n += 1
            if n % 100_000 == 0:
                rate = n / (time.monotonic() - t0)
                print(f"  kept {n:,} / seen {seen:,}  ({rate:,.0f}/s)", flush=True)
            if n >= args.limit:
                break

    codes = codes[:n]
    scores = scores[:n]
    mat_sum = mat_sum[:n]

    # Sign-convention check: material and White-POV score should correlate positively.
    corr = float(np.corrcoef(mat_sum.astype(np.float64), scores.astype(np.float64))[0, 1])
    print(f"\nkept {n:,} positions from {seen:,} lines in {time.monotonic() - t0:.0f}s")
    print(f"score mean {scores.mean():.1f}  std {scores.std():.1f}  "
          f"min {scores.min()}  max {scores.max()}")
    print(f"material/score correlation {corr:+.3f}  "
          f"({'OK, White POV' if corr > 0 else 'WARNING: sign looks flipped'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, codes=codes, scores=scores)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
