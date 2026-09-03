"""Sample the Lichess eval dump into a compact (codes, score) npz.

Run from the repo root:  uv run python training/download.py --limit 2000000

Scores are centipawns from White's point of view (the dataset's convention), clamped to
+/- --clamp, mates mapped to the ceiling. A material-vs-score correlation is printed as a
sign-convention check. Nothing but the npz touches disk.

Two sampling modes, and the default matters:

  scatter (default) -- the dump is a sequence of independent zstd frames and the server
    honours range requests, so we pull --chunks slices from byte offsets spread across the
    whole 21.7 GB and reservoir-sample those. Every region of the file is represented, at a
    few hundred MB of transfer instead of 21.7 GB.

  stream -- decompress the file end to end. Statistically the same population, but at the
    ~0.8 MB/s this server gives, a full pass is ~8 hours. Kept for verification, not routine
    use.

The thing NOT to do is what this script used to do: keep the first N usable lines. The dump
is not shuffled, so a prefix is a biased slice and raising --limit only buys more of it.
"""

import argparse
import io
import json
import random
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import board_to_codes, white_cp_to_mover

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MAGIC = b"\x28\xb5\x2f\xfd"  # zstd frame header; frames land every few MB in this dump
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


def parse_line(line: str, min_depth: int, clamp: int) -> tuple[np.ndarray, int] | None:
    """(codes, mover-POV centipawns) for one dump line, or None if unusable."""
    try:
        obj = json.loads(line)
        best = max(obj["evals"], key=lambda e: e.get("depth", 0))
        if best.get("depth", 0) < min_depth:
            return None
        white_cp = score_from_eval(best["pvs"], clamp)
        if white_cp is None:
            return None
        fen = obj["fen"]
        if fen.count(" ") == 3:  # dataset omits the move counters
            fen += " 0 1"
        board = chess.Board(fen)
    except (KeyError, ValueError, IndexError):
        return None
    return board_to_codes(board), white_cp_to_mover(white_cp, board)


def fetch_range(start: int, length: int, attempts: int = 4) -> bytes:
    """Fetch one byte range, retrying transient network failures.

    Retries matter more than they look: a chunk that gives up is a whole stratum of the file
    missing from the sample, and the run still ends "successfully" with a quietly biased npz.
    A DNS outage mid-run once cost 39 of 64 strata that way.
    """
    last: OSError | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                URL,
                headers={"User-Agent": "chessathon",
                         "Range": f"bytes={start}-{start + length - 1}"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data: bytes = resp.read()
                return data
        except OSError as exc:  # URLError and socket timeouts are both OSError
            last = exc
            if attempt < attempts - 1:
                back_off = 5 * 2**attempt
                print(f"    fetch retry {attempt + 1}/{attempts - 1} in {back_off}s ({exc})",
                      flush=True)
                time.sleep(back_off)
    raise last if last else OSError("fetch failed")


def file_size() -> int:
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "chessathon"})
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers["Content-Length"])


def scatter_lines(chunks: int, chunk_bytes: int, rng: random.Random) -> Iterator[str]:
    """Yield lines from `chunks` slices taken at random offsets across the whole file."""
    total = file_size()
    print(f"file {total / 1e9:.1f} GB; pulling {chunks} x {chunk_bytes / 1e6:.0f} MB "
          f"= {chunks * chunk_bytes / 1e6:.0f} MB from across it", flush=True)
    # Offsets spread over the file, one per equal-width stratum so the slices cannot bunch up.
    stratum = (total - chunk_bytes) // chunks
    offsets = [i * stratum + rng.randrange(max(1, stratum)) for i in range(chunks)]

    failed: list[int] = []
    for k, start in enumerate(offsets, 1):
        try:
            raw = fetch_range(start, chunk_bytes)
        except OSError as exc:
            print(f"  chunk {k}: fetch failed after retries ({exc})", flush=True)
            failed.append(k)
            continue
        frame = raw.find(MAGIC)
        if frame == -1:
            print(f"  chunk {k}: no frame boundary in slice, skipping", flush=True)
            failed.append(k)
            continue
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw[frame:]))
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            text.readline()  # first line is a fragment of whatever preceded the frame
            yield from text
        except (zstandard.ZstdError, EOFError):
            pass  # slices end mid-frame by construction; take what decoded

    if failed:
        lost = len(failed) / chunks * 100
        print(f"\n*** COVERAGE WARNING: {len(failed)}/{chunks} strata missing ({lost:.0f}% of "
              f"the file unsampled): chunks {failed}", flush=True)
        print("*** Each stratum is a contiguous region, so the sample is biased toward the "
              "regions that did load. Re-run before training anything you intend to ship.",
              flush=True)


def stream_lines() -> Iterator[str]:
    """Yield every line by decompressing the file end to end (~8 hours on this link)."""
    req = urllib.request.Request(URL, headers={"User-Agent": "chessathon"})
    with urllib.request.urlopen(req) as resp:
        reader = zstandard.ZstdDecompressor().stream_reader(resp)
        yield from io.TextIOWrapper(reader, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the Lichess eval dump.")
    parser.add_argument("--limit", type=int, default=1_000_000, help="reservoir size")
    parser.add_argument("--mode", choices=("scatter", "stream"), default="scatter")
    parser.add_argument("--chunks", type=int, default=96, help="scatter: slices to pull")
    parser.add_argument("--chunk-mb", type=int, default=8,
                        help="scatter: MB per slice; must exceed the frame spacing (~7 MB)")
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--clamp", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("training/data.npz"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # stdlib random, not numpy: this draws once per usable line, and random() costs ~40ns
    # against ~500ns for numpy's generator.
    rng = random.Random(args.seed)
    codes = np.zeros((args.limit, 64), dtype=np.int8)
    scores = np.zeros(args.limit, dtype=np.int16)
    mat_sum = np.zeros(args.limit, dtype=np.int32)
    n = 0  # reservoir slots filled
    usable = 0  # positions that passed the filters -- the population being sampled
    seen = 0
    t0 = time.monotonic()

    if args.mode == "scatter":
        lines = scatter_lines(args.chunks, args.chunk_mb * 1024 * 1024, rng)
    else:
        lines = stream_lines()

    for line in lines:
        seen += 1
        parsed = parse_line(line, args.min_depth, args.clamp)
        if parsed is None:
            continue
        c, mover_cp = parsed

        # Reservoir: fill the slots, then replace a random slot with probability
        # limit/usable, so every usable position ends up equally likely to be kept.
        slot = -1
        if n < args.limit:
            slot = n
            n += 1
        else:
            j = rng.randrange(usable + 1)
            if j < args.limit:
                slot = j
        usable += 1
        if slot >= 0:
            codes[slot] = c
            scores[slot] = mover_cp
            mat_sum[slot] = material(c)

        if usable % 250_000 == 0:
            print(f"  seen {seen:,}  usable {usable:,}  held {n:,}  "
                  f"({seen / (time.monotonic() - t0):,.0f} lines/s)", flush=True)

    codes, scores, mat_sum = codes[:n], scores[:n], mat_sum[:n]

    # Sign-convention check: material and White-POV score should correlate positively.
    corr = float(np.corrcoef(mat_sum.astype(np.float64), scores.astype(np.float64))[0, 1])
    print(f"\nkept {n:,} of {usable:,} usable from {seen:,} lines "
          f"in {time.monotonic() - t0:.0f}s")
    if n < args.limit:
        print(f"note: reservoir underfilled ({n:,} < {args.limit:,}) -- raise --chunks "
              f"(each ~{args.chunk_mb} MB slice yields roughly "
              f"{usable // max(1, args.chunks):,} usable positions).")
    print(f"score mean {scores.mean():.1f}  std {scores.std():.1f}  "
          f"min {scores.min()}  max {scores.max()}")
    print(f"material/score correlation {corr:+.3f}  "
          f"({'OK, White POV' if corr > 0 else 'WARNING: sign looks flipped'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, codes=codes, scores=scores)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
