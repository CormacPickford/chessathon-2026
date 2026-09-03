"""Can we decode the eval dump from a mid-file byte offset?

    uv run python training/test_seek.py

A full sequential pass costs ~8h at the measured 0.8 MB/s. If the dump is a sequence of
independent zstd frames, range requests let us pull scattered chunks from across the whole
file instead, which samples the same population for a fraction of the bytes. A single-frame
file cannot be decoded from an arbitrary offset and this reports that. Not part of the
submission.
"""

import io
import json
import urllib.request

import zstandard

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MAGIC = b"\x28\xb5\x2f\xfd"  # zstd frame header
SKIPPABLE = b"\x2a\x4d\x18"  # skippable frame magic suffix (0x184D2A5?)
PROBE = 8 * 1024 * 1024


def fetch(start: int, length: int) -> bytes:
    req = urllib.request.Request(
        URL,
        headers={"User-Agent": "chessathon", "Range": f"bytes={start}-{start + length - 1}"},
    )
    with urllib.request.urlopen(req) as resp:
        data: bytes = resp.read()
        return data


req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "chessathon"})
with urllib.request.urlopen(req) as resp:
    total = int(resp.headers["Content-Length"])
print(f"file: {total / 1e9:.1f} GB")

for frac in (0.25, 0.50, 0.75):
    start = int(total * frac)
    chunk = fetch(start, PROBE)
    offsets = []
    pos = chunk.find(MAGIC)
    while pos != -1 and len(offsets) < 5:
        offsets.append(pos)
        pos = chunk.find(MAGIC, pos + 1)

    print(f"\n=== offset {frac:.0%} ({start:,}) ===")
    print(f"  frame magic found at: {offsets if offsets else 'NONE'}")
    if not offsets:
        print("  -> no independent frame here: single-frame file, cannot seek")
        continue

    ok = False
    for off in offsets:
        try:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(io.BytesIO(chunk[off:])) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                first = text.readline()
                second = text.readline()
            obj = json.loads(second)
            print(f"  -> decoded from +{off}: fen={obj.get('fen', '?')[:40]}")
            print(f"     (first partial line discarded, {len(first)}B)")
            ok = True
            break
        except (zstandard.ZstdError, ValueError, EOFError) as exc:
            print(f"  -> +{off} failed: {type(exc).__name__}")
    if not ok:
        print("  -> magic bytes were coincidental data, not real frame starts")
