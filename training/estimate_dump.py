"""Estimate the cost of a full pass over the Lichess eval dump before committing to one.

    uv run python training/estimate_dump.py

Reports the compressed size, whether the server supports range requests, the measured
decompression throughput, and the projected wall time for download.py to reach the end.
Not part of the submission.
"""

import io
import time
import urllib.request

import zstandard

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SAMPLE_COMPRESSED = 64 * 1024 * 1024  # compressed bytes to pull for the rate measurement

req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "chessathon"})
with urllib.request.urlopen(req) as resp:
    total = int(resp.headers.get("Content-Length", 0))
    ranges = resp.headers.get("Accept-Ranges", "none")
print(f"compressed size : {total / 1e9:.1f} GB")
print(f"accept-ranges   : {ranges}")

# Pull a slice and measure how much decompressed text and how many lines it yields.
req = urllib.request.Request(
    URL, headers={"User-Agent": "chessathon", "Range": f"bytes=0-{SAMPLE_COMPRESSED - 1}"}
)
t0 = time.monotonic()
raw = b""
with urllib.request.urlopen(req) as resp:
    raw = resp.read()
dl_s = time.monotonic() - t0
got = len(raw)

t0 = time.monotonic()
lines = 0
uncompressed = 0
dctx = zstandard.ZstdDecompressor()
try:
    with dctx.stream_reader(io.BytesIO(raw)) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        for line in text:
            lines += 1
            uncompressed += len(line)
except (zstandard.ZstdError, EOFError):
    pass  # truncated slice: expected, we only pulled a prefix
decomp_s = time.monotonic() - t0

ratio = uncompressed / got if got else 0
est_lines = total / got * lines if got else 0
dl_mbps = got / dl_s / 1e6
print(f"\nsampled         : {got / 1e6:.0f} MB compressed -> {lines:,} lines "
      f"({uncompressed / 1e6:.0f} MB text, {ratio:.1f}x)")
print(f"download rate   : {dl_mbps:.1f} MB/s  ({dl_s:.1f}s for the slice)")
print(f"decompress rate : {lines / decomp_s:,.0f} lines/s (no JSON parse, already in memory)")
print(f"\nestimated total : {est_lines / 1e6:,.0f}M lines")
print(f"full pass, network bound : {total / 1e6 / dl_mbps / 3600:.1f} h")
for label, rate in (("parse-rate 1.00", 5_700), ("parse-rate 0.02", 13_300)):
    print(f"full pass, {label}    : {est_lines / rate / 3600:.1f} h")
