"""Stream a tiny slice of the Lichess eval dump to confirm the URL and schema."""

import io
import json
import urllib.request

import zstandard

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "chessathon-probe"})
    with urllib.request.urlopen(req) as resp:
        print("HTTP", resp.status, resp.headers.get("Content-Length"), "bytes total")
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(resp)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for i, line in enumerate(text):
            if i >= 3:
                break
            obj = json.loads(line)
            print(f"\n--- line {i} keys: {list(obj.keys())} ---")
            print("fen:", obj["fen"])
            evals = obj["evals"]
            print("num evals:", len(evals))
            best = max(evals, key=lambda e: e.get("depth", 0))
            print("deepest eval depth:", best.get("depth"), "keys:", list(best.keys()))
            print("first pv:", best["pvs"][0])


if __name__ == "__main__":
    main()
