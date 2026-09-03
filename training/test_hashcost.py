"""How expensive is a position key? A transposition table is only worth it if the key is cheap.

    uv run python training/test_hashcost.py

The eval is now ~31us end to end. A key costing 20us would eat most of what a TT saves, so
measure before building. Not part of the submission.
"""

import sys
import time
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalnet
from features import board_to_codes

N = 20_000
rng = np.random.default_rng(0)
board = chess.Board()
for _ in range(20):
    moves = list(board.legal_moves)
    board.push(moves[rng.integers(len(moves))])


def bench(label: str, fn: object, n: int = N) -> float:
    call = fn  # type: ignore[assignment]
    t0 = time.perf_counter()
    for _ in range(n):
        call()  # type: ignore[operator]
    dt = (time.perf_counter() - t0) / n * 1e6
    print(f"{label:<34}{dt:7.2f} us/call")
    return dt


print(f"{N:,} calls on a real middlegame position\n")
bench("chess.polyglot.zobrist_hash", lambda: chess.polyglot.zobrist_hash(board))
bench("board._transposition_key()", lambda: board._transposition_key())
bench("hash(board._transposition_key())", lambda: hash(board._transposition_key()))
bench("board.fen()", lambda: board.fen())
print()
bench("board_to_codes (feature encode)", lambda: board_to_codes(board))
codes = board_to_codes(board)
bench("evalnet.forward (the net itself)", lambda: evalnet.forward(codes))
bench("full evaluate path", lambda: evalnet.forward(board_to_codes(board)))
print()
bench("list(board.legal_moves)", lambda: list(board.legal_moves), 5_000)

print("\nA key is worth it only if it costs well under the eval it lets us skip.")
