"""The fast capture generator must produce EXACTLY the old move set, promotions included.

    uv run python training/test_noisy.py

Swapping `[m for m in legal_moves if is_capture(m) or promotion == QUEEN]` for
`generate_legal_captures()` is only a speed change if the move sets match. If they differ the
search changes too, and any Elo result would be measuring the wrong thing. Checked over random
play plus positions built specifically to have promotions available.
Not part of the submission.
"""

import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent


def old_noisy(board: chess.Board) -> set[chess.Move]:
    return {m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN}


# Positions where promotions are live -- the case generate_legal_captures() alone would miss.
PROMO_FENS = [
    "8/P7/8/8/8/8/8/K6k w - - 0 1",  # quiet promotion only
    "1n6/P7/8/8/8/8/8/K6k w - - 0 1",  # capture-promotion and quiet promotion
    "8/8/8/8/8/8/p7/K6k b - - 0 1",  # black quiet promotion
    "8/8/8/8/8/8/1p6/KN5k b - - 0 1",  # black capture-promotion
    "n1n5/PPP5/8/8/8/8/8/K6k w - - 0 1",  # several at once
]

bad = 0
for fen in PROMO_FENS:
    board = chess.Board(fen)
    old, new = old_noisy(board), set(agent._noisy_moves(board))
    ok = old == new
    bad += not ok
    print(f"{'OK ' if ok else 'DIFF'}  {fen}")
    if not ok:
        print(f"      missing: {[m.uci() for m in old - new]}")
        print(f"      extra  : {[m.uci() for m in new - old]}")

# And across ordinary play, where captures dominate.
rng = np.random.default_rng(0)
board = chess.Board()
checked = 0
for _ in range(4000):
    if board.is_game_over() or board.ply() > 140:
        board = chess.Board()
    old, new = old_noisy(board), set(agent._noisy_moves(board))
    if old != new:
        bad += 1
        print(f"DIFF at {board.fen()}")
        print(f"      missing: {[m.uci() for m in old - new]}")
        print(f"      extra  : {[m.uci() for m in new - old]}")
    checked += 1
    moves = list(board.legal_moves)
    board.push(moves[rng.integers(len(moves))])

print(f"\nchecked {checked:,} played positions + {len(PROMO_FENS)} promotion positions")
print("PASS - move sets identical" if bad == 0 else f"FAIL - {bad} mismatches")
