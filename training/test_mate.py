"""Check that mate scores shrink with distance, so the engine prefers the fastest mate.

    uv run python training/test_mate.py

Not part of the submission (only root *.py and weights/ ship).
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent


def _is_mate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


FENS = {
    "back-rank mate in 1": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
    "queen mate in 1": "6k1/5ppp/8/8/8/8/8/4K2Q w - - 0 1",
}

for label, fen in FENS.items():
    board = chess.Board(fen)
    mates = [m for m in board.legal_moves if _is_mate(board, m)]
    picked = agent.get_move(fen, 60_000)
    ok = chess.Move.from_uci(picked) in mates
    print(f"{label}: mates={[m.uci() for m in mates]}  played={picked}  {'OK' if ok else 'MISS'}")

# A mate found nearer the root must score strictly higher than the same mate found deeper.
board = chess.Board(FENS["back-rank mate in 1"])
agent._deadline = time.perf_counter() + 60.0
agent._timed_out = False
scores = {}
for depth in (1, 3, 5):
    agent._timed_out = False
    scores[depth] = agent.negamax(chess.Board(board.fen()), depth, -1e9, 1e9, 0)
print("\nroot score by search depth (mate value should stay pinned to the shortest line):")
for depth, score in scores.items():
    print(f"  depth {depth}: {score:12,.0f}   distance from MATE = {agent.MATE - score:.0f} ply")

# Direct check of the ply adjustment itself.
mate_now = -agent.MATE + 0
mate_later = -agent.MATE + 4
print(f"\nply adjustment: mate at ply 0 = {mate_now:,}, at ply 4 = {mate_later:,}")
print(f"nearer mate ranks higher when negated: {-mate_now:,} > {-mate_later:,} "
      f"-> {-mate_now > -mate_later}")
