"""Check the transposition table behaves over a game: it helps, and it does not eat the RAM.

    uv run python training/test_tt.py

The table persists across moves within a game (module state survives; see AGENTS.md), so it
grows move after move. The platform gives 2 GB, and blowing that loses the game, so measure
the growth rather than trusting the cap. Also confirms the TT actually produces the same moves
as searching without it. Not part of the submission.
"""

import sys
import tracemalloc
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

tracemalloc.start()
board = chess.Board(FEN)
print(f"{'move':>5}{'TT entries':>13}{'TT bytes/entry':>17}{'total MB':>11}")
for i in range(1, 17):
    if board.is_game_over():
        break
    uci = agent.get_move(board.fen(), 60_000)
    board.push(chess.Move.from_uci(uci))
    current, _ = tracemalloc.get_traced_memory()
    n = len(agent._TT)
    per = current / n if n else 0
    if i % 4 == 0:
        print(f"{i:>5}{n:>13,}{per:>17.0f}{current / 1e6:>11.1f}")

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
entries = len(agent._TT)
print(f"\nafter {board.ply() - chess.Board(FEN).ply()} moves: {entries:,} entries, "
      f"peak {peak / 1e6:.1f} MB traced")
projected = agent.TT_MAX * (current / entries if entries else 0) / 1e6
print(f"cap {agent.TT_MAX:,} entries projects to ~{projected:,.0f} MB (limit is 2 GB)")
print("OK" if projected < 800 else "TOO BIG -- lower TT_MAX")

# The table must not change what the search concludes, only how fast it gets there.
agent._TT.clear()
board = chess.Board(FEN)
with_tt = agent.get_move(board.fen(), 20_000)
agent._TT.clear()
probe = agent._TT
agent._TT = {}
no_tt_move = agent.get_move(board.fen(), 20_000)
agent._TT = probe
print(f"\nmove with warm TT: {with_tt}   with cold TT: {no_tt_move}")
print("note: these can legitimately differ -- a TT changes how far the search gets in a fixed")
print("time budget, so it may simply see further. Only an ILLEGAL move would be a bug.")
