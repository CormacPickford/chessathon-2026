"""Quick smoke test + node-rate benchmark for the agent. Run from the repo root."""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

# A handful of moves from the opening to confirm the agent returns legal UCI.
board = chess.Board()
for i in range(6):
    move = agent.get_move(board.fen(), 120_000)
    print(f"Move {i + 1}: {move}")
    board.push(chess.Move.from_uci(move))

# Benchmark raw eval throughput (the cost that bounds search depth).
board = chess.Board()
n = 2000
t0 = time.monotonic()
for _ in range(n):
    agent.evaluate(board)
dt = time.monotonic() - t0
print(f"\neval: {n / dt:,.0f} calls/sec  ({dt / n * 1e6:.1f} us/call)")

# Benchmark a fixed-depth search to see the end-to-end cost of one deepening step.
agent._deadline = time.monotonic() + 30.0
agent._timed_out = False
t0 = time.monotonic()
agent.negamax(chess.Board(), 4, -1e9, 1e9)
print(f"depth-4 search from startpos: {time.monotonic() - t0:.2f}s")
print("\nDone - agent is working.")
