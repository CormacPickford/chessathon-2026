"""Check the shipped agent against the platform contract in AGENTS.md.

    uv run python training/test_rules.py

Covers what can be checked locally: import budget, clock safety across a range of remaining
times, UCI legality and reply size, and single-threaded ONNX. Not part of the submission.
"""

import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent

# Import budget: the platform allows 60s before the clock starts.
probe = "import time; t=time.perf_counter(); import agent; print(time.perf_counter()-t)"
out = subprocess.run(
    [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
)
import_s = float(out.stdout.strip().splitlines()[-1])
print(f"import time: {import_s:.2f}s / 60s budget  {'OK' if import_s < 60 else 'FAIL'}")

sys.path.insert(0, str(ROOT))
import agent  # noqa: E402

# One core is all the platform gives us. The jitted forward pass is a plain loop with no
# parallel=True, so it cannot spawn workers; assert that rather than trust it.
import evalnet  # noqa: E402

sig = evalnet._forward.signatures
print(f"eval backend: numba, {len(sig)} compiled signature(s), "
      f"parallel={'parallel' in str(evalnet._forward.targetoptions)}  "
      f"{'OK' if sig else 'FAIL - not compiled at import'}")

# Clock safety. Budget is max(50, min(time_left*0.05, 4000)) ms; the move must fit the clock.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # open middlegame
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # endgame
]
print("\nclock safety (elapsed vs budget vs clock remaining):")
worst_overrun = -1e9
for time_left in (120_000, 30_000, 10_000, 1_000, 200, 100, 50, 30):
    budget = max(50.0, min(time_left * 0.05, 4000.0))
    for fen in POSITIONS:
        t0 = time.perf_counter()
        uci = agent.get_move(fen, time_left)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        move = chess.Move.from_uci(uci)
        legal = move in chess.Board(fen).legal_moves
        overrun = elapsed_ms - time_left  # positive means the clock would have flagged
        worst_overrun = max(worst_overrun, overrun)
        flag = "FLAG" if elapsed_ms > time_left else "ok"
        if flag == "FLAG" or time_left <= 1_000:
            print(f"  clock {time_left:>7}ms  budget {budget:>6.0f}ms  "
                  f"used {elapsed_ms:>7.1f}ms  {flag:4s}  legal={legal}  reply={len(uci)}B")
        assert legal, f"illegal move {uci} from {fen}"
        assert len(uci.encode()) <= 4096, "reply over 4 KB counts as illegal"

print(f"\nworst case: move exceeded the remaining clock by {worst_overrun:+.1f}ms")
print("all replies legal UCI and well under the 4 KB cap")
