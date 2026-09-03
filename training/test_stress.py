"""Hammer the agent for illegal moves, crashes, and clock overruns.

    uv run python training/test_stress.py --games 12

Every one of these loses a game outright on the platform, and each is silent until it happens
in a rated game. So play real games to completion from varied openings and assert, on every
single move: the reply parses, is legal, fits the 4 KB cap, and came back inside the clock.

Also exercises the pondering thread, which is the part most able to break these invariants --
it mutates the same search globals from another thread, and the join at the start of the next
get_move is charged to our clock. Not part of the submission.
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

OPENINGS = [
    "", "e4 e5", "d4 d5 c4", "e4 c5 Nf3", "d4 Nf6 c4 g6", "c4 e5 Nc3",
    "e4 e6 d4 d5", "Nf3 d5 g3 c5", "e4 d5 exd5 Qxd5", "d4 f5 g3",
    "e4 e5 Nf3 Nc6 Bb5 a6", "d4 d5 Bf4 Nf6 e3",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress the agent for contract violations.")
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--base-ms", type=int, default=8_000)
    parser.add_argument("--inc-ms", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=140)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    moves_made = 0
    worst_margin = 1e9
    worst_join = 0.0
    illegal = crashed = flagged = 0

    for g in range(args.games):
        board = chess.Board()
        for token in OPENINGS[g % len(OPENINGS)].split():
            board.push_san(token)
        clock = float(args.base_ms)

        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            # The agent plays one side; random legal play answers, which reaches far stranger
            # positions than a real opponent would and is exactly what we want to probe.
            if board.turn == chess.WHITE:
                t0 = time.perf_counter()
                try:
                    uci = agent.get_move(board.fen(), int(clock))
                except Exception as exc:
                    print(f"  CRASH game {g} ply {board.ply()}: {type(exc).__name__}: {exc}")
                    crashed += 1
                    break
                elapsed = (time.perf_counter() - t0) * 1000.0
                worst_margin = min(worst_margin, clock - elapsed)
                if elapsed > clock:
                    print(f"  FLAG game {g} ply {board.ply()}: used {elapsed:.0f}ms "
                          f"of {clock:.0f}ms")
                    flagged += 1
                    break
                clock += args.inc_ms - elapsed
                if len(uci.encode()) > 4096:
                    print(f"  OVERSIZE reply game {g}: {len(uci)}B")
                    illegal += 1
                    break
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    print(f"  MALFORMED game {g} ply {board.ply()}: {uci!r}")
                    illegal += 1
                    break
                if move not in board.legal_moves:
                    print(f"  ILLEGAL game {g} ply {board.ply()}: {uci} in {board.fen()}")
                    illegal += 1
                    break
                moves_made += 1
            else:
                legal = list(board.legal_moves)
                move = legal[rng.integers(len(legal))]
                # Measure how long stopping the ponder thread costs -- it is charged to us.
                t0 = time.perf_counter()
                agent._stop_ponder()
                worst_join = max(worst_join, (time.perf_counter() - t0) * 1000.0)
            board.push(move)
        print(f"game {g + 1}/{args.games}: {board.ply()} plies, clock {clock:.0f}ms", flush=True)

    print(f"\nagent moves played : {moves_made:,}")
    print(f"illegal/malformed  : {illegal}")
    print(f"crashes            : {crashed}")
    print(f"flag falls         : {flagged}")
    print(f"tightest clock gap : {worst_margin:.1f}ms remaining")
    print(f"worst ponder join  : {worst_join:.1f}ms (charged to our clock next move)")
    ok = illegal == 0 and crashed == 0 and flagged == 0
    print("\nPASS - no contract violations" if ok else "\nFAIL - see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
