"""Measure an agent's move quality as average centipawn loss vs a Stockfish reference.

    uv run python training/cploss.py --agent . --budget-ms 800
    uv run python training/cploss.py --agent opponents/base_data10m --budget-ms 800

For each test FEN, Stockfish (local, never ships) gives the best move and the position score.
The agent picks a move at a fixed wall-clock budget; Stockfish scores the resulting position.
cp_loss = ref_score - score_after_agent_move (>= 0). The mean is a sharp, low-variance quality
signal that -- unlike a fast game A/B -- REWARDS reaching more depth, because a deeper search
picks moves Stockfish agrees with more often. It is the right yardstick for depth-buying search
changes, which a starved-clock game A/B cannot see.

The reference (best move + score per FEN) is cached to training/ref_<depth>.json and reused, so
only the agent's side is recomputed between configs. Run configs one at a time on a quiet CPU:
the agent's budget is wall time, so contention would understate its depth.
"""

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
EXE = Path(__file__).resolve().parent.parent / "opponents" / "engines" / "stockfish.exe"
CP_CAP = 1000  # clamp per-position loss so one blunder cannot dominate the mean


def load_agent(path: Path) -> ModuleType:
    if (path / "__init__.py").exists():
        pkg = ".".join(path.resolve().relative_to(Path(__file__).resolve().parent.parent).parts)
        return importlib.import_module(f"{pkg}.agent")
    spec = importlib.util.spec_from_file_location("cand_agent", path / "agent.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sf_score(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int:
    """Stockfish score in centipawns from the side-to-move's POV, mates mapped to +/-CP_CAP*.."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].relative.score(mate_score=100_000)


def main() -> None:
    p = argparse.ArgumentParser(description="Average centipawn loss vs Stockfish.")
    p.add_argument("--agent", type=Path, default=Path("."))
    p.add_argument("--positions", type=Path, default=Path("training/testpos.txt"))
    p.add_argument("--budget-ms", type=int, default=800)
    p.add_argument("--ref-depth", type=int, default=14)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--set", action="append", default=[],
                   help="override an agent module constant, e.g. --set LMP_BASE=5 (repeatable)")
    args = p.parse_args()

    fens = args.positions.read_text(encoding="utf-8").split("\n")
    fens = [f for f in fens if f.strip()][: args.limit]
    engine = chess.engine.SimpleEngine.popen_uci(str(EXE))
    engine.configure({"Threads": 1, "Hash": 64})

    ref_path = Path(f"training/ref_d{args.ref_depth}_{len(fens)}.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    else:
        ref = {}
        t0 = time.monotonic()
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=args.ref_depth))
            ref[fen] = {"best": info["pv"][0].uci(),
                        "score": info["score"].relative.score(mate_score=100_000)}
            if (i + 1) % 50 == 0:
                print(f"  ref {i + 1}/{len(fens)} ({(i + 1) / (time.monotonic() - t0):.1f}/s)")
        ref_path.write_text(json.dumps(ref))
        print(f"cached reference to {ref_path}")

    agent = load_agent(args.agent)
    if hasattr(agent, "PONDER_ENABLED"):
        agent.PONDER_ENABLED = False
    for override in args.set:
        name, _, val = override.partition("=")
        setattr(agent, name.strip(), int(val))
        print(f"  override {name.strip()} = {int(val)}")

    losses: list[int] = []
    agree = 0
    t0 = time.monotonic()
    for fen in fens:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 120_000 if args.budget_ms >= 6000 else args.budget_ms * 20)
        # get_move budgets ~5% of the clock, so pass a clock ~20x the target per-move budget.
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            losses.append(CP_CAP)
            continue
        if uci == ref[fen]["best"]:
            agree += 1
        board.push(move)
        after = sf_score(engine, board, args.ref_depth)  # opponent POV after our move
        loss = ref[fen]["score"] - (-after)
        losses.append(max(0, min(CP_CAP, loss)))
    if hasattr(agent, "_stop_ponder"):
        agent._stop_ponder()
    engine.quit()

    n = len(losses)
    mean = sum(losses) / n
    losses_sorted = sorted(losses)
    median = losses_sorted[n // 2]
    print(f"\nagent {args.agent}  budget ~{args.budget_ms}ms/move  ref depth {args.ref_depth}")
    print(f"positions {n}   mean cp_loss {mean:.1f}   median {median}   "
          f"SF-move agree {agree / n * 100:.1f}%   ({n / (time.monotonic() - t0):.1f} pos/s)")


if __name__ == "__main__":
    main()
