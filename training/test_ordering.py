"""A/B two agent directories on search efficiency, not on games.

    uv run python training/test_ordering.py --a . --b opponents/prev_qs

Games are noisy; node counts are not. For each position we run a fixed-depth search in both
agents and record wall time and evaluate() calls (= leaves reached). Better move ordering
should cut the node count; if it does not, the extra per-node work is not buying anything.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POSITIONS = {
    "startpos": chess.STARTING_FEN,
    "open middlegame": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "tactical": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path / "agent.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(mod: ModuleType, fen: str, depth: int) -> tuple[float, int]:
    """Return (seconds, evaluate calls) for a fixed-depth search."""
    calls = 0
    real = mod.evaluate

    def counting(board: chess.Board) -> int:
        nonlocal calls
        calls += 1
        return int(real(board))

    mod.evaluate = counting
    try:
        mod._deadline = time.perf_counter() + 600.0
        mod._timed_out = False
        t0 = time.perf_counter()
        mod.negamax(chess.Board(fen), depth, -1e9, 1e9)
        dt = time.perf_counter() - t0
    finally:
        mod.evaluate = real
    return dt, calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare search efficiency of two agents.")
    parser.add_argument("--a", type=Path, default=Path("."))
    parser.add_argument("--b", type=Path, default=Path("opponents/prev_qs"))
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    a = load("agent_a", args.a)
    b = load("agent_b", args.b)
    print(f"A = {args.a}    B = {args.b}    fixed depth {args.depth}\n")
    print(f"{'position':<18}{'A nodes':>10}{'B nodes':>10}{'nodes':>9}"
          f"{'A time':>9}{'B time':>9}{'time':>9}")

    tot_a_n = tot_b_n = 0
    tot_a_t = tot_b_t = 0.0
    for label, fen in POSITIONS.items():
        a_t, a_n = probe(a, fen, args.depth)
        b_t, b_n = probe(b, fen, args.depth)
        tot_a_n += a_n
        tot_b_n += b_n
        tot_a_t += a_t
        tot_b_t += b_t
        dn = (a_n - b_n) / b_n * 100 if b_n else 0.0
        dt = (a_t - b_t) / b_t * 100 if b_t else 0.0
        print(f"{label:<18}{a_n:>10,}{b_n:>10,}{dn:>8.1f}%{a_t:>8.2f}s{b_t:>8.2f}s{dt:>8.1f}%")

    dn = (tot_a_n - tot_b_n) / tot_b_n * 100
    dt = (tot_a_t - tot_b_t) / tot_b_t * 100
    print(f"\n{'TOTAL':<18}{tot_a_n:>10,}{tot_b_n:>10,}{dn:>8.1f}%"
          f"{tot_a_t:>8.2f}s{tot_b_t:>8.2f}s{dt:>8.1f}%")
    print("\nnegative = A better. Nodes measure ordering quality; time is what actually")
    print("bounds depth on the clock, so a node win paired with a time loss is a wash.")


if __name__ == "__main__":
    main()
