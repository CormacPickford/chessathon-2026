"""Play a game between two agent directories without the harness. Works on Windows.

    uv run python training/quickplay.py --white . --black baselines/numba --base-ms 8000
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_agent(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path / "agent.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(white_path: Path, black_path: Path, base_ms: int, inc_ms: int, fen: str) -> None:
    print(f"white {white_path}   black {black_path}")
    white = load_agent("agent_white", white_path)
    black = load_agent("agent_black", black_path)

    board = chess.Board(fen)
    clocks = {chess.WHITE: base_ms, chess.BLACK: base_ms}

    while not board.is_game_over(claim_draw=True):
        agent = white if board.turn == chess.WHITE else black
        side = "white" if board.turn == chess.WHITE else "black"
        time_left = clocks[board.turn]

        t0 = time.monotonic()
        uci = agent.get_move(board.fen(), time_left)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        clocks[board.turn] = time_left - elapsed_ms + inc_ms

        if clocks[board.turn] < 0:
            print(f"{side} flagged")
            return
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"{side} played illegal {uci}")
            return

        board.push(move)
        if board.ply() % 2 == 0 or board.is_game_over():
            print(f"move {board.fullmove_number:3d}  {side} {uci:6s} "
                  f"[w {clocks[chess.WHITE] // 1000:3d}s  b {clocks[chess.BLACK] // 1000:3d}s]")

    print(f"\nResult: {board.result(claim_draw=True)}  ({board.outcome(claim_draw=True)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play one game between two agents.")
    parser.add_argument("--white", type=Path, default=Path("."))
    parser.add_argument("--black", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--inc-ms", type=int, default=500)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    args = parser.parse_args()
    play(args.white, args.black, args.base_ms, args.inc_ms, args.fen)


if __name__ == "__main__":
    main()
