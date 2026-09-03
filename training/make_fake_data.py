"""Write a tiny synthetic data.npz so the train/export/evaluate chain can be smoke-tested.

    uv run python training/make_fake_data.py --out training/data_fake.npz --n 20000

The positions are real (random legal play), the scores are a crude material count plus noise.
This proves the plumbing works end to end; it teaches the net nothing worth shipping.
Not part of the submission.
"""

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import board_to_codes, white_cp_to_mover

VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500,
          chess.QUEEN: 900, chess.KING: 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic training data for smoke tests.")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("training/data_fake.npz"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    codes = np.zeros((args.n, 64), dtype=np.int8)
    scores = np.zeros(args.n, dtype=np.int16)

    board = chess.Board()
    for i in range(args.n):
        if board.is_game_over() or board.ply() > 80:
            board = chess.Board()
        moves = list(board.legal_moves)
        board.push(moves[rng.integers(len(moves))])
        white_cp = sum(
            v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
            for p, v in VALUES.items()
        ) + int(rng.normal(0, 50))
        codes[i] = board_to_codes(board)
        scores[i] = int(np.clip(white_cp_to_mover(white_cp, board), -2000, 2000))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, codes=codes, scores=scores)
    print(f"wrote {args.out}: {args.n:,} positions, "
          f"score mean {scores.mean():.1f} std {scores.std():.1f}")


if __name__ == "__main__":
    main()
