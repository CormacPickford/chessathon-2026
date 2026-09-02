"""Sanity-check the trained model on known positions and the color-symmetry invariant.

    uv run python training/evaluate_model.py

Prints the model's centipawn score (mover POV) for a handful of positions whose sign and
rough magnitude we know, and checks that a position and its mirror encode identically -- the
invariant the side-to-move-relative encoding must satisfy.
"""

import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import board_to_codes, codes_to_features
from training.model import EvalNet

SCALE = 100.0

# (description, fen, expected sign from mover POV)
CASES = [
    ("startpos (white to move)", chess.STARTING_FEN, "~0, slightly +"),
    ("white up a queen, white to move",
     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "large +"),
    ("white down a queen, white to move",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "large -"),
    ("white up a rook, black to move",
     "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b Kk - 0 1", "large - (mover is black)"),
    ("scholar's mate threat setup, black to move",
     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1", "?"),
]


def load_model(path: Path) -> EvalNet:
    model = EvalNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def score_cp(model: EvalNet, board: chess.Board) -> float:
    feats = codes_to_features(board_to_codes(board))
    with torch.no_grad():
        pawns = model(torch.from_numpy(feats).unsqueeze(0)).item()
    return pawns * SCALE


def main() -> None:
    model = load_model(Path("training/model.pt"))

    print("=== known positions (score is mover POV, centipawns) ===")
    for desc, fen, expect in CASES:
        board = chess.Board(fen)
        print(f"{score_cp(model, board):+8.0f}cp   expect {expect:24s}  {desc}")

    print("\n=== color-symmetry invariant ===")
    rng = np.random.default_rng(1)
    max_code_diff = 0
    max_score_diff = 0.0
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board = chess.Board()
        # encoding must be identical for a position and its mirror
        c1 = board_to_codes(board)
        c2 = board_to_codes(board.mirror())
        max_code_diff = max(max_code_diff, int(np.abs(c1 - c2).max()))
        diff = abs(score_cp(model, board) - score_cp(model, board.mirror()))
        max_score_diff = max(max_score_diff, diff)
        moves = list(board.legal_moves)
        board.push(moves[rng.integers(len(moves))])
    code_ok = "OK" if max_code_diff == 0 else "BAD"
    print(f"max |codes - mirror codes|  = {max_code_diff}  ({code_ok})")
    print(f"max |score - mirror score|  = {max_score_diff:.2e}cp  "
          f"({'OK' if max_score_diff < 1e-2 else 'BAD'})")


if __name__ == "__main__":
    main()
