"""Board -> feature encoding, shared by training and inference so they never diverge.

The encoding is relative to the side to move (NNUE-style). A position is stored compactly
as 64 int8 piece codes (one per square, from the mover's point of view: the mover's back
rank is always at the bottom):

    0            empty
    1..6         our    P N B R Q K   (the side to move)
    7..12        their  P N B R Q K

When Black is to move the board is flipped vertically (square ^ 56) and colours are swapped,
so the network always sees the position "as if to move". This captures the tempo the plain
piece placement would lose, and lets one set of weights serve both colours.

The network input is 768 binary features: 12 piece-planes x 64 squares. The feature index
for a piece with code c on square sq is (c - 1) * 64 + sq.

Scores are likewise stored from the mover's point of view, which is exactly what a negamax
search wants back from evaluate().
"""

import chess
import numpy as np

NUM_FEATURES = 768

_OUR_BASE = 0
_THEIR_BASE = 6


def board_to_codes(board: chess.Board) -> np.ndarray:
    """Return the 64 int8 piece codes for a position, oriented to the side to move."""
    codes = np.zeros(64, dtype=np.int8)
    mover = board.turn
    flip = mover == chess.BLACK
    for square, piece in board.piece_map().items():
        base = _OUR_BASE if piece.color == mover else _THEIR_BASE
        sq = square ^ 56 if flip else square
        codes[sq] = base + piece.piece_type
    return codes


def white_cp_to_mover(cp: int, board: chess.Board) -> int:
    """Convert a White-POV centipawn score to the side-to-move's point of view."""
    return cp if board.turn == chess.WHITE else -cp


def codes_to_features(codes: np.ndarray) -> np.ndarray:
    """Turn int8 codes into float32 features.

    Accepts a single position (shape (64,)) -> (768,), or a batch (N, 64) -> (N, 768).
    """
    if codes.ndim == 1:
        feats = np.zeros(NUM_FEATURES, dtype=np.float32)
        occupied = codes > 0
        squares = np.nonzero(occupied)[0]
        cols = (codes[occupied].astype(np.int64) - 1) * 64 + squares
        feats[cols] = 1.0
        return feats

    n = codes.shape[0]
    feats = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    rows, squares = np.nonzero(codes > 0)
    vals = codes[rows, squares].astype(np.int64)
    cols = (vals - 1) * 64 + squares
    feats[rows, cols] = 1.0
    return feats
