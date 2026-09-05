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

The network's single output is a LOGIT: `win_probability = sigmoid(output)`. Training fits
`sigmoid(cp / EVAL_SCALE)`, so at the optimum the raw output is `cp / EVAL_SCALE` and
inference recovers centipawns by multiplying by EVAL_SCALE. Fitting probabilities rather than
raw centipawns is what stops the magnitudes compressing: the loss stops caring whether a won
position is +800 or +1500 (both are ~certainly winning) and spends the model's capacity near
equality, where the choice of move actually turns.
"""

import chess
import numpy as np

NUM_FEATURES = 768

# Centipawns per unit of network output, used ONLY at inference to read the logit back as a
# centipawn-like score. Training squashes with TRAIN_SCALE below.
#
# These differ on purpose. Fitting sigmoid(cp / 200) makes the model's logit a *shrunken*
# estimate of cp -- measured slope 0.335 against truth -- because probability space barely
# distinguishes +800 from +2000. Move choice does not care: minimax is invariant to any
# monotone rescaling. Delta pruning does care, because its margin is additive and written in
# absolute centipawns, so against a 3x-shrunken eval the fixed margin stops pruning and the
# tree grows ~43%. Dividing by the measured slope puts the output back on a true centipawn
# footing and hands delta pruning the scale it was tuned for.
TRAIN_SCALE = 200.0  # cp per logit in the training target: sigmoid(cp / TRAIN_SCALE)
EVAL_SCALE = 516.0  # 1 / fitted logit-per-cp slope (0.00194) for the dual-perspective 256 net

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


def codes_to_dual_features(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From mover-POV codes, return (feats_us, feats_them) for the dual-perspective net.

    feats_us is the ordinary encoding. feats_them is the SAME position seen from the opponent:
    the stored codes are already mover-relative, so the opponent's view is recovered by flipping
    every square (sq ^ 56) and swapping the our/their half (code 1..6 <-> 7..12). This lets the
    two-perspective net train from the existing single-perspective (mover-POV) data.
    """
    single = codes.ndim == 1
    batch = codes[None] if single else codes
    n = batch.shape[0]
    feats_us = codes_to_features(batch)
    feats_them = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    rows, squares = np.nonzero(batch > 0)
    vals = batch[rows, squares].astype(np.int64)
    swapped = np.where(vals <= 6, vals + 6, vals - 6)
    cols = (swapped - 1) * 64 + (squares ^ 56)
    feats_them[rows, cols] = 1.0
    if single:
        return feats_us[0], feats_them[0]
    return feats_us, feats_them
