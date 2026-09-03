"""Prototype: can the feature encoding be folded into the jitted forward pass?

    uv run python training/test_encode_speed.py

The corrected timings say `board_to_codes` costs 58us against the network's 22.5us -- the
encoding is now the more expensive half of an evaluation. It is slow for a structural reason:
`board.piece_map()` allocates a dict and a Piece object per piece, then Python loops over it
to fill a numpy array.

python-chess already holds the position as six piece bitboards plus two colour masks. Handing
those integers straight to numba lets the compiled code do the bit extraction and the forward
pass in one call, allocating nothing. This measures whether that is worth building properly.
Not part of the submission.
"""

import sys
import time
from pathlib import Path

import chess
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalnet
from features import board_to_codes

N = 20_000


@njit(cache=False, fastmath=True, nogil=True)
def _forward_bb(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, kings: int,
    us: int, them: int, flip: bool,
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
) -> float:
    """Encode from bitboards and run the net, without materialising codes or features."""
    n1 = b1.shape[0]
    h1 = np.empty(n1, dtype=np.float32)
    for j in range(n1):
        h1[j] = b1[j]

    for plane in range(6):
        if plane == 0:
            bb = pawns
        elif plane == 1:
            bb = knights
        elif plane == 2:
            bb = bishops
        elif plane == 3:
            bb = rooks
        elif plane == 4:
            bb = queens
        else:
            bb = kings
        ours = bb & us
        theirs = bb & them
        for side in range(2):
            work = ours if side == 0 else theirs
            base = plane if side == 0 else plane + 6
            while work:
                sq = 0
                low = work & -work
                # index of the lowest set bit
                tmp = low
                while tmp > 1:
                    tmp >>= 1
                    sq += 1
                work ^= low
                s = sq ^ 56 if flip else sq
                row = base * 64 + s
                for j in range(n1):
                    h1[j] += w1[row, j]

    for j in range(n1):
        if h1[j] < 0.0:
            h1[j] = 0.0
    n2 = b2.shape[0]
    h2 = np.empty(n2, dtype=np.float32)
    for j in range(n2):
        acc = b2[j]
        for k in range(n1):
            acc += h1[k] * w2[k, j]
        h2[j] = acc if acc > 0.0 else 0.0
    out = b3[0]
    for k in range(n2):
        out += h2[k] * w3[k, 0]
    return float(out)


def forward_bb(board: chess.Board) -> float:
    flip = board.turn == chess.BLACK
    us = board.occupied_co[board.turn]
    them = board.occupied_co[not board.turn]
    return _forward_bb(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        us, them, flip,
        evalnet.W1, evalnet.B1, evalnet.W2, evalnet.B2, evalnet.W3, evalnet.B3,
    )


rng = np.random.default_rng(0)
boards = []
board = chess.Board()
for _ in range(200):
    if board.is_game_over() or board.ply() > 100:
        board = chess.Board()
    moves = list(board.legal_moves)
    board.push(moves[rng.integers(len(moves))])
    boards.append(board.copy())

# Correctness first: the fused path must agree with the existing one.
worst = 0.0
for b in boards:
    a = evalnet.forward(board_to_codes(b))
    c = forward_bb(b)
    worst = max(worst, abs(a - c))
print(f"max |current - fused| over {len(boards)} positions: {worst:.3e}")
print("agreement:", "OK" if worst < 1e-4 else "MISMATCH")

probe = boards[50]
t0 = time.perf_counter()
for _ in range(N):
    evalnet.forward(board_to_codes(probe))
cur = (time.perf_counter() - t0) / N * 1e6

t0 = time.perf_counter()
for _ in range(N):
    forward_bb(probe)
fused = (time.perf_counter() - t0) / N * 1e6

print(f"\ncurrent (piece_map -> codes -> net): {cur:6.2f} us/call")
print(f"fused   (bitboards -> net)         : {fused:6.2f} us/call")
print(f"speedup: {cur / fused:.2f}x")
