"""The evaluation network's forward pass, jitted with numba.

This replaces the onnxruntime session the agent used to call. Two things make it faster:

1. **The first layer is not a matmul.** The 768 inputs are a one-hot-per-piece encoding, so at
   most 32 of them are ever 1 and the rest are 0. `W1 @ x` is therefore just the sum of the
   ~32 rows of W1 that correspond to occupied squares -- about 8k adds instead of 197k
   multiply-adds, and it skips building the 768-float input vector at all.
2. **No per-call framework overhead.** onnxruntime re-validates shapes, marshals buffers and
   crosses the Python/C boundary on every one of the tens of thousands of evaluations a single
   move needs. A jitted function does none of that.

The remaining layers are small enough (256->32->1) that plain loops beat any clever approach.

Compilation happens at import, inside the platform's 60 second init budget, never on the
clock. numba's `cache=True` is deliberately NOT used: it writes next to this file, and the
competition filesystem is read-only outside /tmp.
"""

from pathlib import Path

import numpy as np
import torch
from numba import njit

# Weights ship as .pt because the agent contract names that format explicitly. torch is used
# only here, at import, to read the file; the search itself never touches it. One thread,
# since the platform gives us one core and torch would otherwise size its pools for the host.
torch.set_num_threads(1)
_WEIGHTS = torch.load(
    Path(__file__).resolve().parent / "weights" / "net.pt", map_location="cpu"
)
W1: np.ndarray = _WEIGHTS["w1"].numpy()  # (768, H1), row per input feature
B1: np.ndarray = _WEIGHTS["b1"].numpy()
W2: np.ndarray = _WEIGHTS["w2"].numpy()  # (H1, H2) single, or (2*H1, H2) dual-perspective
B2: np.ndarray = _WEIGHTS["b2"].numpy()
W3: np.ndarray = _WEIGHTS["w3"].numpy()  # (H2, 1)
B3: np.ndarray = _WEIGHTS["b3"].numpy()

# Two architectures share this file. The single-perspective net feeds one H1-wide accumulator
# straight into w2 (H1, H2); the dual-perspective net concatenates the mover's and opponent's
# accumulators, so its w2 has 2*H1 rows. That shape is the unambiguous tell, so nothing else has
# to be recorded to know which forward pass the shipped weights want.
MODE_DUAL: bool = W2.shape[0] == 2 * W1.shape[1]

HIDDEN1 = B1.shape[0]
HIDDEN2 = B2.shape[0]

# de Bruijn sequence table: maps the top 6 bits of (isolated_bit * constant) to that bit's
# index. Built once at import rather than hard-coded, so it cannot be transcribed wrong.
_DEBRUIJN = 0x022FDD63CC95386D
_DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEBRUIJN_INDEX[((1 << _i) * _DEBRUIJN & 0xFFFFFFFFFFFFFFFF) >> 58] = _i

_SIGN_BIT = 0x8000000000000000
_WRAP = 0x10000000000000000


def to_signed(bb: int) -> int:
    """Reinterpret a python-chess bitboard as the int64 numba requires.

    Bitboards are unsigned 64-bit, so any piece on Black's back rank sets bit 63 and pushes the
    value past int64's range. Subtracting 2**64 gives the same bit pattern as a negative int64,
    which is what the jitted code is written to expect.
    """
    return bb - _WRAP if bb & _SIGN_BIT else bb


@njit(cache=False, nogil=True)
def _bit_index(low: int) -> int:
    """Index of the single set bit in `low`, by de Bruijn multiplication.

    numba has no exposed ctz intrinsic and a shift loop costs up to 64 iterations per piece,
    so this is a constant-time table lookup instead.

    Everything here is signed int64, because that is all numba accepts: a bitboard with bit 63
    set (anything on Black's back rank) exceeds int64 and is passed in as its two's-complement
    negative. That works throughout -- `x & -x` still isolates the lowest set bit -- except the
    shift, which sign-extends. Masking with 0x3F afterwards recovers exactly bits 58..63, which
    is what the de Bruijn index needs.
    """
    idx: int = _DEBRUIJN_INDEX[((low * _DEBRUIJN) >> 58) & 0x3F]
    return idx


@njit(cache=False, fastmath=True, nogil=True)
def _forward(
    codes: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
) -> float:
    """Logit for one position, given its 64 int8 piece codes (see features.py)."""
    n1 = b1.shape[0]
    h1 = np.empty(n1, dtype=np.float32)
    for j in range(n1):
        h1[j] = b1[j]

    # Layer 1 as a sparse row gather: one row of W1 per occupied square.
    for sq in range(64):
        code = codes[sq]
        if code > 0:
            row = (code - 1) * 64 + sq
            for j in range(n1):
                h1[j] += w1[row, j]

    for j in range(n1):
        if h1[j] < 0.0:
            h1[j] = 0.0

    # Layer 2, restructured for speed at identical output. Two changes over the naive
    # j-outer/k-inner form: (1) k-outer means the inner loop walks w2[k, :] contiguously
    # instead of striding by n2 down a column, which is far kinder to the cache; (2) after the
    # ReLU above, many h1[k] are exactly 0 and contribute nothing, so those rows are skipped --
    # the standard NNUE sparsity trick. Same arithmetic, fewer cache misses and fewer FLOPs.
    n2 = b2.shape[0]
    h2 = np.empty(n2, dtype=np.float32)
    for j in range(n2):
        h2[j] = b2[j]
    for k in range(n1):
        hk = h1[k]
        if hk != 0.0:
            for j in range(n2):
                h2[j] += hk * w2[k, j]
    for j in range(n2):
        if h2[j] < 0.0:
            h2[j] = 0.0

    out = b3[0]
    for k in range(n2):
        out += h2[k] * w3[k, 0]
    return float(out)


@njit(cache=False, fastmath=True, nogil=True)
def _forward_bb(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, kings: int,
    us: int, them: int, flip: bool,
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
) -> float:
    """Logit straight from python-chess's bitboards, skipping the codes array entirely.

    Building the codes first cost more than the network did: `piece_map()` allocates a dict
    and a Piece object per piece, then Python loops to fill a numpy array. The board is
    already six piece bitboards plus two colour masks, so the same work is a handful of
    integer operations in compiled code, allocating nothing.
    """
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
        for side in range(2):
            work = bb & (us if side == 0 else them)
            base = (plane if side == 0 else plane + 6) * 64
            while work:
                low = work & -work
                sq = _bit_index(low)
                work ^= low
                row = base + (sq ^ 56 if flip else sq)
                for j in range(n1):
                    h1[j] += w1[row, j]

    for j in range(n1):
        if h1[j] < 0.0:
            h1[j] = 0.0

    # Layer 2, restructured for speed at identical output. Two changes over the naive
    # j-outer/k-inner form: (1) k-outer means the inner loop walks w2[k, :] contiguously
    # instead of striding by n2 down a column, which is far kinder to the cache; (2) after the
    # ReLU above, many h1[k] are exactly 0 and contribute nothing, so those rows are skipped --
    # the standard NNUE sparsity trick. Same arithmetic, fewer cache misses and fewer FLOPs.
    n2 = b2.shape[0]
    h2 = np.empty(n2, dtype=np.float32)
    for j in range(n2):
        h2[j] = b2[j]
    for k in range(n1):
        hk = h1[k]
        if hk != 0.0:
            for j in range(n2):
                h2[j] += hk * w2[k, j]
    for j in range(n2):
        if h2[j] < 0.0:
            h2[j] = 0.0

    out = b3[0]
    for k in range(n2):
        out += h2[k] * w3[k, 0]
    return float(out)


@njit(cache=False, fastmath=True, nogil=True)
def _forward_dual_bb(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, kings: int,
    white: int, black: int, white_to_move: bool,
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
) -> float:
    """Logit for the dual-perspective accumulator net (see training/model2.py).

    Builds two H1-wide accumulators in one pass over the pieces -- white's perspective and
    black's -- then feeds [side-to-move, opponent] through w2/w3. w1 is (768, H1) shared; w2 is
    (2*H1, H2) with the first H1 rows applying to the mover's accumulator and the next H1 to the
    opponent's. This is the from-scratch form; the search can later maintain the two
    accumulators incrementally across make/unmake, which is the whole point of the split.
    """
    h1 = b1.shape[0]
    accw = np.empty(h1, dtype=np.float32)
    accb = np.empty(h1, dtype=np.float32)
    for j in range(h1):
        accw[j] = b1[j]
        accb[j] = b1[j]

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
        for color in range(2):  # 0 = white pieces, 1 = black pieces
            work = bb & (white if color == 0 else black)
            # From white's view a white piece is "ours" (plane), a black piece "theirs"
            # (plane+6); from black's view it is the mirror, with the square flipped.
            code_w = plane if color == 0 else plane + 6
            code_b = plane if color == 1 else plane + 6
            base_w = code_w * 64
            base_b = code_b * 64
            while work:
                low = work & -work
                sq = _bit_index(low)
                work ^= low
                row_w = base_w + sq
                row_b = base_b + (sq ^ 56)
                for j in range(h1):
                    accw[j] += w1[row_w, j]
                    accb[j] += w1[row_b, j]

    for j in range(h1):
        if accw[j] < 0.0:
            accw[j] = 0.0
        if accb[j] < 0.0:
            accb[j] = 0.0

    h2n = b2.shape[0]
    hid = np.empty(h2n, dtype=np.float32)
    for j in range(h2n):
        acc = b2[j]
        for k in range(h1):
            if white_to_move:
                acc += accw[k] * w2[k, j] + accb[k] * w2[h1 + k, j]
            else:
                acc += accb[k] * w2[k, j] + accw[k] * w2[h1 + k, j]
        hid[j] = acc if acc > 0.0 else 0.0

    out = b3[0]
    for k in range(h2n):
        out += hid[k] * w3[k, 0]
    return float(out)


def forward(codes: np.ndarray) -> float:
    """Network logit for a position's piece codes. Kept for the training-side tooling."""
    return _forward(codes, W1, B1, W2, B2, W3, B3)


def forward_board(
    pawns: int, knights: int, bishops: int, rooks: int, queens: int, kings: int,
    us: int, them: int, flip: bool,
) -> float:
    """Network logit from raw bitboards -- the path the search actually uses.

    `us`/`them` are the side-to-move's and opponent's colour masks and `flip` is True when Black
    is to move, matching the single-perspective encoding. The dual net wants the same facts in
    white/black terms, so they are recovered from `flip`.
    """
    if MODE_DUAL:
        white = them if flip else us
        black = us if flip else them
        return _forward_dual_bb(
            to_signed(pawns), to_signed(knights), to_signed(bishops), to_signed(rooks),
            to_signed(queens), to_signed(kings), to_signed(white), to_signed(black), not flip,
            W1, B1, W2, B2, W3, B3,
        )
    return _forward_bb(
        to_signed(pawns), to_signed(knights), to_signed(bishops), to_signed(rooks),
        to_signed(queens), to_signed(kings), to_signed(us), to_signed(them), flip,
        W1, B1, W2, B2, W3, B3,
    )


# Compile everything now, at import, so no real move ever pays for compilation. Both forward
# passes are warmed regardless of which weights shipped, so a later weight swap never compiles
# on the clock.
forward(np.zeros(64, dtype=np.int8))
forward_board(0, 0, 0, 0, 0, 0, 0, 0, False)
_forward_dual_bb(0, 0, 0, 0, 0, 0, 0, 0, True,
                 W1, B1, np.zeros((2 * W1.shape[1], B2.shape[0]), dtype=np.float32), B2,
                 W3, B3)
