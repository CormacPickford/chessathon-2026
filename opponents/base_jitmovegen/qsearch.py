"""Quiescence search, jitted end to end with numba.

Profiling put ~80% of quiescence node time in python-chess: `board.push` allocates a board
state for pop, capture generation walks Python generators, and the check machinery recomputes
attack masks in interpreted code. Quiescence is most of the tree (~100k of 128k nodes at depth
7), so the whole subtree below the horizon is moved into compiled code: capture and evasion
generation, make, legality, SEE, delta pruning and the eval call all run without touching a
Python object.

Scope is deliberately narrow, which is what makes this safe:

- Quiescence never returns a move, only a score. A bug here can misjudge a position but can
  never emit an illegal move, which is the mistake that loses a game outright.
- Castling cannot occur below the horizon: captures are not castling moves, and castling out
  of check is illegal, so evasions never castle either. No castling logic exists here at all.
- There is no transposition table below the horizon, so no hashing and no collision risk.

Position state is eight bitboards plus the side to move and the en passant square, passed as
scalars and rebuilt functionally by `_make` -- no unmake, no shared mutable state, safe under
the ponder thread. Bitboards are signed int64 throughout, the same two's-complement
reinterpretation `evalnet` uses (see `evalnet.to_signed`).

The deadline is NOT checked inside the jitted tree. A single quiescence subtree is bounded by
QS_MAX_DEPTH and capture exhaustion -- sub-millisecond in practice -- and the caller checks
the clock before entering, so the worst case fits comfortably inside the agent's OVERHEAD_MS
headroom.
"""

import chess
import numpy as np
from numba import njit

from . import evalnet
from .evalnet import _bit_index, to_signed
from .features import EVAL_SCALE

# Mirrors agent.py's values; duplicated rather than imported to keep this module leaf-level.
MATE = 1_000_000
EVAL_CAP = 20_000

# Piece values indexed by python-chess piece type (1=pawn .. 6=king), matching agent.PIECE_VALUE.
PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20_000], dtype=np.int64)

_WRAP = 0x10000000000000000


def _signed_bb(bb: int) -> np.int64:
    return np.int64(bb - _WRAP if bb & 0x8000000000000000 else bb)


# ---------------------------------------------------------------------------
# Attack tables, built once at import in plain Python and stored signed.
# ---------------------------------------------------------------------------

BB_SQ = np.zeros(64, dtype=np.int64)
KNIGHT_ATT = np.zeros(64, dtype=np.int64)
KING_ATT = np.zeros(64, dtype=np.int64)
PAWN_ATT_W = np.zeros(64, dtype=np.int64)  # squares a WHITE pawn on sq attacks
PAWN_ATT_B = np.zeros(64, dtype=np.int64)

# Ray masks per direction per square, direction indices 0..3 positive (N, E, NE, NW: the first
# blocker is the lowest set bit) and 4..7 negative (S, W, SE, SW: the highest).
_DIR_STEPS = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (1, -1), (-1, -1)]
RAYS = np.zeros((8, 64), dtype=np.int64)
ROOK_DIRS = np.array([0, 1, 4, 5], dtype=np.int64)
BISHOP_DIRS = np.array([2, 3, 6, 7], dtype=np.int64)

for _s in range(64):
    _f, _r = _s % 8, _s // 8
    BB_SQ[_s] = _signed_bb(1 << _s)
    _kn = 0
    for _df, _dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        _nf, _nr = _f + _df, _r + _dr
        if 0 <= _nf < 8 and 0 <= _nr < 8:
            _kn |= 1 << (_nr * 8 + _nf)
    KNIGHT_ATT[_s] = _signed_bb(_kn)
    _kg = 0
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == 0 and _dr == 0:
                continue
            _nf, _nr = _f + _df, _r + _dr
            if 0 <= _nf < 8 and 0 <= _nr < 8:
                _kg |= 1 << (_nr * 8 + _nf)
    KING_ATT[_s] = _signed_bb(_kg)
    _pw = _pb = 0
    for _df in (-1, 1):
        if 0 <= _f + _df < 8:
            if _r + 1 < 8:
                _pw |= 1 << ((_r + 1) * 8 + _f + _df)
            if _r - 1 >= 0:
                _pb |= 1 << ((_r - 1) * 8 + _f + _df)
    PAWN_ATT_W[_s] = _signed_bb(_pw)
    PAWN_ATT_B[_s] = _signed_bb(_pb)
    for _d, (_df, _dr) in enumerate(_DIR_STEPS):
        _ray = 0
        _nf, _nr = _f + _df, _r + _dr
        while 0 <= _nf < 8 and 0 <= _nr < 8:
            _ray |= 1 << (_nr * 8 + _nf)
            _nf, _nr = _nf + _df, _nr + _dr
        RAYS[_d, _s] = _signed_bb(_ray)


# ---------------------------------------------------------------------------
# Jitted primitives
# ---------------------------------------------------------------------------


@njit(cache=False, nogil=True)
def _msb_index(x: int) -> int:
    """Index of the highest set bit. Bit 63 shows up as the sign bit, hence the guard."""
    if x < 0:
        return 63
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return _bit_index((x >> 1) + 1)


@njit(cache=False, nogil=True)
def _slider_att(sq: int, occ: int, dirs: np.ndarray) -> int:
    """Classical ray attacks: full ray, then cut everything beyond the first blocker."""
    att = 0
    for i in range(4):
        d = dirs[i]
        ray: int = RAYS[d, sq]
        att |= ray
        blockers = ray & occ
        if blockers != 0:
            first = _bit_index(blockers & -blockers) if d < 4 else _msb_index(blockers)
            att &= ~RAYS[d, first]
    return att


@njit(cache=False, nogil=True)
def _attackers(
    sq: int, occ: int, by_white: bool,
    p: int, n: int, b: int, r: int, q: int, k: int, occw: int, occb: int,
) -> int:
    """Mask of `by_white`'s pieces attacking `sq`, sliders blocked by `occ`."""
    side = occw if by_white else occb
    att: int = KNIGHT_ATT[sq] & n & side
    att |= KING_ATT[sq] & k & side
    # White attackers of sq stand where a black pawn on sq would capture, and vice versa.
    pa = PAWN_ATT_B[sq] if by_white else PAWN_ATT_W[sq]
    att |= pa & p & side
    ra = _slider_att(sq, occ, ROOK_DIRS)
    att |= ra & (r | q) & side
    ba = _slider_att(sq, occ, BISHOP_DIRS)
    att |= ba & (b | q) & side
    return att


@njit(cache=False, nogil=True)
def _piece_type_at(bb: int, p: int, n: int, b: int, r: int, q: int, k: int) -> int:
    if p & bb:
        return 1
    if n & bb:
        return 2
    if b & bb:
        return 3
    if r & bb:
        return 4
    if q & bb:
        return 5
    if k & bb:
        return 6
    return 0


@njit(cache=False, nogil=True)
def _make(
    move: int,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Apply a packed move (from | to<<6 | promo<<12) and return the new position.

    Functional on purpose: no unmake and nothing shared, the recursion just passes new
    scalars down. Castling rights are not tracked because nothing below the horizon can
    castle or needs to know the rights.
    """
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    frm_bb = BB_SQ[frm]
    to_bb = BB_SQ[to]
    us = occw if wtm else occb
    them = occb if wtm else occw
    new_ep = -1

    if them & to_bb:
        # Plain capture: clear the victim from its piece board and colour mask.
        if p & to_bb:
            p ^= to_bb
        elif n & to_bb:
            n ^= to_bb
        elif b & to_bb:
            b ^= to_bb
        elif r & to_bb:
            r ^= to_bb
        elif q & to_bb:
            q ^= to_bb
        else:
            k ^= to_bb
        them ^= to_bb
    elif (p & frm_bb) and to == ep:
        # En passant: the captured pawn is not on the destination square.
        cap_bb = BB_SQ[to - 8] if wtm else BB_SQ[to + 8]
        p ^= cap_bb
        them ^= cap_bb

    if p & frm_bb:
        p ^= frm_bb
        if promo == 0:
            p |= to_bb
            if to - frm == 16 or frm - to == 16:
                new_ep = (frm + to) >> 1
        elif promo == 2:
            n |= to_bb
        elif promo == 3:
            b |= to_bb
        elif promo == 4:
            r |= to_bb
        else:
            q |= to_bb
    elif n & frm_bb:
        n ^= frm_bb
        n |= to_bb
    elif b & frm_bb:
        b ^= frm_bb
        b |= to_bb
    elif r & frm_bb:
        r ^= frm_bb
        r |= to_bb
    elif q & frm_bb:
        q ^= frm_bb
        q |= to_bb
    else:
        k ^= frm_bb
        k |= to_bb

    us ^= frm_bb
    us |= to_bb
    if wtm:
        return p, n, b, r, q, k, us, them, new_ep
    return p, n, b, r, q, k, them, us, new_ep


@njit(cache=False, nogil=True)
def _own_king_safe(
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, moved_white: bool,
) -> bool:
    """After `moved_white` moved, is their king out of check? The legality test."""
    us = occw if moved_white else occb
    kbb = k & us
    ksq = _bit_index(kbb & -kbb)
    occ = occw | occb
    return _attackers(ksq, occ, not moved_white, p, n, b, r, q, k, occw, occb) == 0


# ---------------------------------------------------------------------------
# Move generation. Packed move ints carry their sort key in the high bits:
# key = tier << 40 | (mvv_lva + 32768) << 16 | move, so one descending sort of plain ints
# yields the search order, deterministically (the move bits break every tie).
# ---------------------------------------------------------------------------

_KEY_MOVE_MASK = 0xFFFF
_SCORE_OFFSET = 32768


@njit(cache=False, nogil=True)
def _sort_desc(arr: np.ndarray, cnt: int) -> None:
    for i in range(1, cnt):
        v = arr[i]
        j = i - 1
        while j >= 0 and arr[j] < v:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = v


@njit(cache=False, nogil=True)
def _pack(frm: int, to: int, promo: int, victim: int, attacker: int, tier: int) -> int:
    score: int = PIECE_VALUE[victim] * 16 - PIECE_VALUE[attacker] + _SCORE_OFFSET
    return (tier << 40) | (score << 16) | (promo << 12) | (to << 6) | frm


@njit(cache=False, nogil=True)
def _gen_noisy(
    out: np.ndarray,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    """Pseudo-legal captures (all promotion types on capture), queen push-promotions, and en
    passant -- exactly the move set the Python `_noisy_moves` produced. Returns the count."""
    us = occw if wtm else occb
    them = occb if wtm else occw
    occ = occw | occb
    cnt = 0

    # Knights, bishops, rooks, queens, king: table/ray attacks into enemy occupancy.
    work = n & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        targets = KNIGHT_ATT[frm] & them
        while targets:
            t = targets & -targets
            targets ^= t
            to = _bit_index(t)
            victim = _piece_type_at(t, p, n, b, r, q, k)
            out[cnt] = _pack(frm, to, 0, victim, 2, 1)
            cnt += 1
    work = k & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        targets = KING_ATT[frm] & them
        while targets:
            t = targets & -targets
            targets ^= t
            to = _bit_index(t)
            victim = _piece_type_at(t, p, n, b, r, q, k)
            out[cnt] = _pack(frm, to, 0, victim, 6, 1)
            cnt += 1
    work = (b | r | q) & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        piece = _piece_type_at(low, p, n, b, r, q, k)
        if piece == 3:
            att = _slider_att(frm, occ, BISHOP_DIRS)
        elif piece == 4:
            att = _slider_att(frm, occ, ROOK_DIRS)
        else:
            att = _slider_att(frm, occ, ROOK_DIRS) | _slider_att(frm, occ, BISHOP_DIRS)
        targets = att & them
        while targets:
            t = targets & -targets
            targets ^= t
            to = _bit_index(t)
            victim = _piece_type_at(t, p, n, b, r, q, k)
            out[cnt] = _pack(frm, to, 0, victim, piece, 1)
            cnt += 1

    # Pawns: captures (all four promotion pieces on the last rank), queen push-promotions,
    # and en passant.
    work = p & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        rank = frm >> 3
        promo_rank = rank == (6 if wtm else 1)
        targets = (PAWN_ATT_W[frm] if wtm else PAWN_ATT_B[frm]) & them
        while targets:
            t = targets & -targets
            targets ^= t
            to = _bit_index(t)
            victim = _piece_type_at(t, p, n, b, r, q, k)
            if promo_rank:
                for pr in range(2, 6):
                    out[cnt] = _pack(frm, to, pr, victim, 1, 1)
                    cnt += 1
            else:
                out[cnt] = _pack(frm, to, 0, victim, 1, 1)
                cnt += 1
        if promo_rank:
            push = frm + 8 if wtm else frm - 8
            if not (occ & BB_SQ[push]):
                out[cnt] = _pack(frm, push, 5, 0, 1, 1)
                cnt += 1
        if ep >= 0:
            pa = PAWN_ATT_W[frm] if wtm else PAWN_ATT_B[frm]
            if pa & BB_SQ[ep]:
                out[cnt] = _pack(frm, ep, 0, 1, 1, 1)
                cnt += 1
    return cnt


@njit(cache=False, nogil=True)
def _gen_quiets(
    out: np.ndarray, cnt: int,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool,
) -> int:
    """Pseudo-legal non-captures except castling: piece moves to empty squares, pawn pushes,
    and quiet under-promotions (the queen push-promotion lives in the noisy set). Appends to
    `out` from index `cnt`."""
    us = occw if wtm else occb
    occ = occw | occb
    empty = ~occ

    work = n & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        targets = KNIGHT_ATT[frm] & empty
        while targets:
            t = targets & -targets
            targets ^= t
            out[cnt] = _pack(frm, _bit_index(t), 0, 0, 2, 0)
            cnt += 1
    work = k & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        targets = KING_ATT[frm] & empty
        while targets:
            t = targets & -targets
            targets ^= t
            out[cnt] = _pack(frm, _bit_index(t), 0, 0, 6, 0)
            cnt += 1
    work = (b | r | q) & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        piece = _piece_type_at(low, p, n, b, r, q, k)
        if piece == 3:
            att = _slider_att(frm, occ, BISHOP_DIRS)
        elif piece == 4:
            att = _slider_att(frm, occ, ROOK_DIRS)
        else:
            att = _slider_att(frm, occ, ROOK_DIRS) | _slider_att(frm, occ, BISHOP_DIRS)
        targets = att & empty
        while targets:
            t = targets & -targets
            targets ^= t
            out[cnt] = _pack(frm, _bit_index(t), 0, 0, piece, 0)
            cnt += 1

    work = p & us
    while work:
        low = work & -work
        work ^= low
        frm = _bit_index(low)
        rank = frm >> 3
        promo_rank = rank == (6 if wtm else 1)
        push = frm + 8 if wtm else frm - 8
        if not (occ & BB_SQ[push]):
            if promo_rank:
                # The queen push-promotion is already in the noisy set; add the rest.
                for pr in range(2, 5):
                    out[cnt] = _pack(frm, push, pr, 0, 1, 0)
                    cnt += 1
            else:
                out[cnt] = _pack(frm, push, 0, 0, 1, 0)
                cnt += 1
                if rank == (1 if wtm else 6):
                    push2 = frm + 16 if wtm else frm - 16
                    if not (occ & BB_SQ[push2]):
                        out[cnt] = _pack(frm, push2, 0, 0, 1, 0)
                        cnt += 1
    return cnt


@njit(cache=False, nogil=True)
def _gen_evasions(
    out: np.ndarray,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    """Every pseudo-legal move except castling (which is illegal in check anyway)."""
    cnt = _gen_noisy(out, p, n, b, r, q, k, occw, occb, wtm, ep)
    return _gen_quiets(out, cnt, p, n, b, r, q, k, occw, occb, wtm)


@njit(cache=False, nogil=True)
def _filter_legal(
    out: np.ndarray, cnt: int,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    """Compact `out[:cnt]` down to the moves that leave the mover's king safe."""
    kept = 0
    for i in range(cnt):
        np_, nn, nb, nr, nq, nk, nw, nbl, _nep = _make(
            out[i] & _KEY_MOVE_MASK, p, n, b, r, q, k, occw, occb, wtm, ep)
        if _own_king_safe(np_, nn, nb, nr, nq, nk, nw, nbl, wtm):
            out[kept] = out[i]
            kept += 1
    return kept


@njit(cache=False, nogil=True)
def _legal_noisy_jit(
    out: np.ndarray,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    cnt = _gen_noisy(out, p, n, b, r, q, k, occw, occb, wtm, ep)
    cnt = _filter_legal(out, cnt, p, n, b, r, q, k, occw, occb, wtm, ep)
    _sort_desc(out, cnt)
    return cnt


@njit(cache=False, nogil=True)
def _legal_quiets_jit(
    out: np.ndarray,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    cnt = _gen_quiets(out, 0, p, n, b, r, q, k, occw, occb, wtm)
    return _filter_legal(out, cnt, p, n, b, r, q, k, occw, occb, wtm, ep)


def legal_noisy(out: np.ndarray, board: chess.Board) -> int:
    """Legal captures + queen promotions into `out` as packed ints, MVV-LVA order.

    The interior search pushes these moves, so correctness here is game-critical: the set is
    verified move-for-move against python-chess by training/test_qsearch.py.
    """
    return int(_legal_noisy_jit(
        out,
        to_signed(board.pawns), to_signed(board.knights), to_signed(board.bishops),
        to_signed(board.rooks), to_signed(board.queens), to_signed(board.kings),
        to_signed(board.occupied_co[True]), to_signed(board.occupied_co[False]),
        board.turn, -1 if board.ep_square is None else board.ep_square,
    ))


def legal_quiets(out: np.ndarray, board: chess.Board) -> int:
    """Legal quiet moves (no castling -- the caller adds it) into `out` as packed ints."""
    return int(_legal_quiets_jit(
        out,
        to_signed(board.pawns), to_signed(board.knights), to_signed(board.bishops),
        to_signed(board.rooks), to_signed(board.queens), to_signed(board.kings),
        to_signed(board.occupied_co[True]), to_signed(board.occupied_co[False]),
        board.turn, -1 if board.ep_square is None else board.ep_square,
    ))


@njit(cache=False, nogil=True)
def _see(
    move: int,
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
) -> int:
    """Static exchange evaluation, replicating agent._see exactly -- including its deliberate
    approximation of computing attack masks against the ORIGINAL occupancy, so captures do not
    reveal x-ray attackers behind the piece that just moved."""
    frm = move & 63
    to = (move >> 6) & 63
    frm_bb = BB_SQ[frm]
    to_bb = BB_SQ[to]
    occ_full = occw | occb

    is_ep = bool(p & frm_bb) and to == ep and not ((occb if wtm else occw) & to_bb)
    first_victim = 100 if is_ep else PIECE_VALUE[_piece_type_at(to_bb, p, n, b, r, q, k)]

    gains = np.empty(40, dtype=np.int64)
    gains[0] = first_victim
    length = 1

    occupied = occ_full & ~frm_bb
    if is_ep:
        occupied &= ~(BB_SQ[to - 8] if wtm else BB_SQ[to + 8])
    on_square = PIECE_VALUE[_piece_type_at(frm_bb, p, n, b, r, q, k)]

    att_w = _attackers(to, occ_full, True, p, n, b, r, q, k, occw, occb)
    att_b = _attackers(to, occ_full, False, p, n, b, r, q, k, occw, occb)
    side_white = not wtm

    while True:
        attackers = (att_w if side_white else att_b) & occupied
        if attackers == 0:
            break
        best_bb = np.int64(0)
        best_val = 1 << 30
        remaining = attackers
        while remaining:
            low = remaining & -remaining
            remaining ^= low
            val = PIECE_VALUE[_piece_type_at(low, p, n, b, r, q, k)]
            if val < best_val:
                best_bb, best_val = low, val
        gains[length] = on_square - gains[length - 1]
        length += 1
        on_square = best_val
        occupied &= ~best_bb
        side_white = not side_white

    for i in range(length - 2, -1, -1):
        neg = -gains[i]
        nxt = gains[i + 1]
        gains[i] = -(neg if neg > nxt else nxt)
    return int(gains[0])


# ---------------------------------------------------------------------------
# The quiescence search itself
# ---------------------------------------------------------------------------


@njit(cache=False, fastmath=True, nogil=True)
def _eval_here(
    p: int, n: int, b: int, r: int, q: int, kk: int,
    occw: int, occb: int, wtm: bool, dual: bool,
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
) -> float:
    """Mirror of agent.evaluate: logit -> centipawns, truncated to int, clamped."""
    if dual:
        logit = evalnet._forward_dual_bb(
            p, n, b, r, q, kk, occw, occb, wtm, w1, b1, w2, b2, w3, b3)
    else:
        us = occw if wtm else occb
        them = occb if wtm else occw
        logit = evalnet._forward_bb(
            p, n, b, r, q, kk, us, them, not wtm, w1, b1, w2, b2, w3, b3)
    cp = logit * EVAL_SCALE
    if cp > EVAL_CAP:
        cp = EVAL_CAP
    elif cp < -EVAL_CAP:
        cp = -EVAL_CAP
    return float(int(cp))


@njit(cache=False, nogil=True)
def _quiesce_bb(
    p: int, n: int, b: int, r: int, q: int, k: int,
    occw: int, occb: int, wtm: bool, ep: int,
    alpha: float, beta: float, qdepth: int, ply: int,
    qs_max_depth: int, delta_margin: int, dual: bool,
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
) -> float:
    """agent.quiesce, move for move: stand pat, delta and SEE pruning on captures, every
    evasion when in check, mate scores relative to `ply`."""
    us = occw if wtm else occb
    kbb = k & us
    ksq = _bit_index(kbb & -kbb)
    occ = occw | occb
    in_check = _attackers(ksq, occ, not wtm, p, n, b, r, q, k, occw, occb) != 0

    # numba does not bounds-check; 256 clears the ~218 known maximum with promo fan-out slack.
    moves = np.empty(256, dtype=np.int64)

    if in_check:
        cnt = _gen_evasions(moves, p, n, b, r, q, k, occw, occb, wtm, ep)
        if qdepth >= qs_max_depth:
            # Any legal evasion means "give up and evaluate here"; none means mate.
            for i in range(cnt):
                np_, nn, nb, nr, nq, nk, nw, nbl, _ = _make(
                    moves[i] & _KEY_MOVE_MASK, p, n, b, r, q, k, occw, occb, wtm, ep)
                if _own_king_safe(np_, nn, nb, nr, nq, nk, nw, nbl, wtm):
                    return _eval_here(p, n, b, r, q, k, occw, occb, wtm, dual,
                                      w1, b1, w2, b2, w3, b3)
            return float(-MATE + ply)
        _sort_desc(moves, cnt)
        any_legal = False
        for i in range(cnt):
            mv = moves[i] & _KEY_MOVE_MASK
            np_, nn, nb, nr, nq, nk, nw, nbl, nep = _make(
                mv, p, n, b, r, q, k, occw, occb, wtm, ep)
            if not _own_king_safe(np_, nn, nb, nr, nq, nk, nw, nbl, wtm):
                continue
            any_legal = True
            score = -_quiesce_bb(np_, nn, nb, nr, nq, nk, nw, nbl, not wtm, nep,
                                 -beta, -alpha, qdepth + 1, ply + 1,
                                 qs_max_depth, delta_margin, dual,
                                 w1, b1, w2, b2, w3, b3)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        if not any_legal:
            return float(-MATE + ply)
        return alpha

    stand_pat = _eval_here(p, n, b, r, q, k, occw, occb, wtm, dual, w1, b1, w2, b2, w3, b3)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth >= qs_max_depth:
        return alpha

    cnt = _gen_noisy(moves, p, n, b, r, q, k, occw, occb, wtm, ep)
    _sort_desc(moves, cnt)
    them = occb if wtm else occw
    for i in range(cnt):
        mv = moves[i] & _KEY_MOVE_MASK
        to_bb = BB_SQ[(mv >> 6) & 63]
        promo = (mv >> 12) & 7
        # Delta pruning: winning the victim outright still would not reach alpha.
        if them & to_bb:
            victim_val = PIECE_VALUE[_piece_type_at(to_bb, p, n, b, r, q, k)]
        elif promo == 0 and ((mv >> 6) & 63) == ep and (p & BB_SQ[mv & 63]):
            victim_val = 100
        else:
            victim_val = 0
        if stand_pat + victim_val + delta_margin < alpha:
            continue
        # SEE pruning: skip captures that lose the exchange outright. Promotions exempt.
        if promo == 0 and _see(mv, p, n, b, r, q, k, occw, occb, wtm, ep) < 0:
            continue
        np_, nn, nb, nr, nq, nk, nw, nbl, nep = _make(
            mv, p, n, b, r, q, k, occw, occb, wtm, ep)
        if not _own_king_safe(np_, nn, nb, nr, nq, nk, nw, nbl, wtm):
            continue
        score = -_quiesce_bb(np_, nn, nb, nr, nq, nk, nw, nbl, not wtm, nep,
                             -beta, -alpha, qdepth + 1, ply + 1,
                             qs_max_depth, delta_margin, dual,
                             w1, b1, w2, b2, w3, b3)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def quiesce_board(
    board: chess.Board, alpha: float, beta: float, ply: int,
    qs_max_depth: int, delta_margin: int,
) -> float:
    """Entry point: unpack a python-chess board and run the jitted quiescence below it."""
    return float(_quiesce_bb(
        to_signed(board.pawns), to_signed(board.knights), to_signed(board.bishops),
        to_signed(board.rooks), to_signed(board.queens), to_signed(board.kings),
        to_signed(board.occupied_co[True]), to_signed(board.occupied_co[False]),
        board.turn, -1 if board.ep_square is None else board.ep_square,
        alpha, beta, 0, ply, qs_max_depth, delta_margin, evalnet.MODE_DUAL,
        evalnet.W1, evalnet.B1, evalnet.W2, evalnet.B2, evalnet.W3, evalnet.B3,
    ))


# Warm the whole tree at import so compilation lands in the init budget, never on the clock.
_b = chess.Board()
quiesce_board(_b, -1e9, 1e9, 0, 8, 200)
_b.push_uci("e2e4")
quiesce_board(_b, -1e9, 1e9, 0, 8, 200)
