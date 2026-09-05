"""The submission entrypoint: a learned evaluation with alpha-beta search.

The position is scored by a small MLP (see training/) whose weights ship as weights/net.pt and
run through the numba forward pass in evalnet.py. The network predicts the side-to-move's
winning chances from Stockfish-annotated positions, read back as centipawns, which is exactly
what a negamax search consumes at its leaves. The search is iterative-deepening alpha-beta
whose leaves run a quiescence search, so the net is only ever asked to judge positions where
nothing is hanging.
"""

import math
import threading
import time
from collections.abc import Iterator

import chess
import numpy as np

from . import evalnet
from . import qsearch
from .features import EVAL_SCALE

MATE = 1_000_000
# The net emits a logit (sigmoid of it is the mover's win probability); EVAL_SCALE converts
# that back to centipawns. EVAL_CAP keeps a saturated output well clear of the mate scores, so
# "completely winning" can never be confused with "mate found".
EVAL_CAP = 20_000

# Quiescence search dials.
QS_MAX_DEPTH = 8  # insurance against pathological capture chains; captures alone terminate
DELTA_MARGIN = 200  # centipawns of slack before a capture is written off as hopeless

# Clock safety. Measured ~12ms of per-move work sits outside the deadline loop (board setup,
# move ordering, the first root iteration), so the budget leaves that much headroom and below
# PANIC_MS we skip the search entirely rather than lose on time.
OVERHEAD_MS = 25.0
PANIC_MS = 40

# Shallow-depth pruning margins, in centipawns per remaining ply. Generous on purpose: these
# discard positions without looking at them, so they should only fire when the verdict is not
# in doubt. All are disabled in check and near mate scores, where a static score means little.
RFP_MAX_DEPTH = 4  # reverse futility ("we are already winning by too much to bother")
RFP_MARGIN = 120
FUTILITY_MAX_DEPTH = 3  # futility ("a quiet move will not rescue this")
FUTILITY_MARGIN = 150
CHECK_EXT_MAX_DEPTH = 6  # extend checks only near the leaves, or the tree explodes

# Root search. The score between iterations moves little, so search a narrow band around the
# last one and widen only on a miss.
ASPIRATION_WINDOW = 50.0

# Instability-aware overrun. The search normally stops at the soft budget from `_budget`. But
# when the best move keeps CHANGING between iterations, that is exactly the position where one
# more ply is most likely to change the choice again -- so the deadline is extended toward a
# hard multiple of the soft budget, and only then. A settled position (best move unchanged) is
# never granted the extension, so easy moves stay cheap. This only ever ADDS search on unstable
# moments and never removes any, so it cannot make a move worse; the cost is some extra clock,
# which the hard clamp in get_move keeps clear of a flag fall. OVERRUN applies from depth 5 up,
# below which iterations are nearly free and the best move is not yet meaningful.
OVERRUN_FACTOR = 1.6  # an unstable search may run to this multiple of the soft budget
OVERRUN_MIN_DEPTH = 5

# Subtrees at this depth or below run entirely in the jitted negamax (qsearch.py) instead of
# the Python search, trading the Python TT/killers/history at those shallow nodes for
# compiled-code speed with no python-chess board. The jitted core carries its own null move,
# LMR, futility, RFP, PVS and check extensions, so the tree stays pruned. 3 is the sweet spot:
# depth <= 3 is already the great majority of nodes, so going higher barely adds speed while
# removing the TT from more of the tree. 0 = disabled (old behaviour). Tuned by A/B.
JIT_LEAF_DEPTH = 3
STABLE_ITERS = 2  # consecutive same-best iterations before the choice counts as settled

# Pondering, on the opponent's clock. Bounded so a background search cannot outlive the
# opponent's turn by much, and skipped in time trouble where the join would itself cost too
# much. PONDER_JOIN_S is generous: the thread checks the deadline at every node, so it should
# unwind in microseconds, and this only ever matters if something has gone wrong.
PONDER_MS = 8_000.0
PONDER_MIN_CLOCK_MS = 2_000
PONDER_JOIN_S = 0.5
# Ships enabled. Turned off only for local A/B runs: training/elo.py plays both agents in ONE
# process, so a background search here would steal the core from the opponent's thinking and
# flatter us. On the platform each agent has its own container, so pondering costs the
# opponent nothing and its real value cannot be measured locally either way.
PONDER_ENABLED = True

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

# Importing evalnet loads the weights and compiles the jitted forward pass. That happens once
# per game inside the platform's 60 second init budget, before the clock starts.


def evaluate(board: chess.Board) -> int:
    """Network score in centipawns, from the side-to-move's point of view."""
    turn = board.turn
    logit = evalnet.forward_board(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[turn], board.occupied_co[not turn], turn == chess.BLACK,
    )
    cp = logit * EVAL_SCALE
    return int(max(-EVAL_CAP, min(EVAL_CAP, cp)))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_deadline: float = 0.0
_timed_out: bool = False
_ponder_thread: threading.Thread | None = None
_ponder_stop: bool = False

# Transposition table: position key -> (depth, flag, score, best move). Move orders reach the
# same position constantly, and quiescence revisits near-identical ones, so the same subtree
# gets searched over and over. Caching it buys two things: whole searches skipped, and a
# known-good move to try first at nodes we cannot skip.
_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2
TT_MAX = 1_000_000  # entries; ~250 MB of dict overhead, well inside the 2 GB limit. Bigger is
# better here because the table is wiped WHOLESALE when it fills (see negamax), so a larger cap
# means far fewer of those wipes over a long game and more surviving cutoffs at platform depth.
_TT: dict[int, tuple[int, int, float, chess.Move | None]] = {}

# Quiet-move ordering. Captures order themselves by MVV-LVA, but quiet moves had no ordering
# at all, and they are the majority of every node's move list -- so the cutoff was landing
# late and the tree paid for it.
#
# Killers: the last two quiet moves that caused a cutoff at this ply. Sibling positions are
# nearly identical, so a refutation that worked in one usually works in the next.
# History: a running score per (from, to) square pair across the whole search, which
# generalises the same idea across plies -- moves that keep causing cutoffs get tried first.
MAX_PLY = 64
_KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
_HISTORY: list[int] = [0] * 4096  # indexed from_square * 64 + to_square


def _tt_key(board: chess.Board) -> int:
    """Hash of everything that makes two positions the same for search purposes.

    Built from public bitboards rather than `chess.polyglot.zobrist_hash`, which costs 74us --
    as much as a whole evaluation -- and would have spent more than the table saves. This is
    ~4us. Castling rights and the en passant square are included because they change what is
    legal, so positions that differ in them are not interchangeable.
    """
    return hash((
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.castling_rights, board.ep_square,
    ))


def _order(board: chess.Board, moves: list[chess.Move], ply: int = 0) -> list[chess.Move]:
    """Noisy moves first by MVV-LVA, then killers, then quiet moves by history score.

    A plain captures-first split already prunes well; ordering *within* each group is what
    makes the cutoff land on the first move tried rather than the fifth. The tier number keeps
    the groups apart because an MVV-LVA score alone would sort a king capture (a cheap victim
    taken by the most expensive attacker) below the quiet moves.
    """
    killers = _KILLERS[ply] if ply <= MAX_PLY else (None, None)

    def key(move: chess.Move) -> tuple[int, int]:
        if board.is_capture(move) or move.promotion == chess.QUEEN:
            return (2, _mvv_lva(board, move))
        if move == killers[0] or move == killers[1]:
            return (1, 0)
        return (0, _HISTORY[move.from_square * 64 + move.to_square])

    return sorted(moves, key=key, reverse=True)


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    """Centipawn value of the piece `move` captures; en passant takes a pawn off a square
    the move does not land on, so it needs its own case."""
    if board.is_en_passant(move):
        return PIECE_VALUE[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    return PIECE_VALUE[victim] if victim is not None else 0


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most Valuable Victim / Least Valuable Attacker: pawn-takes-queen before queen-takes-pawn.

    Trying the captures most likely to be good first makes the beta cutoffs come early, which
    is most of what keeps the quiescence tree small.
    """
    attacker = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
    return _victim_value(board, move) * 16 - attacker_value


def _cached_move(packed: int) -> chess.Move:
    """chess.Move for a packed (from | to<<6 | promo<<12) int, memoised.

    The jitted generators hand back packed ints; the search still wants Move objects for
    push/pop and the TT. Constructing one costs more than the generation did, so each distinct
    move is built once per session. Bounded: there are only ~4k from-to pairs times a few
    promotion types, and equality/hashing of a cached Move matches a fresh one exactly.
    """
    move = _MOVE_CACHE.get(packed)
    if move is None:
        promo = (packed >> 12) & 7
        move = chess.Move(packed & 63, (packed >> 6) & 63, promo or None)
        _MOVE_CACHE[packed] = move
    return move


_MOVE_CACHE: dict[int, chess.Move] = {}


def _noisy_moves(board: chess.Board) -> list[chess.Move]:
    """Legal captures and queen promotions -- the moves that can swing material -- best first.

    Generated by the jitted bitboard generator in qsearch.py (pseudo-legal + king-safety
    filter), which is what quiescence runs below the horizon; these are the same packed moves
    converted to chess.Move for the interior search. Set-verified move-for-move against
    python-chess by training/test_qsearch.py, because the search pushes what this returns and
    an illegal move loses the game outright.
    """
    buf = np.empty(256, dtype=np.int64)
    cnt = qsearch.legal_noisy(buf, board)
    return [_cached_move(int(buf[i]) & 0xFFFF) for i in range(cnt)]


def quiesce(board: chess.Board, alpha: float, beta: float, ply: int) -> float:
    """Search on past the horizon until the position is quiet, then evaluate.

    Calling the net in the middle of an exchange is what makes an engine hang pieces: the leaf
    looks a pawn up because the recapture sits one ply out of sight. So instead of evaluating
    the moment depth runs out, keep playing captures (and, when in check, every evasion) until
    nothing is hanging, and ask the net only there.

    The whole subtree runs jitted in qsearch.py -- generation, make, SEE, delta pruning and
    the eval never touch a Python object, which is worth ~8x on the bulk of the search tree.
    Verified move-for-move and score-for-score against the Python original by
    training/test_qsearch.py. The clock is checked at every negamax entry rather than inside
    the jitted tree; a single quiescence subtree is bounded by QS_MAX_DEPTH and capture
    exhaustion, far inside the OVERHEAD_MS headroom.
    """
    return qsearch.quiesce_board(board, alpha, beta, ply, QS_MAX_DEPTH, DELTA_MARGIN)


def _has_non_pawn_material(board: chess.Board) -> bool:
    """Does the side to move have a piece other than pawns and the king?

    Null move pruning assumes having the move is worth something, which is false in zugzwang
    -- positions where every legal move makes things worse. Zugzwang is overwhelmingly a
    pawns-and-king endgame phenomenon, so requiring a real piece is the standard guard.
    """
    us = board.occupied_co[board.turn]
    return bool(us & (board.knights | board.bishops | board.rooks | board.queens))


def _staged_moves(
    board: chess.Board, tt_move: chess.Move | None, ply: int
) -> Iterator[tuple[chess.Move, bool]]:
    """Yield (move, is_quiet) in cutoff-friendly order, generating quiet moves last and only
    when reached.

    A full `list(board.legal_moves)` is the single most expensive thing a node does, and the
    quiet moves -- the bulk of it -- are wasted work at the many nodes that cut off on the TT
    move or a capture. So the phases here are generated lazily: the transposition-table move
    first, then captures and queen promotions by MVV-LVA, then the killer moves, and only if
    none of those has caused a cutoff does the caller pull on the quiet moves, which are the
    one generation this defers. Captures land on opponent-occupied squares and quiets on empty
    ones, so the two sets are disjoint and together exactly cover the legal moves; TT and
    killer moves are de-duplicated so nothing is searched twice. Legality of the TT and killer
    moves is checked directly, which is far cheaper than generating the whole list to look them
    up and closes the hash-collision hole that an illegal move would otherwise open.
    """
    yielded: set[chess.Move] = set()

    if tt_move is not None and board.is_legal(tt_move):
        yielded.add(tt_move)
        yield tt_move, not board.is_capture(tt_move) and tt_move.promotion is None

    for move in _noisy_moves(board):  # captures + queen promotions, MVV-LVA order
        if move not in yielded:
            yielded.add(move)
            yield move, False

    if ply <= MAX_PLY:
        for killer in _KILLERS[ply]:
            if killer is not None and killer not in yielded and board.is_legal(killer):
                yielded.add(killer)
                yield killer, not board.is_capture(killer) and killer.promotion is None

    # Quiet moves from the jitted generator, which knows nothing of castling -- python-chess
    # supplies that separately, and only when someone still has the right, which is rare.
    buf = np.empty(256, dtype=np.int64)
    qcnt = qsearch.legal_quiets(buf, board)
    quiets = []
    for i in range(qcnt):
        move = _cached_move(int(buf[i]) & 0xFFFF)
        if move not in yielded:
            quiets.append(move)
    if board.castling_rights:  # skip the castling generator's work in the vast majority of nodes
        quiets.extend(m for m in board.generate_castling_moves() if m not in yielded)
    quiets.sort(key=lambda m: _HISTORY[m.from_square * 64 + m.to_square], reverse=True)
    for move in quiets:
        # These land on empty squares, so none is a capture; only a pawn under-promotion is
        # noisy, and it is the one quiet-phase move that should not be reduced or pruned.
        yield move, move.promotion is None


def negamax(
    board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0,
    can_null: bool = True,
) -> float:
    global _timed_out
    if time.perf_counter() > _deadline:
        _timed_out = True
        return 0.0

    if depth == 0:
        return quiesce(board, alpha, beta, ply)

    # Hand the shallowest subtrees -- the vast majority of interior nodes -- to the fully
    # jitted plain negamax, which runs the whole subtree (make, generation, legality,
    # quiescence) without a python-chess object. It carries no TT/pruning, so it is only worth
    # it where the subtree is small; above JIT_LEAF_DEPTH the Python search keeps its TT, null
    # move, LMR and futility. JIT_LEAF_DEPTH=0 reproduces the old behaviour exactly. The jitted
    # core is proven score-identical to a plain Python search by training/test_negamax.py, and
    # its move generation perft-matches python-chess.
    if depth <= JIT_LEAF_DEPTH:
        return qsearch.negamax_pruned_board(
            board, depth, alpha, beta, ply, QS_MAX_DEPTH, DELTA_MARGIN)

    alpha_orig = alpha
    key = _tt_key(board)
    entry = _TT.get(key)
    tt_move: chess.Move | None = None
    if entry is not None:
        e_depth, e_flag, e_score, tt_move = entry
        # Only trust a stored score that came from a search at least as deep as this one.
        if e_depth >= depth:
            if e_flag == _TT_EXACT:
                return e_score
            if e_flag == _TT_LOWER:
                alpha = max(alpha, e_score)
            else:
                beta = min(beta, e_score)
            if alpha >= beta:
                return e_score

    in_check = board.is_check()

    # The static score feeds both reverse futility and futility below. It is computed at most
    # once per node and only when one of them actually needs it -- deep nodes (past both depth
    # guards) never pay for it, and a node that checks both no longer evaluates the same
    # position twice. `evaluate` is now cheap enough (~2.3us) that this double call showed up.
    static_eval: int | None = None

    # Reverse futility: if the static score is already so far above beta that even a big
    # concession could not drag it back, take the cheap exit instead of searching. Skipped in
    # check and near mate scores, where the static eval means little.
    if not in_check and depth <= RFP_MAX_DEPTH and beta < MATE - 10_000:
        static_eval = evaluate(board)
        if static_eval - RFP_MARGIN * depth >= beta:
            return beta

    # Null move pruning: give the opponent a free move. If our position is still so good that
    # they cannot pull the score below beta even with two moves in a row, the real line is far
    # too good for them and they will avoid it -- so there is no point searching it properly.
    # Skipped in check (passing is not legal there), in likely-zugzwang endgames, and directly
    # after another null, which would just be forfeiting two moves.
    if (
        can_null
        and not in_check
        and depth >= 3
        and beta < MATE - 10_000
        and _has_non_pawn_material(board)
    ):
        reduction = 2 + (depth >= 6)
        board.push(chess.Move.null())
        # Null window: we only care whether the score reaches beta, not by how much.
        score = -negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False)
        board.pop()
        if _timed_out:
            return alpha
        if score >= beta:
            return beta

    # Futility: at shallow depth, a quiet move from a position already far below alpha is very
    # unlikely to rescue it. Computed once per node, not once per move, and reuses the static
    # score above when reverse futility already paid for it.
    futile = False
    if not in_check and depth <= FUTILITY_MAX_DEPTH and alpha > -MATE + 10_000:
        if static_eval is None:
            static_eval = evaluate(board)
        futile = static_eval + FUTILITY_MARGIN * depth <= alpha

    best_move: chess.Move | None = None
    any_move = False
    for i, (move, is_quiet) in enumerate(_staged_moves(board, tt_move, ply)):
        any_move = True
        if futile and is_quiet and i > 0 and not board.gives_check(move):
            continue  # keep at least one move so the node still returns something real

        board.push(move)
        gives_check = board.is_check()

        # Check extensions: a forcing line is worth another ply. Cheap because checks are rare,
        # and it is where tactics that quiescence cannot see tend to hide.
        extension = 1 if gives_check and depth <= CHECK_EXT_MAX_DEPTH else 0

        # Late move reductions. Move ordering puts the plausible moves first, so a move sitting
        # sixth in the list is probably not the best one -- searching it at full depth to prove
        # that is most of what the tree costs. Search it shallow instead, and only pay full
        # price if it surprises us by beating alpha. Captures, promotions, checks and the first
        # few moves are exempt, since those are exactly where surprises live.
        reduction = 0
        if is_quiet and depth >= 3 and i >= 3 and not in_check and not gives_check:
            reduction = 1 + (i >= 6)

        new_depth = depth - 1 + extension
        if i == 0:
            # Principal variation: the first move gets a full window, since it is the one most
            # likely to be best and we need its exact score.
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1)
        else:
            # Everything after it only has to prove it is NOT better than alpha, which a
            # null window settles far more cheaply.
            score = -negamax(board, new_depth - reduction, -alpha - 1, -alpha, ply + 1)
            if alpha < score < beta and not _timed_out:
                # It beat alpha, so the cheap search was not enough: get the real score.
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1)
        board.pop()

        if _timed_out:
            return alpha
        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta:
            # A quiet move that causes a cutoff is a refutation worth remembering. Weight the
            # history bonus by depth^2 so a cutoff found deep in the tree, which stood for far
            # more work, counts for more than a shallow one.
            if is_quiet and ply <= MAX_PLY:
                killers = _KILLERS[ply]
                if move != killers[0]:
                    killers[1] = killers[0]
                    killers[0] = move
                _HISTORY[move.from_square * 64 + move.to_square] += depth * depth
            break

    if not any_move:
        # No legal move was generated: checkmate if the king is attacked, else stalemate. The
        # ply-adjusted mate score makes a nearer mate outrank a farther one. Detected after the
        # fact rather than up front, so a full move generation is never paid just to notice.
        return float(-MATE + ply) if in_check else 0.0

    # Mate scores are relative to this node's ply, so they would be wrong if the same position
    # were reached at a different distance from the root. Keep them out of the table.
    if abs(alpha) < MATE - 10_000:
        if len(_TT) >= TT_MAX:
            _TT.clear()  # keep the recent, useful entries rather than growing without bound
        if alpha <= alpha_orig:
            flag = _TT_UPPER  # every move failed low: alpha is only an upper bound
        elif alpha >= beta:
            flag = _TT_LOWER  # cut off early: alpha is only a lower bound
        else:
            flag = _TT_EXACT
        _TT[key] = (depth, flag, alpha, best_move)
    return alpha


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    global _deadline, _timed_out

    # Before anything else: a background ponder search owns the search globals and would
    # otherwise fight this one for the single core we are given.
    _stop_ponder()

    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if len(moves) == 1:
        return moves[0].uci()

    ordered = _order(board, moves)
    best_move = ordered[0]

    # In severe time trouble, answering at all beats answering well: a flag fall loses the game
    # outright, and the fixed overhead below already costs more than this much clock.
    if time_left_ms <= PANIC_MS:
        return best_move.uci()

    # Soft budget: what this move deserves. Hard budget: how far the search may overrun it, and
    # only while the best move is still unstable. Both are clamped so the clock can never flag.
    clamp = max(1.0, time_left_ms - OVERHEAD_MS)
    soft_ms = min(_budget(board, time_left_ms, len(moves)), clamp)
    hard_ms = min(soft_ms * OVERRUN_FACTOR, clamp)
    start = time.perf_counter()
    _deadline = start + soft_ms / 1000.0

    best_move, _ = _search_root(board, ordered, best_move, start, soft_ms, hard_ms)

    # The clock becomes the opponent's the moment we return, so hand the idle core a head
    # start on the position we expect to be asked about next.
    if PONDER_ENABLED and time_left_ms > PONDER_MIN_CLOCK_MS:
        _ponder(fen, best_move)

    return best_move.uci()


def _budget(board: chess.Board, time_left_ms: int, n_moves: int) -> float:
    """Milliseconds to spend on this move.

    A flat fraction of the clock spends the same effort on a forced recapture as on the
    critical moment of the game. This scales that fraction by how much the position looks
    like it deserves: few legal moves means little to decide, being in check usually means
    the reply is close to forced, and a crowded position is where the game turns.
    """
    share = 0.05
    if n_moves <= 3:
        share *= 0.4  # almost nothing to choose between
    elif n_moves >= 35:
        share *= 1.3  # a lot going on; this is where games are decided
    if board.is_check():
        share *= 0.6  # evasions are usually forced

    budget = max(50.0, min(time_left_ms * share, 4000.0))
    # Never reach for more than the clock holds: board setup and the first root iteration sit
    # outside the deadline loop, so without OVERHEAD_MS headroom a nearly-empty clock flags.
    return min(budget, max(1.0, time_left_ms - OVERHEAD_MS))


def _search_root(
    board: chess.Board,
    ordered: list[chess.Move],
    best_move: chess.Move,
    start: float,
    soft_ms: float,
    hard_ms: float,
) -> tuple[chess.Move, float]:
    """Iterative deepening with aspiration windows. Returns the best move and its score.

    The deadline starts at the soft budget and is re-set after every completed iteration: a
    settled position keeps the soft deadline (so the next iteration that crosses it is aborted
    and we stop), while a position whose best move just changed is granted the hard deadline,
    buying it another ply. This never shortens the search below the soft budget, so it cannot
    make a move worse -- it only spends extra clock where the choice is still moving.
    """
    global _timed_out, _deadline
    score = 0.0

    for depth in range(1, MAX_PLY):
        if depth >= 2 and time.perf_counter() >= _deadline:
            break
        # Aspiration: the score rarely moves much between iterations, so search a narrow band
        # around the last one. A hit prunes far more; a miss costs one re-search, so the window
        # widens on failure rather than being retried at the same size.
        window = ASPIRATION_WINDOW if depth >= 4 else math.inf
        while True:
            lo = -math.inf if window == math.inf else score - window
            hi = math.inf if window == math.inf else score + window
            _timed_out = False
            alpha = lo
            candidate = best_move

            search_order = [best_move, *(m for m in ordered if m != best_move)]
            for i, move in enumerate(search_order):
                board.push(move)
                if i == 0:
                    value = -negamax(board, depth - 1, -hi, -alpha, 1)
                else:
                    value = -negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < value < hi and not _timed_out:
                        value = -negamax(board, depth - 1, -hi, -alpha, 1)
                board.pop()
                if _timed_out:
                    break
                if value > alpha:
                    alpha = value
                    candidate = move

            if _timed_out:
                return best_move, score
            if window != math.inf and (alpha <= lo or alpha >= hi):
                window *= 4  # fell outside the band: widen and redo this depth
                continue
            changed = candidate != best_move
            best_move, score = candidate, alpha
            # Grant the hard deadline only when the choice is still moving and we are deep
            # enough for the extra ply to be worth it; otherwise snap back to the soft deadline,
            # so a position that has just settled stops at the soft budget even if a previous
            # unstable iteration had extended it.
            if changed and depth >= OVERRUN_MIN_DEPTH:
                _deadline = start + hard_ms / 1000.0
            else:
                _deadline = start + soft_ms / 1000.0
            break

    return best_move, score


def _stop_ponder() -> None:
    """Halt any background search and wait for it, before touching the search globals.

    Stopping is done by moving the deadline into the past: every negamax and quiesce entry
    already tests it, so the thread unwinds within microseconds rather than needing a second
    flag threaded through the search.
    """
    global _ponder_thread, _deadline, _ponder_stop
    if _ponder_thread is not None:
        # Both signals are needed. The past deadline unwinds whatever negamax is running; the
        # flag stops the deepening loop from starting another iteration. Without the flag, the
        # thread could sail past the deadline we just set and pick up the NEW one the main
        # search is about to install, and then never stop.
        _ponder_stop = True
        _deadline = 0.0
        _ponder_thread.join(timeout=PONDER_JOIN_S)
        _ponder_thread = None


def _ponder(fen: str, our_move: chess.Move) -> None:
    """Search the position we expect next, on the opponent's clock, in the background.

    The rules allow this -- the process keeps its core while the opponent thinks, and module
    state survives to our next move in the same game. Whatever this puts in the transposition
    table is work the next real search does not repeat. A wrong guess costs nothing: those
    entries are simply never probed.

    It MUST be a background thread. Doing this inline would spend our own clock, since the
    referee charges us for the whole duration of get_move.
    """
    global _ponder_thread, _ponder_stop

    def run() -> None:
        global _deadline, _timed_out
        board = chess.Board(fen)  # our own copy; the caller's board is never shared
        board.push(our_move)
        replies = list(board.legal_moves)
        if not replies:
            return
        # Our search just told us which reply it expected; the TT holds it.
        entry = _TT.get(_tt_key(board))
        guess = entry[3] if entry is not None and entry[3] in replies else _order(
            board, replies)[0]
        board.push(guess)
        _deadline = time.perf_counter() + PONDER_MS / 1000.0
        for depth in range(1, MAX_PLY):
            if _ponder_stop:
                return
            _timed_out = False
            negamax(board, depth, -math.inf, math.inf, 2)
            if _timed_out or _ponder_stop:
                return

    _ponder_stop = False
    _ponder_thread = threading.Thread(target=run, daemon=True)
    _ponder_thread.start()


# Warm up the session at import so the first real move is not the one that pays for it.
evaluate(chess.Board())
