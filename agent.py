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

import chess

import evalnet
from features import EVAL_SCALE

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

# Where a pawn must stand to promote next move; used to skip the promotion scan entirely in
# the vast majority of positions.
_PROMO_RANK = {chess.WHITE: chess.BB_RANK_7, chess.BLACK: chess.BB_RANK_2}

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
TT_MAX = 400_000  # entries; ~100 MB of dict overhead, well inside the 2 GB limit
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


def _see(board: chess.Board, move: chess.Move) -> int:
    """Net material won by `move` after both sides trade optimally on that square.

    Quiescence currently searches every capture, including queen-takes-defended-pawn, which
    cannot possibly be good. Playing the exchange out with the cheapest available attacker
    each time tells us that for the cost of a few attack lookups instead of a whole subtree.

    Returns centipawns from the mover's point of view; negative means the exchange loses
    material. Approximate by design: it ignores pins and does not notice that a capture might
    expose a bigger threat, which is exactly why it only ever prunes CLEARLY losing captures.
    """
    target = move.to_square
    gain = [_victim_value(board, move)]
    attacker_sq = move.from_square
    attacker_piece = board.piece_type_at(attacker_sq)
    if attacker_piece is None:
        return 0

    occupied = board.occupied & ~chess.BB_SQUARES[attacker_sq]
    if board.is_en_passant(move):
        occupied &= ~chess.BB_SQUARES[board.ep_square or 0]
    side = not board.turn
    on_square = PIECE_VALUE[attacker_piece]

    while True:
        attackers = board.attackers_mask(side, target) & occupied
        if not attackers:
            break
        # Recapture with the least valuable attacker: it risks the least on the square.
        best_sq, best_val = -1, 1 << 30
        remaining = attackers
        while remaining:
            low = remaining & -remaining
            remaining ^= low
            sq = low.bit_length() - 1
            piece = board.piece_type_at(sq)
            if piece is not None and PIECE_VALUE[piece] < best_val:
                best_sq, best_val = sq, PIECE_VALUE[piece]
        if best_sq < 0:
            break
        gain.append(on_square - gain[-1])
        on_square = best_val
        occupied &= ~chess.BB_SQUARES[best_sq]
        side = not side

    # Walk back up: at each point the side to move could simply decline to recapture.
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = -max(-gain[i], gain[i + 1])
    return gain[0]


def _noisy_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and queen promotions -- the moves that can swing material -- best first.

    Generating captures directly rather than filtering every legal move matters here because
    quiescence is most of the tree: full generation costs 100us in a tactical position against
    23us for captures alone. `generate_legal_captures` omits QUIET promotions, so those are
    added back, guarded by a bitboard test that is false in almost every position.
    """
    moves = list(board.generate_legal_captures())
    promo_pawns = board.pawns & board.occupied_co[board.turn] & _PROMO_RANK[board.turn]
    if promo_pawns:
        moves.extend(
            m for m in board.generate_legal_moves(from_mask=promo_pawns)
            if m.promotion == chess.QUEEN and not board.is_capture(m)
        )
    moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    return moves


def quiesce(board: chess.Board, alpha: float, beta: float, qdepth: int, ply: int = 0) -> float:
    """Search on past the horizon until the position is quiet, then evaluate.

    Calling the net in the middle of an exchange is what makes an engine hang pieces: the leaf
    looks a pawn up because the recapture sits one ply out of sight. So instead of evaluating
    the moment depth runs out, keep playing captures (and, when in check, every evasion) until
    nothing is hanging, and ask the net only there.

    Stalemate is not detected here -- when not in check we only generate captures, so we cannot
    tell "no captures" from "no moves at all". The main search catches it at every deeper node,
    and paying for a full legal-move generation at every quiet leaf would cost more than the
    rare misjudged leaf does.
    """
    global _timed_out
    if time.perf_counter() > _deadline:
        _timed_out = True
        return 0.0

    if board.is_check():
        # Standing pat is not on offer while in check: every evasion has to be tried, quiet
        # ones included, or the search would call a forced position quiet and misjudge it.
        moves = list(board.legal_moves)
        if not moves:
            return float(-MATE + ply)
        if qdepth >= QS_MAX_DEPTH:
            return float(evaluate(board))
        for move in _order(board, moves, ply):
            board.push(move)
            score = -quiesce(board, -beta, -alpha, qdepth + 1, ply + 1)
            board.pop()
            if _timed_out:
                return alpha
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # Not in check the side to move can always decline to capture, so the static score is a
    # lower bound on what the position is worth -- the "stand pat" score.
    stand_pat = float(evaluate(board))
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth >= QS_MAX_DEPTH:
        return alpha

    for move in _noisy_moves(board):
        # Delta pruning: if winning the victim outright still falls short of alpha, the line is
        # hopeless and not worth a network call.
        if stand_pat + _victim_value(board, move) + DELTA_MARGIN < alpha:
            continue
        # SEE pruning: a capture that loses material once the square is fought over is not
        # worth a subtree. Promotions are exempt -- the promoted piece is the point, and SEE
        # only counts the piece that moved.
        if move.promotion is None and _see(board, move) < 0:
            continue
        board.push(move)
        score = -quiesce(board, -beta, -alpha, qdepth + 1, ply + 1)
        board.pop()
        if _timed_out:
            return alpha
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _has_non_pawn_material(board: chess.Board) -> bool:
    """Does the side to move have a piece other than pawns and the king?

    Null move pruning assumes having the move is worth something, which is false in zugzwang
    -- positions where every legal move makes things worse. Zugzwang is overwhelmingly a
    pawns-and-king endgame phenomenon, so requiring a real piece is the standard guard.
    """
    us = board.occupied_co[board.turn]
    return bool(us & (board.knights | board.bishops | board.rooks | board.queens))


def negamax(
    board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0,
    can_null: bool = True,
) -> float:
    global _timed_out
    if time.perf_counter() > _deadline:
        _timed_out = True
        return 0.0

    if depth == 0:
        return quiesce(board, alpha, beta, 0, ply)

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

    # Reverse futility: if the static score is already so far above beta that even a big
    # concession could not drag it back, take the cheap exit instead of searching. Skipped in
    # check and near mate scores, where the static eval means little.
    if (
        not in_check
        and depth <= RFP_MAX_DEPTH
        and beta < MATE - 10_000
        and evaluate(board) - RFP_MARGIN * depth >= beta
    ):
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

    moves = list(board.legal_moves)
    if not moves:
        # Mate scores shrink with distance from the root, so a mate in one outranks a mate in
        # six and the engine actually finishes won games instead of shuffling.
        return float(-MATE + ply) if in_check else 0.0

    ordered = _order(board, moves, ply)
    # The stored move was best here last time, so it is the likeliest cutoff. Membership is
    # checked rather than assumed: keys are 64-bit hashes and a collision would otherwise let
    # an illegal move through, which loses the game outright.
    if tt_move is not None and tt_move in moves:
        ordered = [tt_move, *(m for m in ordered if m != tt_move)]

    # Futility: at shallow depth, a quiet move from a position already far below alpha is very
    # unlikely to rescue it. Computed once per node, not once per move.
    futile = (
        not in_check
        and depth <= FUTILITY_MAX_DEPTH
        and alpha > -MATE + 10_000
        and evaluate(board) + FUTILITY_MARGIN * depth <= alpha
    )

    best_move: chess.Move | None = None
    for i, move in enumerate(ordered):
        is_quiet = not board.is_capture(move) and move.promotion is None
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

    budget_ms = _budget(board, time_left_ms, len(moves))
    _deadline = time.perf_counter() + budget_ms / 1000.0

    best_move, _ = _search_root(board, ordered, best_move)

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
    board: chess.Board, ordered: list[chess.Move], best_move: chess.Move
) -> tuple[chess.Move, float]:
    """Iterative deepening with aspiration windows. Returns the best move and its score."""
    global _timed_out
    score = 0.0

    for depth in range(1, MAX_PLY):
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
            best_move, score = candidate, alpha
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
