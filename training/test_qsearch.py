"""Verify qsearch.py against python-chess and against a reference Python quiescence.

    uv run python training/test_qsearch.py

Three layers, in order of what would hurt:

1. **Move-set parity.** Over thousands of positions from random games, the jitted noisy
   generator (legality-filtered) must produce exactly the set agent._noisy_moves does, and the
   evasion generator exactly list(board.legal_moves) when in check. A miss here is a wrong
   score with no other symptom.
2. **Score parity.** A Python quiescence that mirrors agent.quiesce but orders moves by the
   jitted code's exact deterministic key must return bit-identical scores for random windows.
   This exercises make, en passant, promotions, SEE, delta pruning, mate scores and the eval
   call together.
3. **Throughput.** The point of the exercise.

Not part of the submission.
"""

import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import qsearch  # noqa: E402
from qsearch import _KEY_MOVE_MASK, _SCORE_OFFSET  # noqa: E402

random.seed(20260904)


def packed(move: chess.Move) -> int:
    promo = 0
    if move.promotion is not None and move.promotion != chess.KING:
        promo = move.promotion if move.promotion != chess.QUEEN else 5
        promo = move.promotion  # piece types already 2..5
    return (promo << 12) | (move.to_square << 6) | move.from_square


def sort_key(board: chess.Board, move: chess.Move) -> int:
    """The jitted generators' exact ordering key, reproduced move-for-move."""
    noisy = board.is_capture(move) or move.promotion is not None
    victim = 0
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        vt = board.piece_type_at(move.to_square)
        if vt is not None:
            victim = vt
    attacker = board.piece_type_at(move.from_square) or 0
    pv = [0, 100, 320, 330, 500, 900, 20_000]
    score = pv[victim] * 16 - pv[attacker] + _SCORE_OFFSET
    tier = 1 if noisy else 0
    # Quiet under-promotions sit in the evasion generator's quiet tier with a pawn attacker
    # and no victim, exactly like the jitted _pack call that emits them.
    if move.promotion is not None and not board.is_capture(move):
        tier = 1 if move.promotion == chess.QUEEN else 0
    return (tier << 40) | (score << 16) | packed(move)


def py_noisy(board: chess.Board) -> list[chess.Move]:
    """The pure python-chess noisy generator agent._noisy_moves used before the jitted one --
    the independent reference, since agent's own is now jit-backed."""
    moves = list(board.generate_legal_captures())
    promo_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
    promo_pawns = board.pawns & board.occupied_co[board.turn] & promo_rank
    if promo_pawns:
        moves.extend(
            m for m in board.generate_legal_moves(from_mask=promo_pawns)
            if m.promotion == chess.QUEEN and not board.is_capture(m)
        )
    return moves


def ref_see(board: chess.Board, move: chess.Move) -> int:
    """The Python SEE the jitted one replaced, kept here as the reference implementation --
    including its deliberate quirk of computing attack masks against the original occupancy."""
    target = move.to_square
    gain = [agent._victim_value(board, move)]
    attacker_piece = board.piece_type_at(move.from_square)
    if attacker_piece is None:
        return 0
    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        occupied &= ~chess.BB_SQUARES[board.ep_square or 0]
    side = not board.turn
    on_square = agent.PIECE_VALUE[attacker_piece]
    while True:
        attackers = board.attackers_mask(side, target) & occupied
        if not attackers:
            break
        best_sq, best_val = -1, 1 << 30
        remaining = attackers
        while remaining:
            low = remaining & -remaining
            remaining ^= low
            sq = low.bit_length() - 1
            piece = board.piece_type_at(sq)
            if piece is not None and agent.PIECE_VALUE[piece] < best_val:
                best_sq, best_val = sq, agent.PIECE_VALUE[piece]
        if best_sq < 0:
            break
        gain.append(on_square - gain[-1])
        on_square = best_val
        occupied &= ~chess.BB_SQUARES[best_sq]
        side = not side
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = -max(-gain[i], gain[i + 1])
    return gain[0]


def ref_quiesce(board: chess.Board, alpha: float, beta: float, qdepth: int, ply: int) -> float:
    """agent.quiesce with the jitted ordering, no deadline, no killers/history."""
    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return float(-agent.MATE + ply)
        if qdepth >= agent.QS_MAX_DEPTH:
            return float(agent.evaluate(board))
        for move in sorted(moves, key=lambda m: sort_key(board, m), reverse=True):
            board.push(move)
            score = -ref_quiesce(board, -beta, -alpha, qdepth + 1, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    stand_pat = float(agent.evaluate(board))
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth >= agent.QS_MAX_DEPTH:
        return alpha

    noisy = py_noisy(board)
    for move in sorted(noisy, key=lambda m: sort_key(board, m), reverse=True):
        if stand_pat + agent._victim_value(board, move) + agent.DELTA_MARGIN < alpha:
            continue
        if move.promotion is None and ref_see(board, move) < 0:
            continue
        board.push(move)
        score = -ref_quiesce(board, -beta, -alpha, qdepth + 1, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def jit_moves(board: chess.Board, evasions: bool) -> set[tuple[int, int, int]]:
    """Legality-filtered move set from the jitted generators, as (from, to, promo) tuples."""
    import numpy as np

    from evalnet import to_signed

    out = np.empty(256, dtype=np.int64)
    args = (
        to_signed(board.pawns), to_signed(board.knights), to_signed(board.bishops),
        to_signed(board.rooks), to_signed(board.queens), to_signed(board.kings),
        to_signed(board.occupied_co[True]), to_signed(board.occupied_co[False]),
        board.turn, -1 if board.ep_square is None else board.ep_square,
    )
    gen = qsearch._gen_evasions if evasions else qsearch._gen_noisy
    cnt = gen(out, *args)
    result = set()
    for i in range(cnt):
        mv = int(out[i]) & _KEY_MOVE_MASK
        child = qsearch._make(mv, *args)
        if qsearch._own_king_safe(child[0], child[1], child[2], child[3], child[4],
                                  child[5], child[6], child[7], board.turn):
            result.add((mv & 63, (mv >> 6) & 63, (mv >> 12) & 7))
    return result


def py_set(moves: list[chess.Move]) -> set[tuple[int, int, int]]:
    return {(m.from_square, m.to_square, m.promotion or 0) for m in moves}


def random_positions(n_games: int) -> list[chess.Board]:
    boards = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(random.randint(10, 120)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(random.choice(moves))
            boards.append(board.copy())
    return boards


# Positions chosen to hit the traps a bitboard generator gets wrong: en passant pins (the two
# pawns leaving the rank expose a rook/queen), discovered checks, promotions of every flavour,
# pinned pieces, and kiwipete for general coverage.
TRICKY_FENS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3pPk/8/4P3/8 b - g3 0 1",       # ep would expose Black's king to Rb4? no: classic ep-pin family
    "8/8/3p4/1Pp4r/1K3R2/6k1/4P1P1/8 w - c6 0 3",     # white ep pinned horizontally
    "8/8/3p4/KPp4r/5R2/6k1/4P1P1/8 w - c6 0 3",
    "k7/8/8/2pP4/8/8/8/K2R4 w - c6 0 2",              # legal ep
    "k2r4/8/8/2pP4/8/8/8/K7 w - c6 0 2",              # ep leaves king file attack? (vertical pin on d-file)
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",        # promotion festival
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",           # castling must NOT appear in quiets
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "2K5/8/8/8/4k3/8/8/3q4 w - - 0 1",                # king in check, few evasions
    "8/8/8/8/1k6/8/K1p5/8 b - - 0 1",                 # push-promo near enemy king
    "4k3/8/8/8/8/8/r6P/4K3 w - - 0 1",
]


def main() -> None:
    boards = random_positions(120)
    boards.extend(chess.Board(f) for f in TRICKY_FENS)
    print(f"{len(boards)} positions from random games + {len(TRICKY_FENS)} trap positions")

    import numpy as np

    checked = noisy_ok = quiet_ok = evas = evas_ok = 0
    buf = np.empty(256, dtype=np.int64)
    for board in boards:
        if board.is_check():
            evas += 1
            mine = jit_moves(board, evasions=True)
            ref = py_set(list(board.legal_moves))
            if mine == ref:
                evas_ok += 1
            else:
                print(f"EVASION MISMATCH {board.fen()}")
                print(f"  jit-only: {mine - ref}\n  ref-only: {ref - mine}")
        else:
            checked += 1
            mine = jit_moves(board, evasions=False)
            ref = py_set(py_noisy(board))
            if mine == ref:
                noisy_ok += 1
            else:
                print(f"NOISY MISMATCH {board.fen()}")
                print(f"  jit-only: {mine - ref}\n  ref-only: {ref - mine}")
        # The legal_* entry points are what the interior search PUSHES; verify them on every
        # position, in check or not (staged movegen runs at in-check nodes too).
        cnt = qsearch.legal_noisy(buf, board)
        mine = {(int(buf[i]) & 63, (int(buf[i]) >> 6) & 63, (int(buf[i]) >> 12) & 7)
                for i in range(cnt)}
        cnt = qsearch.legal_quiets(buf, board)
        mine |= {(int(buf[i]) & 63, (int(buf[i]) >> 6) & 63, (int(buf[i]) >> 12) & 7)
                 for i in range(cnt)}
        ref_all = py_set([m for m in board.legal_moves if not board.is_castling(m)])
        if mine == ref_all:
            quiet_ok += 1
        else:
            print(f"LEGAL SET MISMATCH {board.fen()}")
            print(f"  jit-only: {mine - ref_all}\n  ref-only: {ref_all - mine}")
    print(f"noisy movegen:   {noisy_ok}/{checked} positions match")
    print(f"evasion movegen: {evas_ok}/{evas} positions match")
    print(f"legal noisy+quiets == all legal minus castling: {quiet_ok}/{len(boards)}")
    if noisy_ok != checked or evas_ok != evas or quiet_ok != len(boards):
        sys.exit(1)

    # Score parity on a subsample, with assorted windows including the full one.
    agent._deadline = time.perf_counter() + 3600.0
    sample = random.sample(boards, min(1500, len(boards)))
    windows = [(-1e9, 1e9), (-200.0, 200.0), (-50.0, 50.0), (0.0, 1.0)]
    bad = 0
    for board in sample:
        a, bta = random.choice(windows)
        want = ref_quiesce(board, a, bta, 0, 0)
        got = qsearch.quiesce_board(board, a, bta, 0, agent.QS_MAX_DEPTH, agent.DELTA_MARGIN)
        if want != got:
            bad += 1
            if bad <= 5:
                print(f"SCORE MISMATCH {board.fen()}  window=({a},{bta})  "
                      f"ref={want} jit={got}")
    print(f"score parity:    {len(sample) - bad}/{len(sample)} positions match")
    if bad:
        sys.exit(1)

    # Throughput: same positions, python vs jitted.
    bench = random.sample([b for b in sample if not b.is_check()], 300)
    t0 = time.perf_counter()
    for board in bench:
        ref_quiesce(board, -1e9, 1e9, 0, 0)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    for board in bench:
        qsearch.quiesce_board(board, -1e9, 1e9, 0, agent.QS_MAX_DEPTH, agent.DELTA_MARGIN)
    t_jit = time.perf_counter() - t0
    print(f"throughput:      python {t_py * 1e6 / len(bench):.1f}us/tree, "
          f"jitted {t_jit * 1e6 / len(bench):.1f}us/tree  ({t_py / t_jit:.2f}x)")


if __name__ == "__main__":
    main()
