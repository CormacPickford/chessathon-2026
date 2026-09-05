"""Score parity: jitted _negamax_plain vs a python-chess negamax with identical ordering.

    uv run python training/test_negamax.py

The jitted plain negamax and this reference share three things exactly: the legal move set (a
perft match, see test_perft.py), the move ORDER (both sort by the same packed key, which is a
total order so there are no ties), and the leaf evaluation (both call the jitted quiescence).
Plain alpha-beta over an identical ordering with identical leaf values must return bit-identical
scores -- so any mismatch is a bug in the jitted recursion (make, legality, mate handling, or
the alpha-beta itself). This is the correctness gate before TT and pruning, which will
deliberately diverge from this reference, are layered on.

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
from qsearch import _SCORE_OFFSET  # noqa: E402

random.seed(20260905)

_PV = [0, 100, 320, 330, 500, 900, 20_000]


def sort_key(board: chess.Board, move: chess.Move) -> int:
    """The packed sort key _gen_all emits, reproduced for the python-chess reference."""
    noisy = board.is_capture(move) or move.promotion is not None
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        vt = board.piece_type_at(move.to_square)
        victim = vt if vt is not None else 0
    attacker = board.piece_type_at(move.from_square) or 0
    score = _PV[victim] * 16 - _PV[attacker] + _SCORE_OFFSET
    tier = 1 if noisy else 0
    promo = move.promotion or 0
    packed = (promo << 12) | (move.to_square << 6) | move.from_square
    return (tier << 40) | (score << 16) | packed


def ref_negamax(board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
    """Plain alpha-beta on python-chess, same order and same jitted leaf eval as the jitted core."""
    if depth == 0:
        return qsearch.quiesce_board(board, alpha, beta, ply, agent.QS_MAX_DEPTH,
                                     agent.DELTA_MARGIN)
    any_legal = False
    for move in sorted(board.legal_moves, key=lambda m: sort_key(board, m), reverse=True):
        any_legal = True
        board.push(move)
        score = -ref_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    if not any_legal:
        return float(-agent.MATE + ply) if board.is_check() else 0.0
    return alpha


def random_positions(n_games: int) -> list[chess.Board]:
    boards: list[chess.Board] = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(random.randint(6, 80)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(random.choice(moves))
            boards.append(board.copy())
    return boards


TRICKY = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
    "2K5/8/8/8/4k3/8/8/3q4 w - - 0 1",
    "8/8/8/8/1k6/8/K1p5/8 b - - 0 1",
]


def main() -> None:
    boards = random.sample(random_positions(60), 400)
    boards += [chess.Board(f) for f in TRICKY]
    print(f"{len(boards)} positions")

    windows = [(-1e9, 1e9), (-300.0, 300.0), (-50.0, 50.0)]
    for depth in (2, 3, 4):
        bad = 0
        t0 = time.perf_counter()
        for board in boards:
            a, bta = random.choice(windows)
            want = ref_negamax(board.copy(), depth, a, bta, 0)
            got = qsearch.negamax_plain_board(board.copy(), depth, a, bta, 0,
                                              agent.QS_MAX_DEPTH, agent.DELTA_MARGIN)
            if want != got:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH d{depth} {board.fen()}  ref={want} jit={got}")
        dt = time.perf_counter() - t0
        print(f"depth {depth}: {len(boards) - bad}/{len(boards)} match  ({dt:.1f}s)")
        if bad:
            sys.exit(1)
    print("jitted plain negamax is score-identical to the python-chess reference")


if __name__ == "__main__":
    main()
