"""Perft parity: the jitted _gen_all + _make_full + legality vs python-chess.

    uv run python training/test_perft.py

Perft (counting legal leaf nodes to a fixed depth) is the standard proof that a move generator,
make, and legality test are all correct together -- a single wrong, missing, or extra move
anywhere in the tree changes the count. These are the primitives the jitted interior negamax
will be built on, and castling + castling-rights tracking are new here, so they are checked
against python-chess (whose own move generation is trusted) across positions chosen to stress
castling, en passant, promotions, and pins.

Not part of the submission.
"""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qsearch  # noqa: E402


def ref_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += ref_perft(board, depth - 1)
        board.pop()
    return total


# (name, fen, max depth). The classic perft suite -- startpos, kiwipete, and the endgame/
# promotion/castling positions from the Chess Programming Wiki -- plus a couple of castling-
# rights edge cases (rook captured on its home square must drop that right).
CASES = [
    ("startpos", chess.STARTING_FEN, 4),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4),
    ("position4-mirror",
     "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 4),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4),
    ("position6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4),
    ("castle-rights-rook-capture",
     "r3k2r/8/8/8/8/8/6p1/R3K2R w KQkq - 0 1", 4),
    ("black-to-move-castle", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", 3),
]


def main() -> None:
    ok = True
    for name, fen, depth in CASES:
        board = chess.Board(fen)
        for d in range(1, depth + 1):
            want = ref_perft(board.copy(), d)
            got = qsearch.perft(board.copy(), d)
            status = "OK" if want == got else "MISMATCH"
            if want != got:
                ok = False
            print(f"{name:<28} depth {d}: python {want:>10}  jit {got:>10}  {status}")
        print()
    if not ok:
        print("PERFT MISMATCH -- the jitted make/gen/legality is wrong somewhere")
        sys.exit(1)
    print("all perft counts match python-chess")


if __name__ == "__main__":
    main()
