"""Is quiescence generating every legal move just to keep the captures?

    uv run python training/test_movegen.py

`_noisy_moves` currently filters `board.legal_moves`, which generates and legality-checks
every quiet move first and then throws them away. python-chess can generate captures directly.
Quiescence nodes are the bulk of the tree, so this is measured before the eval encoding.
Not part of the submission.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N = 20_000
POSITIONS = {
    "startpos (no captures)": chess.STARTING_FEN,
    "open middlegame": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "tactical (many captures)":
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}

print(f"{'position':<26}{'filter all':>13}{'gen captures':>15}{'speedup':>10}{'same?':>8}")
for label, fen in POSITIONS.items():
    board = chess.Board(fen)

    t0 = time.perf_counter()
    for _ in range(N):
        a = [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
    filt = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        b = list(board.generate_legal_captures())
    gen = (time.perf_counter() - t0) / N * 1e6

    # generate_legal_captures omits non-capturing promotions; check what the difference is.
    same = set(a) == set(b)
    print(f"{label:<26}{filt:>11.1f}us{gen:>13.1f}us{filt / gen:>9.2f}x{same!s:>8}")

print("\nnote: generate_legal_captures() omits QUIET queen promotions (a pawn pushing to the")
print("last rank without capturing). Those are worth searching in quiescence, so a correct")
print("replacement adds them back -- cheaply, since they only exist with a pawn on rank 7.")

board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
caps = list(board.generate_legal_captures())
promos = [m for m in board.legal_moves if m.promotion == chess.QUEEN]
print("\nprobe position with a quiet promotion available:")
print(f"  generate_legal_captures(): {[m.uci() for m in caps]}")
print(f"  quiet queen promotions   : {[m.uci() for m in promos]}  <- would be missed")
