"""Generate a diverse set of realistic middlegame/endgame FENs for quality testing.

    uv run python training/gen_positions.py --n 300 --out training/testpos.txt

Plays lightly-randomised games from the opening book lines (biased toward decent moves via a
shallow material greedy, with random noise) and records positions sampled from move 12..60.
The point is a spread of non-trivial, not-yet-decided positions to measure move quality on --
not self-play strength. Deterministic given --seed.
"""

import argparse
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OPENINGS = [
    "e4 e5", "e4 c5", "e4 e6", "e4 c6", "d4 d5 c4 e6", "d4 Nf6 c4 g6",
    "c4 e5", "Nf3 d5 g3", "e4 e5 Nf3 Nc6 Bb5", "d4 d5 c4 c6", "e4 g6", "d4 f5",
]
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board: chess.Board) -> int:
    return sum(_VAL.get(p.piece_type, 0) * (1 if p.color == board.turn else -1)
               for p in board.piece_map().values())


def pick(board: chess.Board, rng: random.Random) -> chess.Move:
    """A shallow, noisy move: usually grab the best 1-ply material move, sometimes random."""
    moves = list(board.legal_moves)
    if rng.random() < 0.35:
        return rng.choice(moves)
    best, best_score = moves[0], -1e9
    for m in moves:
        board.push(m)
        score = -material(board) + rng.uniform(-0.5, 0.5)
        board.pop()
        if score > best_score:
            best, best_score = m, score
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Generate test FENs.")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("training/testpos.txt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    fens: list[str] = []
    while len(fens) < args.n:
        board = chess.Board()
        for tok in rng.choice(OPENINGS).split():
            board.push_san(tok)
        plies = rng.randint(12, 60)
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(pick(board, rng))
        if board.is_game_over() or board.is_check():
            continue
        # Skip near-decided positions: those add noise, not signal, to a cp-loss average.
        if abs(material(board)) > 6:
            continue
        fens.append(board.fen())

    args.out.write_text("\n".join(fens) + "\n", encoding="utf-8")
    print(f"wrote {len(fens)} FENs to {args.out}")


if __name__ == "__main__":
    main()
