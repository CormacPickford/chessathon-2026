"""Measure an agent's Elo by round-robin against the baselines, on this machine.

    uv run python training/elo.py                          # default pool, quick TC
    uv run python training/elo.py --openings 12 --base-ms 4000 --rounds 2

Every pair of agents plays each opening twice with colours reversed, so deterministic agents
still produce varied, first-move-fair games. Ratings are fit with the Bradley-Terry model
(draws as half points) and reported with bootstrap 95% intervals, anchored so `random` = 0.

Elo here is RELATIVE to this pool and depends on the time control -- it is a ladder for
tracking your own progress, not an absolute Lichess/FIDE number. To make it absolute, anchor a
pool member to a known rating (e.g. by measuring it against a locally-run rated engine).

Runs in-process because the platform harness (harness/) cannot open its pipe selector on
Windows; game semantics (flag, illegal = loss, 300-ply material adjudication) mirror the referee.
"""

import argparse
import importlib.util
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.rules import PLY_CAP

# Opening lines as SAN; each is played from both sides. Trimmed with --openings.
OPENING_LINES = [
    "e4 e5",
    "e4 c5",
    "e4 e6",
    "e4 c6",
    "e4 e5 Nf3 Nc6 Bb5",
    "e4 e5 Nf3 Nc6 Bc4",
    "d4 d5 c4 e6",
    "d4 d5 c4 c6",
    "d4 Nf6 c4 g6",
    "d4 Nf6 c4 e6",
    "c4 e5",
    "c4 c5",
    "Nf3 d5 g3",
    "e4 d5",
    "d4 f5",
    "e4 g6",
]

ADJ_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


@dataclass
class Player:
    name: str
    path: Path
    module: ModuleType


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def opening_fens(count: int) -> list[str]:
    fens = []
    for line in OPENING_LINES[:count]:
        board = chess.Board()
        for token in line.split():
            board.push_san(token)
        fens.append(board.fen())
    return fens


def adjudicate(board: chess.Board) -> float:
    balance = sum(
        v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
        for p, v in ADJ_VALUES.items()
    )
    if balance > 0:
        return 1.0
    if balance < 0:
        return 0.0
    return 0.5


def play_game(white: Player, black: Player, base_ms: int, inc_ms: int, start_fen: str) -> float:
    """Return White's score: 1.0 win, 0.0 loss, 0.5 draw. Mirrors the referee's rules."""
    board = chess.Board(start_fen)
    agents = {chess.WHITE: white, chess.BLACK: black}
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    while True:
        finish = board.outcome(claim_draw=True)
        if finish is not None:
            return 0.5 if finish.winner is None else float(finish.winner == chess.WHITE)
        if len(board.move_stack) >= PLY_CAP:
            return adjudicate(board)

        mover = board.turn
        t0 = time.monotonic()
        try:
            uci = agents[mover].module.get_move(board.fen(), int(clock[mover]))
        except Exception:
            return float(mover == chess.BLACK)
        clock[mover] -= (time.monotonic() - t0) * 1000.0
        if clock[mover] < 0:
            return float(mover == chess.BLACK)
        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            return float(mover == chess.BLACK)
        if move not in board.legal_moves:
            return float(mover == chess.BLACK)
        board.push(move)
        clock[mover] += inc_ms


def fit_elo(n: int, games: list[tuple[int, int, float]], iters: int = 400) -> np.ndarray:
    """Bradley-Terry MLE via MM iteration. games = (white_idx, black_idx, white_score)."""
    wins = np.zeros(n)
    pair = np.zeros((n, n))
    for i, j, s in games:
        wins[i] += s
        wins[j] += 1.0 - s
        pair[i, j] += 1
        pair[j, i] += 1
    gamma = np.ones(n)
    for _ in range(iters):
        denom = (pair / (gamma[:, None] + gamma[None, :] + 1e-12)).sum(axis=1)
        new = np.where(wins > 0, wins / np.where(denom > 0, denom, 1.0), gamma)
        new = np.clip(new, 1e-9, None)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(gamma))) < 1e-9:
            gamma = new
            break
        gamma = new
    return 400.0 * np.log10(gamma)


def sf_dials(names: list[str]) -> list[tuple[int, int]]:
    """(index, nominal Elo) for players named sfNNNN."""
    return [(i, int(n[2:])) for i, n in enumerate(names) if n.startswith("sf") and n[2:].isdigit()]


def make_anchor(names: list[str], anchor: str) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Choose the rating scale. With Stockfish dials present, shift the fitted ratings so they
    best match the dials' known Elos (absolute). Otherwise pin `anchor` (or the mean) to 0."""
    dials = sf_dials(names)
    if dials:
        idx = np.array([i for i, _ in dials])
        nominal = np.array([e for _, e in dials], dtype=np.float64)
        label = "absolute, calibrated to Stockfish " + "/".join(str(e) for _, e in dials)
        return (lambda elo: elo + float((nominal - elo[idx]).mean())), label
    if anchor in names:
        a = names.index(anchor)
        return (lambda elo: elo - elo[a]), f"relative, anchor {anchor}=0"
    return (lambda elo: elo - elo.mean()), "relative, mean=0"


def bootstrap_ci(
    n: int,
    games: list[tuple[int, int, float]],
    anchor: Callable[[np.ndarray], np.ndarray],
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    g = np.array(games, dtype=np.float64)
    draws = np.empty((samples, n))
    for b in range(samples):
        idx = rng.integers(0, len(g), len(g))
        sample = [(int(i), int(j), s) for i, j, s in g[idx]]
        draws[b] = anchor(fit_elo(n, sample))
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin Elo vs the baselines.")
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--opponents", nargs="+", default=[
        "baselines/random", "baselines/greedy", "baselines/minimax", "baselines/numba"])
    parser.add_argument("--openings", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=1, help="repeat the whole schedule N times")
    parser.add_argument("--base-ms", type=int, default=2000)
    parser.add_argument("--inc-ms", type=int, default=100)
    parser.add_argument("--anchor", default="random")
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    specs = [("candidate", args.candidate)] + [(Path(o).name, Path(o)) for o in args.opponents]
    players = [Player(name, path, load_module(f"agent_{k}", path)) for k, (name, path) in
               enumerate(specs)]
    names = [p.name for p in players]
    fens = opening_fens(args.openings)
    n = len(players)

    schedule = [(i, j, fen) for i in range(n) for j in range(n) if i < j for fen in fens]
    schedule = schedule * args.rounds
    total = len(schedule) * 2
    print(f"{n} players, {len(fens)} openings, {args.rounds} round(s) -> {total} games "
          f"@ {args.base_ms}ms+{args.inc_ms}ms\n")

    games: list[tuple[int, int, float]] = []
    wdl = np.zeros((n, 3))  # wins, draws, losses per player
    done = 0
    t0 = time.monotonic()
    for i, j, fen in schedule:
        for white, black in ((i, j), (j, i)):
            s = play_game(players[white], players[black], args.base_ms, args.inc_ms, fen)
            games.append((white, black, s))
            wdl[white, 0 if s == 1 else 1 if s == 0.5 else 2] += 1
            wdl[black, 0 if s == 0 else 1 if s == 0.5 else 2] += 1
            done += 1
        if done % 20 == 0 or done == total:
            print(f"  {done}/{total} games ({done / (time.monotonic() - t0):.1f}/s)", flush=True)

    anchor_fn, scale = make_anchor(names, args.anchor)
    elo = anchor_fn(fit_elo(n, games))
    lo, hi = bootstrap_ci(n, games, anchor_fn, args.bootstrap)

    order = np.argsort(-elo)
    print(f"\n{'agent':<12}{'elo':>7}{'95% ci':>16}{'games':>7}{'score':>8}   W-D-L")
    for k in order:
        w, d, ll = wdl[k]
        g = w + d + ll
        score = (w + 0.5 * d) / g * 100 if g else 0
        star = "  *" if names[k] == "candidate" else ""
        print(f"{names[k]:<12}{elo[k]:7.0f}  [{lo[k]:5.0f},{hi[k]:5.0f}]{int(g):7d}"
              f"{score:7.1f}%   {int(w)}-{int(d)}-{int(ll)}{star}")

    dials = sf_dials(names)
    if dials:
        resid = [elo[i] - nom for i, nom in dials]
        rms = float(np.sqrt(np.mean(np.square(resid))))
        print("\nStockfish calibration (fitted vs nominal Elo):")
        for (i, nom), r in zip(dials, resid, strict=True):
            print(f"  {names[i]:<8} fitted {elo[i]:5.0f}  nominal {nom}  ({r:+.0f})")
        print(f"  residual RMS {rms:.0f} Elo (smaller = more self-consistent)")
    print(f"\nScale: {scale}, at {args.base_ms}ms+{args.inc_ms}ms. * = your agent.")


if __name__ == "__main__":
    main()
