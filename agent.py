"""The submission entrypoint: a learned evaluation with alpha-beta search.

The position is scored by a small MLP (see training/) exported to weights/model.onnx and run
with onnxruntime. The network is trained to predict Stockfish-annotated centipawns from the
side-to-move's point of view, which is exactly what a negamax search consumes at its leaves.
The search itself is plain iterative-deepening alpha-beta with capture-first move ordering.
"""

import math
import time
from pathlib import Path

import chess
import onnxruntime as ort  # type: ignore[import-untyped]

from features import board_to_codes, codes_to_features

MATE = 1_000_000
SCALE = 100.0  # the network outputs pawns; multiply to centipawns

# Import time runs once per game inside a 60 second budget, before the clock starts. Build the
# inference session here, pinned to one thread (the platform gives us one core), and warm it up.
_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "model.onnx"
_OPTS = ort.SessionOptions()
_OPTS.intra_op_num_threads = 1
_OPTS.inter_op_num_threads = 1
_SESSION = ort.InferenceSession(
    str(_MODEL_PATH), sess_options=_OPTS, providers=["CPUExecutionProvider"]
)
_INPUT = _SESSION.get_inputs()[0].name


def evaluate(board: chess.Board) -> int:
    """Network score in centipawns, from the side-to-move's point of view."""
    feats = codes_to_features(board_to_codes(board)).reshape(1, -1)
    pawns = float(_SESSION.run(None, {_INPUT: feats})[0][0])
    return int(pawns * SCALE)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_deadline: float = 0.0
_timed_out: bool = False


def _order(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Captures first, then quiet moves -- cheap ordering that helps alpha-beta prune."""
    return sorted(moves, key=board.is_capture, reverse=True)


def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    global _timed_out
    if time.monotonic() > _deadline:
        _timed_out = True
        return 0.0

    moves = list(board.legal_moves)
    if not moves:
        return float(-MATE) if board.is_check() else 0.0
    if depth == 0:
        return float(evaluate(board))

    for move in _order(board, moves):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if _timed_out:
            return alpha
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return alpha


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    global _deadline, _timed_out

    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if len(moves) == 1:
        return moves[0].uci()

    # Spend ~5% of the remaining clock, floored at 50ms and capped at 4s.
    budget_ms = max(50.0, min(time_left_ms * 0.05, 4000.0))
    _deadline = time.monotonic() + budget_ms / 1000.0

    ordered = _order(board, moves)
    best_move = ordered[0]

    for depth in range(1, 40):
        _timed_out = False
        alpha = -math.inf
        candidate = best_move

        # Search the previous iteration's best move first for tighter pruning.
        search_order = [best_move, *(m for m in ordered if m != best_move)]
        for move in search_order:
            board.push(move)
            score = -negamax(board, depth - 1, -math.inf, -alpha)
            board.pop()
            if _timed_out:
                break
            if score > alpha:
                alpha = score
                candidate = move

        if not _timed_out:
            best_move = candidate
        else:
            break

    return best_move.uci()


# Warm up the session at import so the first real move is not the one that pays for it.
evaluate(chess.Board())
