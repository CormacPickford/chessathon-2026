"""Shared driver for the fixed-Elo Stockfish calibration opponents.

Each opponents/sfNNNN/agent.py calls build(NNNN) to get a get_move that plays at that UCI_Elo.
One Stockfish process is started per Elo, lazily, and reused across all games. LOCAL ONLY --
this never ships (see fetch_stockfish.py).
"""

import atexit
from collections.abc import Callable
from pathlib import Path

import chess
import chess.engine

EXE = Path(__file__).resolve().parent / "engines" / "stockfish.exe"

# Stockfish's UCI_Elo is clamped to this range; requests outside it would be rejected.
MIN_ELO = 1320
MAX_ELO = 3190

_engines: dict[int, chess.engine.SimpleEngine] = {}


def _engine(elo: int) -> chess.engine.SimpleEngine:
    if elo not in _engines:
        engine = chess.engine.SimpleEngine.popen_uci(str(EXE))
        engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16}
        )
        _engines[elo] = engine
        atexit.register(engine.quit)
    return _engines[elo]


def build(elo: int, think_ms: int = 100) -> Callable[[str, int], str]:
    elo = max(MIN_ELO, min(MAX_ELO, elo))

    def get_move(fen: str, time_left_ms: int) -> str:
        board = chess.Board(fen)
        # A small think time is plenty; UCI_Elo caps strength regardless. Never flag.
        seconds = min(think_ms, max(10, int(time_left_ms * 0.02))) / 1000.0
        result = _engine(elo).play(board, chess.engine.Limit(time=seconds))
        assert result.move is not None
        return result.move.uci()

    return get_move
