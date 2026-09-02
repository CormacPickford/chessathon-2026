"""Stockfish pinned to ~2200 Elo. Local calibration opponent -- never ships."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opponents.sf_engine import build

get_move = build(2200)
