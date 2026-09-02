"""Generate opponents/sfNNNN/agent.py wrappers, one per target Elo.

    uv run python opponents/make_dials.py            # default dials
    uv run python opponents/make_dials.py 1500 2000  # custom dials
"""

import sys
from pathlib import Path

DEFAULT_DIALS = [1400, 1600, 1800, 2000, 2200]
ROOT = Path(__file__).resolve().parent

TEMPLATE = '''\
"""Stockfish pinned to ~{elo} Elo. Local calibration opponent -- never ships."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opponents.sf_engine import build

get_move = build({elo})
'''


def main() -> None:
    dials = [int(a) for a in sys.argv[1:]] or DEFAULT_DIALS
    for elo in dials:
        d = ROOT / f"sf{elo}"
        d.mkdir(exist_ok=True)
        (d / "agent.py").write_text(TEMPLATE.format(elo=elo))
        print(f"wrote {d / 'agent.py'}")


if __name__ == "__main__":
    main()
