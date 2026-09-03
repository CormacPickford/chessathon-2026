"""Compare two sampled datasets to see whether the sampling change moved the population.

    uv run python training/compare_data.py --a training/data.npz --b training/data_partial39pct.npz

Sampling fixes are easy to claim and easy to get wrong, and a biased sample looks perfectly
healthy on its own. Putting two samples side by side is what makes the difference visible:
if whole-file coverage matters, the score distribution, material balance and game-phase mix
should all shift. Not part of the submission.
"""

import argparse
from pathlib import Path

import numpy as np

_MATERIAL = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int64)


def stats(path: Path) -> dict[str, float]:
    blob = np.load(path)
    codes = blob["codes"]
    scores = blob["scores"].astype(np.float64)

    # Piece counts per position: how far into the game these positions sit.
    occupied = (codes > 0).sum(axis=1)
    # Material balance from the mover's POV.
    ours = np.isin(codes, [1, 2, 3, 4, 5]).astype(np.int64)
    theirs = np.isin(codes, [7, 8, 9, 10, 11]).astype(np.int64)
    vals = np.zeros_like(codes, dtype=np.int64)
    for code in range(1, 6):
        vals += (codes == code) * _MATERIAL[code]
    for code in range(7, 12):
        vals -= (codes == code) * _MATERIAL[code - 6]
    balance = vals.sum(axis=1).astype(np.float64)

    return {
        "positions": float(len(scores)),
        "score mean": float(scores.mean()),
        "score std": float(scores.std()),
        "|score| mean": float(np.abs(scores).mean()),
        "frac |score|>500": float((np.abs(scores) > 500).mean()),
        "frac near equal (<50)": float((np.abs(scores) < 50).mean()),
        "frac at clamp (2000)": float((np.abs(scores) >= 2000).mean()),
        "pieces on board": float(occupied.mean()),
        "frac endgame (<=12 pc)": float((occupied <= 12).mean()),
        "material balance mean": float(balance.mean()),
        "material/score corr": float(np.corrcoef(balance, scores)[0, 1]),
        "ours/theirs ratio": float(ours.sum() / max(1, theirs.sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two sampled datasets.")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    sa, sb = stats(args.a), stats(args.b)
    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'metric':<26}{'A':>14}{'B':>14}{'A-B':>14}")
    for key in sa:
        d = sa[key] - sb[key]
        print(f"{key:<26}{sa[key]:>14,.3f}{sb[key]:>14,.3f}{d:>+14,.3f}")
    print("\nA whole-file sample and a partial-coverage sample drawn from the same dump should")
    print("differ only by noise. Large gaps mean coverage was doing real work.")


if __name__ == "__main__":
    main()
