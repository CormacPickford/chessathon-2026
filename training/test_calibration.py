"""Is the eval's magnitude compressed? Compare predicted vs true centipawns by bucket.

    uv run python training/test_calibration.py --data training/data_partial39pct.npz \
        --new weights/model.onnx --old opponents/prev_mvv/weights/model.onnx

"Compressed magnitudes" means the model under-states how won a won position is: ask it about
a +800 position and it says +300. That is invisible in an overall MAE, which is dominated by
the near-equal bulk, but it matters for play -- an engine that cannot tell +300 from +900 will
trade down out of winning positions.

So bucket positions by their TRUE score and report the mean prediction per bucket. A
calibrated model tracks the diagonal. Not part of the submission.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import EVAL_SCALE, TRAIN_SCALE, codes_to_features

# The pre-win-probability net emitted pawns; agent.py multiplied by 100.
OLD_SCALE = 100.0
BUCKETS = [(-2001, -800), (-800, -400), (-400, -150), (-150, -50), (-50, 50),
           (50, 150), (150, 400), (400, 800), (800, 2001)]


def predict(path: Path, feats: np.ndarray, scale: float, batch: int = 8192) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = np.empty(len(feats), dtype=np.float64)
    for i in range(0, len(feats), batch):
        chunk = feats[i : i + batch]
        out[i : i + len(chunk)] = np.asarray(sess.run(None, {name: chunk})[0]).reshape(-1)
    return out * scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval magnitude calibration.")
    parser.add_argument("--data", type=Path, default=Path("training/data_partial39pct.npz"))
    parser.add_argument("--new", type=Path, default=Path("training/model.onnx"))
    parser.add_argument("--old", type=Path, default=None)
    parser.add_argument("--n", type=int, default=60_000)
    args = parser.parse_args()

    blob = np.load(args.data)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(blob["scores"]), size=min(args.n, len(blob["scores"])), replace=False)
    codes = blob["codes"][idx]
    true_cp = blob["scores"][idx].astype(np.float64)
    feats = codes_to_features(codes)
    print(f"{len(true_cp):,} positions from {args.data}\n")

    new = predict(args.new, feats, EVAL_SCALE)
    old = predict(args.old, feats, OLD_SCALE) if args.old else None

    header = f"{'true cp bucket':<18}{'n':>8}{'true mean':>11}{'new pred':>11}"
    if old is not None:
        header += f"{'old pred':>11}"
    print(header)
    for lo, hi in BUCKETS:
        m = (true_cp >= lo) & (true_cp < hi)
        if not m.any():
            continue
        bucket = f"[{lo:>5},{hi:>5})"
        row = (f"{bucket:<18}{m.sum():>8,}{true_cp[m].mean():>11.0f}"
               f"{new[m].mean():>11.0f}")
        if old is not None:
            row += f"{old[m].mean():>11.0f}"
        print(row)

    def slope(pred: np.ndarray) -> float:
        """Least-squares slope of prediction against truth. 1.0 = calibrated, <1 = compressed."""
        return float(np.polyfit(true_cp, pred, 1)[0])

    print(f"\n{'':<18}{'MAE':>11}{'slope':>11}{'|pred| mean':>13}")
    print(f"{'new':<18}{np.abs(new - true_cp).mean():>11.0f}{slope(new):>11.3f}"
          f"{np.abs(new).mean():>13.0f}")
    if old is not None:
        print(f"{'old':<18}{np.abs(old - true_cp).mean():>11.0f}{slope(old):>11.3f}"
              f"{np.abs(old).mean():>13.0f}")
    print(f"{'truth':<18}{0:>11}{1.0:>11.3f}{np.abs(true_cp).mean():>13.0f}")
    print("\nslope < 1 means the model shrinks toward zero: the compression this step targets.")

    # Centipawn error is the wrong court for a win-probability model: it is dominated by
    # positions that are already decided, where the probability barely moves. Score both
    # models on probabilities too, and on the only thing search really consumes -- whether
    # the eval ORDERS two positions the same way the truth does.
    def to_wp(cp_vals: np.ndarray) -> np.ndarray:
        # TRAIN_SCALE, not EVAL_SCALE: this is the target's own definition of win probability.
        return 1.0 / (1.0 + np.exp(-cp_vals / TRAIN_SCALE))

    true_wp = to_wp(true_cp)
    rng2 = np.random.default_rng(1)
    a = rng2.integers(0, len(true_cp), 400_000)
    b = rng2.integers(0, len(true_cp), 400_000)
    decisive = np.abs(true_cp[a] - true_cp[b]) > 30  # skip ties the truth cannot separate

    print(f"\n{'':<18}{'wp MAE':>11}{'pairwise order':>16}")
    for label, pred in (("new", new), ("old", old)):
        if pred is None:
            continue
        wp_mae = np.abs(to_wp(pred) - true_wp).mean()
        agree = ((pred[a] > pred[b]) == (true_cp[a] > true_cp[b]))[decisive].mean()
        print(f"{label:<18}{wp_mae:>11.4f}{agree * 100:>15.1f}%")
    print("\nPairwise order is the metric closest to what alpha-beta actually needs: given two")
    print("positions, does the eval rank them the way the truth does? Magnitude is irrelevant")
    print("to move choice under minimax, which is invariant to any monotone rescaling.")


if __name__ == "__main__":
    main()
