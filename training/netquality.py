"""Rank a trained .pt eval net by the metrics that predict playing strength, cheaply.

    uv run python training/netquality.py --model training/model.pt --data training/data.npz

Reports, on a held-out slice:
  - win-prob MAE (the training target's own error)
  - PAIRWISE ORDERING: given two decisive positions, does the net rank them like Stockfish?
    This is the metric that has tracked game strength here, since minimax only consumes order.
  - calibration SLOPE of the raw logit against true cp, which is what EVAL_SCALE must invert
    for delta pruning to keep its tuned margin (features.py: EVAL_SCALE = TRAIN_SCALE / slope).

Architecture-agnostic: it reads hidden sizes from the checkpoint, so it ranks any 3-layer MLP.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import TRAIN_SCALE, codes_to_dual_features, codes_to_features
from training.model import EvalNet
from training.model2 import AccEvalNet


def load_any(path: Path) -> tuple[torch.nn.Module, bool]:
    """Load either a single-perspective EvalNet or a dual-perspective AccEvalNet.

    Returns (model, is_dual). The state-dict keys are the tell: `l1.*` is the dual net.
    """
    sd = torch.load(path, map_location="cpu")
    if "l1.weight" in sd:
        model: torch.nn.Module = AccEvalNet(h1=sd["l1.bias"].shape[0], h2=sd["l2.bias"].shape[0])
        model.load_state_dict(sd)
        model.eval()
        print(f"loaded {path}  arch 768->{sd['l1.bias'].shape[0]} (x2 persp) ->"
              f"{sd['l2.bias'].shape[0]}->1")
        return model, True
    model = EvalNet(hidden1=sd["net.0.bias"].shape[0], hidden2=sd["net.2.bias"].shape[0])
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {path}  arch 768->{sd['net.0.bias'].shape[0]}->{sd['net.2.bias'].shape[0]}->1")
    return model, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank an eval net by strength-predictive metrics.")
    parser.add_argument("--model", type=Path, default=Path("training/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("training/data.npz"))
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model, is_dual = load_any(args.model)
    blob = np.load(args.data)
    rng = np.random.default_rng(args.seed)
    # Use a tail slice as held-out; train.py shuffles with its own seed, so this is an
    # approximate holdout, adequate for ranking (not for a headline number).
    n_total = len(blob["scores"])
    idx = rng.choice(n_total, size=min(args.n, n_total), replace=False)
    codes = blob["codes"][idx]
    true_cp = blob["scores"][idx].astype(np.float64)

    logits = np.empty(len(codes), dtype=np.float64)
    with torch.no_grad():
        if is_dual:
            us_all, them_all = codes_to_dual_features(codes)
            for i in range(0, len(codes), 16384):
                logits[i : i + 16384] = model(
                    torch.from_numpy(us_all[i : i + 16384]),
                    torch.from_numpy(them_all[i : i + 16384])).numpy()
        else:
            feats = codes_to_features(codes)
            for i in range(0, len(feats), 16384):
                chunk = torch.from_numpy(feats[i : i + 16384])
                logits[i : i + len(chunk)] = model(chunk).numpy()

    def to_wp(cp: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-cp / TRAIN_SCALE))

    true_wp = to_wp(true_cp)
    pred_wp = 1.0 / (1.0 + np.exp(-logits))
    wp_mae = np.abs(pred_wp - true_wp).mean()

    a = rng.integers(0, len(true_cp), 1_000_000)
    b = rng.integers(0, len(true_cp), 1_000_000)
    decisive = np.abs(true_cp[a] - true_cp[b]) > 30
    agree = ((logits[a] > logits[b]) == (true_cp[a] > true_cp[b]))[decisive].mean()

    slope = float(np.polyfit(true_cp, logits, 1)[0])  # logit per cp
    eval_scale = 1.0 / slope if slope > 0 else float("inf")

    print(f"\npositions {len(true_cp):,}")
    print(f"win-prob MAE     {wp_mae:.4f}")
    print(f"pairwise order   {agree * 100:.2f}%   (higher = ranks positions like Stockfish)")
    print(f"logit/cp slope   {slope:.5f}  -> EVAL_SCALE ~= {eval_scale:.0f} "
          f"(currently set to import from features)")


if __name__ == "__main__":
    main()
