"""Train the evaluation MLP on the sampled Lichess positions.

    uv run python training/train.py --data training/data.npz --epochs 20

Targets are centipawns from the mover's point of view, scaled to pawns (/100) and fit with
a Huber loss so the decisive tails do not dominate. Reports validation MAE in centipawns and
saves the best model to training/model.pt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import codes_to_features
from training.model import EvalNet

SCALE = 100.0  # centipawns per pawn


def batches(n: int, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def to_features(codes: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(codes_to_features(codes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the eval MLP.")
    parser.add_argument("--data", type=Path, default=Path("training/data.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path("training/model.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    blob = np.load(args.data)
    codes = blob["codes"]
    scores = blob["scores"].astype(np.float32) / SCALE  # pawns, mover POV
    n = len(scores)
    print(f"loaded {n:,} positions from {args.data}")

    perm = np.random.permutation(n)
    codes, scores = codes[perm], scores[perm]
    n_val = int(n * args.val_frac)
    val_codes, val_scores = codes[:n_val], scores[:n_val]
    tr_codes, tr_scores = codes[n_val:], scores[n_val:]
    val_x = to_features(val_codes)
    val_y = torch.from_numpy(val_scores)
    print(f"train {len(tr_scores):,}  val {n_val:,}  threads {torch.get_num_threads()}")

    model = EvalNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()  # Huber, delta = 1 pawn

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        running = 0.0
        seen = 0
        for idx in batches(len(tr_scores), args.batch_size, shuffle=True):
            x = to_features(tr_codes[idx])
            y = torch.from_numpy(tr_scores[idx])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = torch.empty_like(val_y)
            for i in range(0, len(val_y), args.batch_size):
                pred[i : i + args.batch_size] = model(val_x[i : i + args.batch_size])
            val_mae_cp = (pred - val_y).abs().mean().item() * SCALE
            corr = np.corrcoef(pred.numpy(), val_y.numpy())[0, 1]

        dt = time.monotonic() - t0
        flag = ""
        if val_mae_cp < best_mae:
            best_mae = val_mae_cp
            torch.save(model.state_dict(), args.out)
            flag = "  <- saved"
        print(
            f"epoch {epoch:2d}  train_huber {running / seen:.4f}  "
            f"val_MAE {val_mae_cp:6.1f}cp  corr {corr:.3f}  {dt:4.0f}s{flag}",
            flush=True,
        )

    print(f"\nbest val MAE {best_mae:.1f}cp  saved to {args.out}")


if __name__ == "__main__":
    main()
