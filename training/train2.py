"""Train the dual-perspective accumulator net (training/model2.py) from data.npz.

    uv run python training/train2.py --data training/data.npz --epochs 22 --h1 256

Same win-probability target and MSE loss as train.py; the only difference is that each position
is fed as two feature vectors (mover and opponent perspectives), both derived from the stored
mover-POV codes. Saves the best model to --out (default training/model2.pt).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import TRAIN_SCALE, codes_to_dual_features
from training.model2 import AccEvalNet


def batches(n: int, bs: int, shuffle: bool) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + bs] for i in range(0, n, bs)]


def dual(codes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    us, them = codes_to_dual_features(codes)
    return torch.from_numpy(us), torch.from_numpy(them)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the dual-perspective accumulator net.")
    p.add_argument("--data", type=Path, default=Path("training/data.npz"))
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--out", type=Path, default=Path("training/model2.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    blob = np.load(args.data)
    codes = blob["codes"]
    cp = blob["scores"].astype(np.float32)
    scores = 1.0 / (1.0 + np.exp(-cp / TRAIN_SCALE))
    n = len(scores)
    print(f"loaded {n:,} positions; arch 768->{args.h1} (x2 persp) ->{args.h2}->1")

    perm = np.random.permutation(n)
    codes, scores = codes[perm], scores[perm]
    n_val = int(n * args.val_frac)
    val_codes, val_scores = codes[:n_val], scores[:n_val]
    tr_codes, tr_scores = codes[n_val:], scores[n_val:]
    val_us, val_them = dual(val_codes)
    val_y = torch.from_numpy(val_scores)
    print(f"train {len(tr_scores):,}  val {n_val:,}  threads {torch.get_num_threads()}")

    model = AccEvalNet(h1=args.h1, h2=args.h2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        running, seen = 0.0, 0
        for idx in batches(len(tr_scores), args.batch_size, shuffle=True):
            us, them = dual(tr_codes[idx])
            y = torch.from_numpy(tr_scores[idx])
            opt.zero_grad()
            loss = loss_fn(torch.sigmoid(model(us, them)), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = torch.empty_like(val_y)
            for i in range(0, len(val_y), args.batch_size):
                logits[i : i + args.batch_size] = model(
                    val_us[i : i + args.batch_size], val_them[i : i + args.batch_size])
            pred_wp = torch.sigmoid(logits)
            wp_mae = (pred_wp - val_y).abs().mean().item()
            corr = np.corrcoef(pred_wp.numpy(), val_y.numpy())[0, 1]
        dt = time.monotonic() - t0
        flag = ""
        if wp_mae < best:
            best = wp_mae
            torch.save(model.state_dict(), args.out)
            flag = "  <- saved"
        print(f"epoch {epoch:2d}  train_mse {running / seen:.5f}  val_MAE {wp_mae:.4f}wp  "
              f"corr {corr:.3f}  {dt:4.0f}s{flag}", flush=True)

    print(f"\nbest val MAE {best:.4f} win-prob  saved to {args.out}")


if __name__ == "__main__":
    main()
