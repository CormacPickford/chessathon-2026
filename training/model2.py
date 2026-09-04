"""Dual-perspective accumulator eval net (proper NNUE structure).

A single shared first layer W1: 768 -> H1 builds a per-perspective accumulator. The position
is scored from the concatenation of the side-to-move's accumulator and the opponent's:

    acc_us    = relu(W1 @ features_from_mover_perspective  + b1)      (H1,)
    acc_them  = relu(W1 @ features_from_opponent_perspective + b1)    (H1,)
    h         = [acc_us, acc_them]                                    (2*H1,)
    out       = W3 @ relu(W2 @ h + b2) + b3                           logit, mover POV

Why this shape and not the current single-perspective MLP: the first-layer accumulator depends
only on where the pieces are for one perspective, so across a make/unmake it changes by the
handful of pieces that moved. That makes it INCREMENTALLY UPDATABLE in search -- the standard
NNUE trick -- so a wide, expressive first layer costs almost nothing per node. Two perspectives
also give the net the tempo/initiative asymmetry a single flipped view blurs.

Trained from the same (mover-POV codes, cp) data as the single-perspective net: the opponent
perspective is recovered from the mover-POV codes by flipping the square (sq ^ 56) and swapping
the our/their half (see features_dual). ReLU, not clipped ReLU, to match the numba inference.
"""

import torch
from torch import nn

NUM_FEATURES = 768
H1 = 256
H2 = 32


class AccEvalNet(nn.Module):
    def __init__(self, h1: int = H1, h2: int = H2) -> None:
        super().__init__()
        self.l1 = nn.Linear(NUM_FEATURES, h1)  # shared across both perspectives
        self.l2 = nn.Linear(2 * h1, h2)
        self.l3 = nn.Linear(h2, 1)

    def forward(self, feats_us: torch.Tensor, feats_them: torch.Tensor) -> torch.Tensor:
        acc_us = torch.relu(self.l1(feats_us))
        acc_them = torch.relu(self.l1(feats_them))
        h = torch.cat([acc_us, acc_them], dim=1)
        h = torch.relu(self.l2(h))
        return self.l3(h).squeeze(-1)
