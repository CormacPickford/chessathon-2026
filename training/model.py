2"""The evaluation network, shared by training and ONNX export.

A small NNUE-style MLP: 768 binary features -> 256 -> 32 -> 1. The single output is the
position value in *pawns* from the side-to-move's point of view (the training targets are
centipawns / 100). Multiply by 100 at inference to get centipawns.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import NUM_FEATURES

HIDDEN1 = 256
HIDDEN2 = 32


class EvalNet(nn.Module):
    def __init__(self, hidden1: int = HIDDEN1, hidden2: int = HIDDEN2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
