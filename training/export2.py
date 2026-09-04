"""Export a dual-perspective AccEvalNet to the numba weight layout and verify the forward pass.

    uv run python training/export2.py --model training/model2_256.pt --out weights/net.pt

Layout mirrors export.py: torch stores Linear weights (out, in); the numba pass wants
per-feature rows contiguous, so every matrix is transposed. w2 becomes (2*H1, H2), whose row
count is how evalnet.py detects the dual architecture. Verifies numba == torch on random
positions before writing anything the agent will ship.
"""

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalnet
from features import board_to_codes, codes_to_dual_features
from training.model2 import AccEvalNet


def to_arrays(model: AccEvalNet) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    arrays = {
        "w1": sd["l1.weight"].numpy().T.copy(),   # (768, H1)
        "b1": sd["l1.bias"].numpy().copy(),
        "w2": sd["l2.weight"].numpy().T.copy(),   # (2*H1, H2)
        "b2": sd["l2.bias"].numpy().copy(),
        "w3": sd["l3.weight"].numpy().T.copy(),   # (H2, 1)
        "b3": sd["l3.bias"].numpy().copy(),
    }
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for k, v in arrays.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Export dual-perspective net for numba inference.")
    p.add_argument("--model", type=Path, default=Path("training/model2_256.pt"))
    p.add_argument("--out", type=Path, default=Path("weights/net.pt"))
    p.add_argument("--random", action="store_true", help="verify with a random net, write nothing")
    args = p.parse_args()

    if args.random:
        model = AccEvalNet()
    else:
        sd = torch.load(args.model, map_location="cpu")
        h1 = sd["l1.bias"].shape[0]
        h2 = sd["l2.bias"].shape[0]
        model = AccEvalNet(h1=h1, h2=h2)
        model.load_state_dict(sd)
    model.eval()
    tensors = to_arrays(model)

    # Correctness: numba dual forward must match torch on real positions of both colours.
    w = {k: v.numpy() for k, v in tensors.items()}
    rng = np.random.default_rng(0)
    board = chess.Board()
    max_diff = 0.0
    for _ in range(400):
        if board.is_game_over():
            board = chess.Board()
        us, them = codes_to_dual_features(board_to_codes(board))
        with torch.no_grad():
            ref = model(torch.from_numpy(us).unsqueeze(0),
                        torch.from_numpy(them).unsqueeze(0)).item()
        turn = board.turn
        got = evalnet._forward_dual_bb(
            evalnet.to_signed(board.pawns), evalnet.to_signed(board.knights),
            evalnet.to_signed(board.bishops), evalnet.to_signed(board.rooks),
            evalnet.to_signed(board.queens), evalnet.to_signed(board.kings),
            evalnet.to_signed(board.occupied_co[chess.WHITE]),
            evalnet.to_signed(board.occupied_co[chess.BLACK]), turn == chess.WHITE,
            w["w1"], w["b1"], w["w2"], w["b2"], w["w3"], w["b3"],
        )
        max_diff = max(max_diff, abs(ref - got))
        moves = list(board.legal_moves)
        board.push(moves[rng.integers(len(moves))])

    print(f"max |torch - numba dual| = {max_diff:.2e}  ({'OK' if max_diff < 1e-3 else 'BAD'})")
    if args.random:
        print("random-net check only; nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, args.out)
    shapes = "  ".join(f"{k}{tuple(v.shape)}" for k, v in tensors.items())
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)\n  {shapes}")


if __name__ == "__main__":
    main()
