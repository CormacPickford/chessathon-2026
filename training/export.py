"""Export the trained MLP to weights/model.onnx and verify onnxruntime matches torch.

    uv run python training/export.py

The model ships at weights/model.onnx, which the packager includes automatically. The output
is a logit from the mover's POV: `sigmoid(output)` is the mover's win probability, and
agent.py multiplies by features.EVAL_SCALE to read it back as centipawns.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import NUM_FEATURES
from training.model import EvalNet


def export_numba(model: EvalNet, out: Path) -> None:
    """Dump raw weights for the numba forward pass in agent.py.

    Saved as `.pt`, which the agent contract names explicitly alongside .onnx and
    .safetensors. An .npz would be equally valid data, but the contract does not list it and
    a validator that whitelists extensions would reject the whole submission -- not a risk
    worth taking for a file format.

    Layout matters. torch stores Linear weights as (out_features, in_features); we transpose
    to (in, out) so that the first layer's per-feature row `w1[idx]` is contiguous. The first
    layer is never a matmul at inference: the input is a 768-wide binary vector with ~32 ones,
    so the layer is a sum of ~32 rows, which is why the transpose is the useful orientation.
    """
    sd = model.state_dict()
    arrays = {
        "w1": sd["net.0.weight"].numpy().T.copy(),  # (768, 256)
        "b1": sd["net.0.bias"].numpy().copy(),
        "w2": sd["net.2.weight"].numpy().T.copy(),  # (256, 32)
        "b2": sd["net.2.bias"].numpy().copy(),
        "w3": sd["net.4.weight"].numpy().T.copy(),  # (32, 1)
        "b3": sd["net.4.bias"].numpy().copy(),
    }
    tensors = {
        k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
        for k, v in arrays.items()
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, out)
    shapes = "  ".join(f"{k}{tuple(v.shape)}" for k, v in tensors.items())
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KB)\n  {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eval MLP to ONNX.")
    parser.add_argument("--model", type=Path, default=Path("training/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("training/model.onnx"))
    parser.add_argument("--numba-out", type=Path, default=Path("weights/net.pt"))
    args = parser.parse_args()

    model = EvalNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    export_numba(model, args.numba_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, NUM_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["features"],
        output_names=["value"],
        dynamic_axes={"features": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

    # Verify: onnxruntime output must match torch on random binary feature vectors.
    rng = np.random.default_rng(0)
    x = (rng.random((256, NUM_FEATURES)) < 0.05).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["value"], {"features": x})[0]
    max_diff = float(np.abs(torch_out - ort_out.reshape(-1)).max())
    print(f"max |torch - onnxruntime| = {max_diff:.2e}  ({'OK' if max_diff < 1e-4 else 'BAD'})")


if __name__ == "__main__":
    main()
