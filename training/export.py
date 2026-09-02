"""Export the trained MLP to weights/model.onnx and verify onnxruntime matches torch.

    uv run python training/export.py

The model ships at weights/model.onnx, which the packager includes automatically. The
output is position value in pawns from the mover's POV; agent.py multiplies by 100.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eval MLP to ONNX.")
    parser.add_argument("--model", type=Path, default=Path("training/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("weights/model.onnx"))
    args = parser.parse_args()

    model = EvalNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

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
