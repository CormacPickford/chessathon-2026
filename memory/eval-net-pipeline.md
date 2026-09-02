---
name: eval-net-pipeline
description: The learned-eval pipeline in training/ and how agent.py uses it
metadata:
  type: project
---

`agent.py` scores positions with a small NNUE-style MLP (768->256->32->1) run via
onnxruntime, wrapped in iterative-deepening alpha-beta search. The net predicts
Stockfish-annotated centipawns from the **side-to-move's** point of view (so negamax leaves
use it directly, no sign flip). Training on engine-annotated data is allowed; shipping an
engine is not — this ships only our own trained weights.

Pipeline (all under `training/`, none of which ships — only root `*.py` + `weights/` are zipped):
- `features.py` (root, ships): board -> 64 int8 codes -> 768 features, oriented to the mover
  (board flipped + colours swapped when Black moves). A position and its mirror encode
  identically — the symmetry invariant checked in `evaluate_model.py`.
- `download.py`: streams `lichess_db_eval.jsonl.zst` (~22GB, never saved), samples N positions
  to `training/data.npz`. Convention: dataset evals are White-POV; we store mover-POV.
- `train.py` -> `model.pt`; `export.py` -> `weights/model.onnx`; `evaluate_model.py` = sanity.
- Training-only deps `zstandard`, `onnx` are installed via `uv pip install` (NOT in pyproject,
  which must mirror the platform stack). `uv sync` will remove them; reinstall to retrain.

First run: 2M positions, val MAE ~237cp, corr 0.82; beats the numba baseline. Known weak spots
to improve next: eval magnitudes are compressed (train on win-prob target), no quiescence
search, and ONNX eval is ~39us/call (~25k nodes/s) — a numba forward pass would be much faster.
See [[windows-testing]] for how to play games here.
