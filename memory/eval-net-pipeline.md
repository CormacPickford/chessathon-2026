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

Current net (2026-09-03): 3.5M **scatter-sampled** positions against a **win-probability**
target, `sigmoid(cp / TRAIN_SCALE)` with TRAIN_SCALE=200, fit with MSE on probabilities. Val
0.1250 wp MAE, corr 0.780. The output is a LOGIT, not pawns. (Superseded the first run: 2M
prefix-sampled positions, raw-cp Huber, val MAE ~237cp.)

**The two scales differ on purpose.** `TRAIN_SCALE` (200) defines the target; `EVAL_SCALE`
(600) converts the logit back to centipawns at inference. Fitting probabilities makes the
logit a ~3x shrunken estimate of cp (fitted slope 0.335), because probability space barely
separates +800 from +2000. That shrinkage cannot affect move choice — minimax is invariant to
monotone rescaling — but it silently broke **delta pruning**, whose margin is additive and
written in absolute centipawns: the tree grew 43%. EVAL_SCALE = 200/0.335 puts the output back
on a true-cp footing (slope 1.006, node cost +6%). **If you retrain, re-measure the slope with
`training/test_calibration.py` and update EVAL_SCALE to match** — otherwise pruning silently
degrades and nothing else will tell you.

Sampling: `download.py --mode scatter` pulls stratified range-request slices across the whole
21.7 GB dump (multi-frame zstd) in ~14 min; a full sequential pass is ~8h at 0.8 MB/s. The old
"first N lines" prefix was genuinely biased, but only at the very start of the file — 39% and
100% coverage agree within 1.5%, so do not expect Elo from coverage alone.

Known weak spots next: ONNX eval is the per-node bottleneck (~55us/call here) — a numba
forward pass buys depth for free; no transposition table; the eval's pairwise ordering (80.6%)
is the real ceiling, not its scale. Quiescence, MVV-LVA and ply-adjusted mates are DONE.
See [[sf-elo-calibration]] for results and [[agent-test-drivers]] for how to verify.
