---
name: agent-test-drivers
description: The training/ scripts that verify the agent on Windows, and which question each answers
metadata:
  type: project
---

None of these ship (packager globs root `*.py` + `weights/` only). Run from the repo root with
`python -m uv run python training/<script>.py` — see [[windows-testing]] for why `uv` needs the
`python -m` prefix on this box.

- `test.py` — smoke test (6 legal opening moves) + eval throughput + a depth-4 timing. First
  thing to run after touching `agent.py`.
- `test_rules.py` — **the platform-contract check** (added 2026-09-03). Import budget vs the
  60s limit, single-threaded ONNX, legal UCI, reply size vs the 4 KB cap, and clock safety
  swept across remaining-time values down to 30ms. It caught a real flag-fall bug: the budget
  floored at 50ms without consulting the clock, so any clock <=63ms overran and lost on time.
- `test_mate.py` — checks mate-in-1 is found and that mate scores carry ply distance, so the
  engine prefers faster mates instead of shuffling in won positions.
- `test_ordering.py` — A/B two agent dirs on **node counts** at fixed depth, not games. Use
  this for any search change: it resolved a 55% node reduction that a 32-game match reported
  as nothing. See [[sf-elo-calibration]] for why games are the blunter instrument.
- `test_calibration.py` — buckets predicted vs true centipawns for two ONNX models and reports
  the calibration **slope** plus **pairwise ordering**. Run it after ANY retrain: the slope
  sets `features.EVAL_SCALE`, and a stale EVAL_SCALE silently breaks delta pruning (see
  [[eval-net-pipeline]]). Pairwise ordering is the metric closest to what search consumes.
- `test_sampling.py` — proves `download.py`'s reservoir is uniform across the file.
- `test_seek.py` — checks the dump is still multi-frame zstd so `--mode scatter` can seek.
- `compare_data.py` — puts two sampled datasets side by side; a biased sample looks healthy
  alone and only reveals itself against another.
- `estimate_dump.py` — sizes a full pass (compressed bytes, link rate, projected hours) before
  committing to one.
- `make_fake_data.py` — synthetic npz for smoke-testing train/export/evaluate without a
  download.
- `quickplay.py` / `elo.py` — one game / round-robin Elo. `elo.py` auto-calibrates to an
  absolute scale when `opponents/sfNNNN` dials are in the pool. NOTE `elo.py` caps at 16
  openings (`OPENING_LINES`), so 128 games = `--openings 16 --rounds 4`.

A/B snapshots live in `opponents/`: `prev` (pre-quiescence), `prev_qs` (quiescence only,
pre-MVV-LVA). Snapshot the agent there BEFORE each change — absolute Elo is not comparable
across machines or sessions, so the baseline has to be re-measured in the same pool.
