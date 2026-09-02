---
name: sf-elo-calibration
description: The candidate agent's absolute Elo from the local Stockfish gauntlet (~1373)
metadata:
  type: project
---

Ran the Stockfish gauntlet on 2026-09-01 (`training/elo.py`, 180 games, 3000ms+100ms, dials
sf1400..sf2200). The NNUE-eval candidate lands at **~1373 absolute Elo** (95% CI [1214, 1477]),
just below sf1400 — score 12.5%, W-D-L 2-11-47. Dials fit their nominal ratings within
±60–120 (residual RMS 91 Elo), so the scale is trustworthy to ~100 Elo.

This is far below the relative-ladder number (785 vs weak baselines) because that pool had no
strong opponents. Absolute strength is roughly a 1400 club player. The backlog items are what
move it: win-probability eval target (fixes compressed magnitudes), quiescence search, and a
numba forward pass to replace the ~39us ONNX eval for more depth. See [[eval-net-pipeline]].

Gotcha: the SF calibration runs HANG ON EXIT on Windows — `python-chess` `SimpleEngine.quit()`
via the `atexit` handler in `opponents/sf_engine.py` gets stuck shutting down each stockfish.exe.
The gauntlet finishes and prints the full table first; only cleanup hangs. Kill the leftover
`python.exe`/`stockfish.exe` after reading the results. Packaging is clean: `harness/package.py`
only globs root `*.py` + `weights/`, so `opponents/` and `training/` never enter the zip
(verified: submission = agent.py + features.py + weights/model.onnx, 768 KB).
