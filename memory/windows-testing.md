---
name: windows-testing
description: How to test the agent on this Windows machine (the harness does not run here)
metadata:
  type: project
---

The `make play/arena/gate` harness does NOT run on Windows: `harness/sandbox.py` registers
OS pipes with `selectors.DefaultSelector()`, which on Windows only accepts sockets, so it
fails with `OSError [WinError 10038] ... not a socket`. Do not edit `harness/` to fix this.

Test with the Windows-safe driver instead (imports agents in-process, no subprocess):

    uv run python training/quickplay.py --white . --black baselines/numba --base-ms 8000
    uv run python training/test.py        # smoke test + eval/node-rate benchmark
    uv run python training/elo.py         # round-robin vs baselines -> Elo table (anchor random=0)

Static gate parts still work: `uv run ruff check .` and `uv run mypy`.

Also: a `.venv` created under WSL has a `lib64 -> lib` symlink that PowerShell's `uv sync`
can't remove (`Access is denied`). Recreate a native venv with:
`rm -rf .venv && UV_LINK_MODE=copy uv sync --python 3.12`. See [[eval-net-pipeline]].

Environment bootstrap (2026-09-03): this box had no `uv` on PATH and no `.venv`. `uv` is not
installed standalone here — get it with `python -m pip install uv`, then drive everything as
`python -m uv run ...` (a bare `uv` still resolves to nothing). `uv sync` pulls torch CPU and
takes several minutes. Stockfish for calibration is gitignored, so a fresh clone also needs
`python -m uv run python opponents/fetch_stockfish.py` before any SF gauntlet.

The `training/` drivers buffer stdout: only lines with `flush=True` (elo.py's per-20-game
progress) appear promptly, so an empty output file does not mean the run is stuck.

`uv pip install` defaults to the SYSTEM python here (the WindowsApps store build) and fails
with Access Denied. Target the venv explicitly:
`python -m uv pip install --python .venv\Scripts\python.exe zstandard onnx`.

Close the browser before training. With 15.4 GB total and Brave/Discord/VS Code running, a
3.5M-position run left ~4 GB free and epochs randomly spiked from ~25s to 514s on paging.
Compute was never the limit; memory pressure was.

**Never time anything here with `time.monotonic()`** — on Windows it resolves to ~15.6ms, so a
benchmark running 0.016s measures a single tick and reports nonsense (it made the eval look
like 31us when it is 78us). Use `time.perf_counter()`. `agent.py` uses it too: 15.6ms
granularity against a 50ms move budget is a real precision problem, not just a measurement one.

A/B snapshots in `opponents/` share the ROOT `features.py` and `evalnet.py` — `import features`
resolves on sys.path, not next to the snapshot's own agent.py. Only `weights/` is per-snapshot
(it is found via `__file__`). So editing `features.py` silently changes every old snapshot's
behaviour. Fine when the change should apply to both sides; a confound otherwise.
