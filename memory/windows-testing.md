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
