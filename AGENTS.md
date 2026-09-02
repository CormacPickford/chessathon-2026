# Working in this repo

This is a starter for AI Chessathon, a chess-engine competition. The deliverable is one file,
`agent.py`, exposing `get_move(fen, time_left_ms) -> str`. It gets zipped and uploaded, and the
platform plays it against other people's agents on a fixed cadence.

## Read the rules from the source

The competition rules and the agent contract live on the site and change. Fetch them before you
answer anything about limits, deadlines, or what is allowed:

- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md

The quick reference below is a convenience for common questions. The two URLs are canonical
and they change, so fetch them before you rely on a number.

## The contract, in one place

- `agent.py` at the root of the zip, not inside a folder. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str` returning UCI, `e2e4` or `e7e8q`.
- Your colour is the side to move in the fen. There is no other input.
- The process starts once per game and stays alive between your moves. Module state survives to
  your next move in the same game, never to the next game.
- Import time has a 60 second budget before the clock starts. Load weights there.
- 120 s + 0.5 s per move, per side, on wall time. One core, 2 GB, no network, no GPU.
- Illegal move, malformed output, crash, out of memory, or flag fall loses that game. A move
  reply over 4 KB counts as illegal. 300 plies without a result goes to material adjudication.
- Everything in the zip together stays under 50 MB unzipped.
- Six uploads per team per day, and the latest one that passed validation is the one that plays.
- Rated games start from curated opening positions, not the standard start. The set is not
  published.
- The process keeps its core while the opponent thinks, so pondering on their time is allowed.
  Two of your games can run at once, in separate containers.

## Things that break agents here

- The filesystem is read-only apart from 256 MB at `/tmp`. `HOME` and every cache path already
  point there; do not write anywhere else.
- No network at all. Nothing downloads at runtime. Weights ship inside the zip.
- One core. `torch.set_num_threads(1)`. More threads lose time rather than winning it.
- Your zip is first on `sys.path`. Never name a file after a module you import: `chess.py`,
  `types.py`, `random.py` will shadow the real one and the failure will look unrelated.
- The environment is fixed. The platform preinstalls torch 2.13 (CPU), numpy 2.5, python-chess
  1.11, onnxruntime 1.29 and numba 0.67 and installs nothing else. A `requirements.txt` is
  ignored, so an import outside that stack crashes on the platform even when it works locally.
  Additions can be requested at hello@aichessathon.com.
- Native binaries in the zip are rejected. Ship Python source. Model weights like `.onnx`,
  `.safetensors` and `.pt` are fine, and any model shipped must be one the team trained.
- numba is how Python gets fast here. Warm every jitted function once at import so compilation
  lands in the init budget, not on the clock. Cython does not work on the platform.
- `print` is safe. The runner points file descriptor 1 at stderr before importing the agent, so
  nothing you write can corrupt the protocol. It is discarded in rated games and shown in the
  validation log.

## Do not

- Do not use Stockfish, Lc0, Maia, or any existing engine inside the submission, including a
  pip package that embeds one. It is an instant disqualification and it is checked after the
  fact. Training on data an engine annotated is allowed; the ban covers what ships and runs
  inside the zip.
- Do not add network calls, subprocess calls to external binaries, or anything that reads outside
  the agent directory and `/tmp`.
- Do not obfuscate. What ships has to be source a judge can read.
- Do not edit `harness/`. It mirrors the platform's protocol and clock. Changing it makes local
  results meaningless.

## Verify

```
make play      # one game against a baseline, real time control
make arena     # 20 fast games against a baseline, with a score
make zip       # build submission.zip with agent.py at the root
make gate      # ruff, mypy, and two games that have to finish cleanly
```

Nothing here decides whether an upload is accepted. The platform validates on upload and writes a
log to the dashboard; that log is the authority. The harness exists so local games are honest.

## Style

Python 3.12, type-annotated, ruff and mypy strict clean. Keep `agent.py` readable: it is the
thing a judge reads if your games get flagged, and the thing you have to explain at the final.

## Project status (updated 2026-08-31)

Current agent: NNUE-style learned eval (768->256->32->1 MLP via onnxruntime) inside
iterative-deepening alpha-beta. Ships as `agent.py` + `features.py` + `weights/model.onnx`
(~0.8 MB zip; build with `uv run python -m harness.package`).

Done this session:
- Trained the eval net on 2M Lichess+Stockfish positions: val MAE ~237cp, corr 0.82. Pipeline
  in `training/`: download.py -> data.npz, train.py -> model.pt, export.py -> weights/model.onnx,
  evaluate_model.py (sanity + color-symmetry invariant). Encoding is side-to-move-relative, so
  the net output feeds negamax directly.
- Beats every baseline. Relative Elo (anchor random=0, 3000ms+100ms): candidate 785, numba 663,
  minimax 656, greedy 283, random 0.
- Windows-safe testing (the harness can't run here): `training/quickplay.py` (one game),
  `training/test.py` (smoke + node-rate bench), `training/elo.py` (round-robin Elo with bootstrap
  CIs; auto-calibrates to an ABSOLUTE scale when Stockfish dials are in the pool).
- Local Stockfish calibration opponents in `opponents/` (NEVER ship): `fetch_stockfish.py`
  (SF18 in opponents/engines/), `sf_engine.py`, `make_dials.py`, and sf1400..sf2200 wrappers.
  Verified SF plays (opens ~0.3s, moves in ~0.1s once Defender has scanned the exe).

Absolute Elo (done 2026-09-01): the SF gauntlet (180 games, 3000ms+100ms, sf1400..sf2200) puts
the candidate at ~1373 absolute Elo (95% CI [1214, 1477]), just below sf1400 (score 12.5%,
2-11-47); dials self-consistent to ~91 Elo RMS. Confirmed the packager bundles only root `*.py` +
`weights/` (submission = agent.py + features.py + weights/model.onnx, 768 KB); `opponents/` and
`training/` stay out. Note: SF gauntlet runs HANG ON EXIT on Windows (SimpleEngine.quit via
atexit) after printing the table -- kill leftover python/stockfish once you have the results.

Next step: work the backlog to push past 1400.

Backlog (highest value first): eval magnitudes are compressed -> retrain on a win-probability
target; add quiescence search; replace the ~39us/call ONNX eval with a numba forward pass for
more search depth.

Gotchas: the harness fails on Windows (`selectors` on pipes, WinError 10038) -- use the
`training/` drivers. Training-only deps `zstandard` + `onnx` were `uv pip install`ed and are NOT
in pyproject (which mirrors the platform), so `uv sync` removes them -- reinstall to retrain.
More detail in `memory/`.
