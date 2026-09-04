2# Working in this repo

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

## Project status (updated 2026-09-03)

**~1773 absolute Elo** — 336-game Stockfish gauntlet, CI [1717, 1841], 49.0% against the pool,
W-D-L 37-20-39. Above sf1600, just under sf1800. Started the session at 1508.

The agent is a learned NNUE-style eval inside iterative-deepening alpha-beta. It ships as
`agent.py` + `evalnet.py` + `features.py` + `weights/net.pt` (0.82 MB zip; build with
`uv run python -m harness.package`).

- **Eval**: 768->256->32->1 MLP emitting a logit, run by a numba forward pass straight from
  python-chess bitboards. Layer 1 is a sparse row gather, not a matmul: at most 32 of the 768
  inputs are ever set. 14.0 us/call.
- **Search**: quiescence (SEE + delta pruning), transposition table, MVV-LVA + killers +
  history ordering, null move, LMR, PVS, aspiration windows, check extensions, futility and
  reverse futility, ply-adjusted mate scores, position-aware time budgeting, and pondering on
  the opponent's clock.

### What changed on 2026-09-03

Each item was measured on its own against a snapshot in `opponents/`. Self-play Elo first,
then the gauntlet's within-pool number where one exists.

**Eval (gauntlet: 1508 -> 1613, +105 measured against +224 summed self-play)**
- Win-probability target: fit `sigmoid(cp / TRAIN_SCALE)` under MSE on probabilities instead
  of raw centipawns under Huber. **+24 self-play, not significant.**
- Numba forward pass replacing onnxruntime. **+170 self-play, CI [64, 113], significant** —
  the largest single gain of the session, from computing *identical* numbers faster.
- Transposition table, Zobrist-free key from the bitboard tuple. **+30, not significant.**
- Sampling fixed from a biased prefix to a whole-file reservoir.

**Search pruning (gauntlet: 1646 -> 1661, +15 measured against +140 summed self-play)**
- Fast capture generation in quiescence. **+36 self-play, marginal.**
- Null move pruning with a zugzwang guard. **+58, CI [3, 52], significant.**
- LMR + killers/history, measured together because LMR's value depends on ordering quality.
  **+46, marginal.**

**Eval speed and search refinement (gauntlet: 1689 -> 1773, +84 measured against +60 self-play)**
- Feature encoding folded into the jitted forward pass. Eval 41.3 -> 14.0 us/call, 2.89x.
- SEE pruning, check extensions, futility + reverse futility, PVS, aspiration windows,
  position-aware time budgeting. **+60 together, CI [12, 52], significant.**
- Pondering, in a background thread. **Elo unmeasurable locally** (see below); excluded.

**Cumulative:** ~2.15x faster to the same depth; eval 78.5 -> 14.0 us/call.

### Bugs found and fixed

- **Flag fall.** The time budget floored at 50ms without consulting the clock, and ~12ms of
  per-move work sits outside the deadline loop, so **any clock <=63ms overran and lost on
  time**. Now clamped to `time_left_ms - OVERHEAD_MS` with a `PANIC_MS` path that returns the
  best-ordered move unsearched. Caught by `training/test_rules.py`.
- **Delta pruning silently dead.** Its margin is in absolute centipawns, and the
  win-probability target shrank the eval ~3x, so the fixed margin stopped firing and the tree
  grew **43%**. Fixed by splitting `TRAIN_SCALE` (200) from `EVAL_SCALE` (600).
- **Every benchmark number was wrong.** `time.monotonic()` resolves to ~15.6ms on Windows and
  the fast benchmarks ran 0.016-0.06s — one to four ticks. All drivers and `agent.py` now use
  `perf_counter`; the agent matters most, since 15.6ms granularity against a 50ms budget is a
  real precision problem, not just a measurement one.
- **`training/model.py` had a stray character** making it a syntax error, so `train.py` and
  `export.py` could not import and the retrain pipeline had been dead in the repo.
- **Weights shipped as `.npz`**, which the contract does not list. Now `.pt`.

### Issues and traps — read before changing anything here

- **`EVAL_SCALE` must be re-measured after any retrain** with
  `training/test_calibration.py`. A stale value silently breaks delta pruning and nothing else
  will tell you.
- **Bitboards with bit 63 set exceed int64** and numba rejects them. `evalnet.to_signed()`
  reinterprets them as two's-complement negatives, and the de Bruijn bit scan masks `>> 58`
  with `0x3F` to undo the arithmetic shift's sign extension. Break either half and the eval
  goes wrong only on positions with pieces on Black's back rank.
- **`njit(cache=True)` must never be used**: numba writes its cache next to the source and the
  platform filesystem is read-only outside `/tmp`.
- **Mate scores are ply-relative** and must never enter the TT.
- **TT moves are membership-checked** against the legal list. A 64-bit hash collision returning
  an illegal move loses the game outright.
- **Null move needs its zugzwang guard.** The technique assumes having the move is worth
  something, which is exactly false in pawn endgames.
- **`chess.polyglot.zobrist_hash` costs 74us**, as much as a whole eval. The TT key is a hash
  of the public bitboard tuple, at 4us.
- **Pondering must be a background thread.** Inline would spend our own clock, since the
  referee charges us for all of `get_move`. It is stopped at the top of the next `get_move` by
  BOTH a past deadline and an explicit `_ponder_stop` flag — without the flag the thread can
  outlive the old deadline and adopt the new one the real search just installed.
- **A/B snapshots in `opponents/` share the ROOT `features.py` and `evalnet.py`** — imports
  resolve on `sys.path`, not next to the snapshot. Only `weights/` is per-snapshot. Editing
  `features.py` therefore changes every old snapshot's behaviour.

### Next steps, highest value first

Everything here buys depth. That is deliberate: every large gain this session came from
searching more in the same time, not from judging positions better.

1. **Incremental accumulator updates.** push/pop changes at most a couple of pieces, so the
   first-layer accumulator can be updated rather than rebuilt. The standard NNUE trick and the
   natural follow-on now that encoding is jitted.
2. **Numba bitboard move generator.** `list(board.legal_moves)` costs **109.5us**, more than a
   whole evaluation, and is the single biggest remaining cost — the wall that caps everything
   else. Potentially worth more than all other items combined. Also the most dangerous: this
   is where illegal-move bugs come from, and an illegal move loses the game outright. Days of
   work; verify relentlessly with `training/test_stress.py` and perft.
3. **Time management refinement.** The current heuristic scales by legal-move count and check
   status. Real engines also spend more when the best move is unstable between iterations.
4. **Eval quality, LAST.** Pairwise ordering is 80.6% and that is the ceiling on accuracy, but
   accuracy has repeatedly been worth less than speed here. If pursued: fit K to the data
   rather than assuming it (the Texel first step), and blend game outcomes into the target
   alongside engine evals, which needs a different dataset.

### Measurement discipline — earned the hard way

- **Self-play transfer is unpredictable.** Measured 47%, 11% and 140% across the three
  batches. Pure-depth changes convert into wins against an opponent that shares your blind
  spots and much less against Stockfish; eval-speed changes help against everyone. **A
  self-play number alone predicts almost nothing — seat the predecessor in the gauntlet.**
- **32 games resolves nothing under ~60 Elo**; 128 games reaches about +-35. Quote
  significance, not the point estimate.
- **For search changes, prefer node counts** (`training/test_ordering.py`). It resolved a 55%
  effect that 32 games reported as zero.
- **Validation loss is not strength.** A better val MAE has twice failed to predict the game
  result, and centipawn MAE actively misleads for a win-probability model.
- **Snapshot into `opponents/` BEFORE each change**, one per step.
- **Absolute Elo drifts ~30 between gauntlets for identical code**, and dials are compressed
  (fitted 1521-2025 against nominal 1400-2200). The 2026-09-03 third run was the first with
  monotonic dials (RMS 119); earlier runs had inversions and carry +-150. Never compare
  absolute numbers across runs.
- **Pondering cannot be measured locally at all.** `elo.py` runs both agents in one process, so
  a background search steals the opponent's core; on the platform each agent has its own
  container. `PONDER_ENABLED` exists to switch it off for local A/B runs. **It ships True.**

### Environment gotchas

- The `make play/arena/gate` harness does not run on Windows (`selectors` on pipes, WinError
  10038). Use the `training/` drivers — see `memory/agent-test-drivers.md`.
- `uv` is not on PATH here: `python -m pip install uv`, then drive everything as
  `python -m uv run ...`. `uv pip install` defaults to the SYSTEM python and fails with Access
  Denied; target the venv with `--python .venv\Scripts\python.exe`.
- Training-only deps `zstandard` and `onnx` are NOT in pyproject (which mirrors the platform),
  so `uv sync` removes them. Reinstall before retraining.
- **Close the browser before training.** A 3.5M-position run left ~4 GB free of 15.4 GB and
  epochs randomly spiked from 25s to 514s on paging. Compute was never the limit.
- Stockfish gauntlets **hang on exit** on Windows (`SimpleEngine.quit` via `atexit`). The table
  prints first; watch the output file for the final `Scale:` line, then kill the leftovers.
- `training/*.npz`, `training/model.onnx` and `opponents/prev*/` are gitignored — regenerable,
  and `data.npz` reached 54.6 MB, past GitHub's 50 MB warning.

More detail in `memory/`.
