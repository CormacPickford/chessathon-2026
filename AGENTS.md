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

## Project status (updated 2026-09-03)

**~1661 absolute Elo** (336-game SF gauntlet), up from 1508 for the pre-eval-backlog agent.
Measured within-pool gains: eval backlog +105, pruning batch +15 (nothing), and the final
seven-change batch is +60 self-play but not yet gauntlet-measured.

**The headline lesson: speed beats accuracy, and self-play lies.** A 2.2x faster eval bought
+170 self-play; a genuinely better eval bought +24. But summed self-play deltas (+224, +140)
transferred at ~47% and ~11% respectively -- agents sharing an eval share blind spots, so
beating your own predecessor overstates real strength. Always seat the previous agent in the
gauntlet, and halve any self-play number before believing it.

Absolute scale is only +-150: dials are compressed and occasionally inverted (sf1800 fitted
1916 above sf2000's 1880), and identical code drifts ~30 Elo between runs. Real strength sits
between sf1400 and sf1600.

Current agent: NNUE-style learned eval (768->256->32->1 MLP via onnxruntime) inside
iterative-deepening alpha-beta with a quiescence search at the leaves. Ships as `agent.py` +
`features.py` + `weights/model.onnx` (~0.8 MB zip; build with `uv run python -m harness.package`).

Done 2026-09-03 -- quiescence search (`quiesce()` in agent.py):
- Leaves no longer evaluate mid-exchange. Not in check: stand-pat floor, then captures and queen
  promotions ordered MVV-LVA, with delta pruning (200cp margin). In check: no stand-pat, all
  evasions searched, no-evasions = mate. `QS_MAX_DEPTH = 8` as insurance. Stalemate is
  deliberately not detected at qs leaves -- a legal-move generation at every quiet leaf costs
  more than the rare misjudged leaf.
- Head-to-head vs the pre-change agent (32 games, 3000ms+100ms): **+150 Elo, 13-19-0, zero
  losses**. This is the trustworthy number -- direct A/B, same hardware, no calibration layer.
- SF gauntlet (252 games, 6 openings, sf1400..sf2200): candidate **~1407** [1288, 1485] vs
  prev **1321** [1200, 1389]. Candidate won 6 games off the dial pool; prev won 0. Caveat: only
  72 games/player, so dials came out noisy -- residual RMS 147 Elo and sf1600 fitted (1832)
  ABOVE sf1800 (1750), an ordering inversion. Treat the absolute scale as +/-150 here; rerun
  with `--rounds 2` or more openings for a tighter anchor.
- `opponents/prev/` holds the pre-quiescence agent + its weights as a permanent A/B opponent.
  Not shipped (packager globs root `*.py` + `weights/` only).

Also 2026-09-03 -- MVV-LVA in the main search, ply-adjusted mate scores, clock-safety fix:
- `_order()` now sorts noisy moves first by MVV-LVA via a tier tuple. The tier is load-bearing:
  a raw MVV-LVA score sorts a king capture (cheap victim, most expensive attacker) BELOW the
  quiet moves.
- Mate scores are `-MATE + ply`, so a mate in one outranks a mate in six. Verified by
  `training/test_mate.py` (both mate-in-1s played; score pins to MATE-1 at every depth).
- **Clock-safety bug fixed (was a real contract violation).** The budget floored at 50ms
  without consulting the clock, and ~12ms of per-move work sits outside the deadline loop, so
  ANY clock <=63ms overran and flagged = automatic loss. Budget is now also clamped to
  `time_left_ms - OVERHEAD_MS`, with a `PANIC_MS` path returning the best-ordered move
  unsearched. Worst case went from +33ms over the clock to 18ms of margin
  (`training/test_rules.py`).
- Search efficiency A/B (`training/test_ordering.py`, fixed depth 4): MVV-LVA cuts nodes
  **-55%** and time **-56%** overall, and **-75%/-77%** in a tactical position -- never worse
  on any position tested.
- BUT the head-to-head vs `opponents/prev_qs` (32 games, same setup as the quiescence test)
  came back **-10 Elo, 8-15-9, CI [-53, +39]** -- a dead heat. The prediction going in was
  +20..+60; it was not confirmed. 32 games has a ~+/-65 Elo noise floor, so this rules out a
  large gain but cannot resolve a small one. Changes were KEPT on the node-count evidence,
  which is a far less noisy instrument than 32 games.
- Read-across: search efficiency is no longer the binding constraint on strength. The eval
  (val MAE ~237cp) likely is. Weight the eval backlog items above further search tuning, and
  do not trust 32-game matches to resolve anything under ~+60 Elo.

Done 2026-08-31:
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

Step 1 of the eval backlog -- win-probability target + unbiased sampling (2026-09-03):
- **Sampling was the real bug.** `download.py` kept the first N usable lines, a prefix. The
  file's opening lines are genuinely unrepresentative (score mean 94.5 vs 41.5 for the
  population, material/score corr 0.653 vs 0.520). Replaced with a reservoir over the whole
  dump, uniformity proven in `training/test_sampling.py`.
- Full sequential passes are infeasible: this link runs at 0.8 MB/s, so 21.7 GB is ~8 hours,
  and JSON parsing is NOT the bottleneck (skipping 98% of decodes bought only 2.3x). The dump
  is multi-frame zstd on a range-supporting server (`training/test_seek.py`), so `--mode
  scatter` pulls stratified slices and samples the whole file in ~14 min. Retries and a loud
  coverage warning were added after a DNS outage silently cost 39 of 64 strata mid-run.
- Caveat worth remembering: whole-file coverage vs 39% coverage changed the population by
  <1.5% on every metric. The dump is homogeneous along its length, so the sampling fix, while
  correct, is NOT where Elo comes from. Only the tiny opening prefix is skewed.
- Trained on 3.5M positions: val 0.1250 win-prob MAE, corr 0.780.
- **The "compressed magnitudes" framing was largely a red herring.** Minimax is invariant to
  any monotone rescaling of the eval, so magnitude cannot affect move choice. The first
  attempt (EVAL_SCALE 200) measured **+8 Elo, 27-77-24** -- flat.
- What magnitude DOES affect is delta pruning, whose margin is additive and written in
  absolute centipawns. A 3x-shrunken eval (fitted slope 0.335) left the fixed 200cp margin
  relatively huge, pruning stopped firing, and the tree grew **+43% nodes**. Splitting
  `TRAIN_SCALE` (200, the target) from `EVAL_SCALE` (600 = 200/0.335, the inverse transform)
  restored slope to 1.006 and cut the node cost to +6%.
- Final: **+24 Elo, 27-83-18 (53.5%), CI [-5, 32] over 128 games** -- p ~ 0.18, NOT
  significant. Kept because three independent measurements agree in direction: pairwise
  ordering 80.6% vs 78.7%, neutral nodes, positive score. Do not claim it as a proven win.
- New drivers: `test_calibration.py` (bucketed predicted-vs-true cp, calibration slope,
  pairwise ordering), `test_sampling.py`, `test_seek.py`, `compare_data.py`, `estimate_dump.py`.

Step 2 of the eval backlog -- numba forward pass (2026-09-03): **+170 Elo, 71-44-13 (72.7%),
CI [64, 113] over 128 games.** The biggest win since quiescence, and it changed nothing about
what the engine thinks -- only how fast.
- `evalnet.py` (new, ships at root) holds a jitted forward pass; `agent.py` no longer imports
  onnxruntime at all.
- **Layer 1 is a sparse row gather, not a matmul.** At most 32 of the 768 inputs are non-zero
  (one per piece), so `W1 @ x` is the sum of ~32 rows of W1: ~8k adds instead of 197k
  multiply-adds, and the 768-float input vector is never built. That, not the loss of
  onnxruntime's per-call overhead, is where most of the speed comes from.
- `cache=True` is deliberately NOT set on the njit: numba writes its cache next to the source
  and the platform filesystem is read-only outside /tmp. Compilation lands at import in ~1.2s,
  far inside the 60s budget.
- Verified before trusting the games: numerically identical to ONNX (max 2.9e-6, i.e. 0.002cp)
  and **identical node counts on every test position (0.0%)**, with 36.8% less time. Same
  tree, same decisions, less clock. End-to-end eval 70.0 -> 31.5 us/call.
- `weights/` now ships only `net.npz`; the ONNX export moved to `training/model.onnx`, where
  the tooling still uses it to cross-check and to compare against older snapshots.
- Read-across: **speed is worth far more than eval quality at this strength.** A 2.2x faster
  eval bought +170; a better-fitting eval bought +24. Prioritise depth over accuracy.

Step 3 of the eval backlog -- transposition table (2026-09-03): **+30 Elo, 43-53-32 (54.3%),
CI [-10, 39] over 128 games** -- positive, not significant. Kept: -8.6% nodes cold, and the
table compounds in real play where it survives across iterative-deepening iterations and moves.
- Key is `hash()` of the public bitboard tuple (pieces, colour masks, turn, castling, ep).
  **Not `chess.polyglot.zobrist_hash`, which costs 74us -- as much as a whole evaluation** and
  would have spent more than the table saves. The bitboard tuple is 4us.
- Depth-preferred probe (only trust a score searched at least as deep), EXACT/LOWER/UPPER
  bounds, and the stored move tried first for ordering.
- **The TT move is checked for legality, never trusted.** Keys are 64-bit hashes; a collision
  returning a move illegal in this position would lose the game outright.
- **Mate scores are never stored** -- they are ply-relative, so reusing one at a different
  distance from the root would be wrong.
- `TT_MAX = 400_000` with clear-on-full. Measured ~255 bytes/entry, so the cap projects to
  ~101 MB against the 2 GB limit; real growth is ~600 entries/move, so the cap rarely trips.

**Timing bug found and fixed (affects any old benchmark number).** Every `training/` driver
used `time.monotonic()`, which on Windows has ~15.6ms resolution, and the fast benchmarks ran
0.016-0.06s total -- one to four ticks. All drivers and `agent.py` now use `perf_counter`;
`agent.py` matters most because 15.6ms granularity against a 50ms budget is a real precision
problem in the deadline check. Corrected costs per call: full evaluate 78.5us, of which
`board_to_codes` is **58us** and the network only 22.5us; `list(board.legal_moves)` is 109.5us.
Elo results are unaffected -- those come from games, not timers.

Step 4 -- search pruning (2026-09-03), each measured separately over 128 games at
3000ms+100ms:
- **Fast capture generation: +36 Elo, 49-43-36, CI [-4, 47].** `_noisy_moves` was generating
  every legal move and discarding the quiet ones at every quiescence node. `generate_legal
  _captures()` is 2.8-4.4x faster (100us -> 23us in a tactical position). It omits QUIET queen
  promotions, so those are added back behind a bitboard test that is false in almost every
  position; `training/test_noisy.py` proves the move sets are identical over 4,000 played
  positions plus hand-built promotion cases. Node counts confirmed 0.0% change, -10.8% time.
- **Null move pruning: +58 Elo, 54-41-33 (58.2%), CI [3, 52] -- SIGNIFICANT.** Give the
  opponent a free move at reduced depth and a null window; if the score still clears beta, the
  line is far too good for them and gets pruned. -12.2% nodes, -35.6% in the endgame. Guards
  matter more than the technique: skipped in check (passing is illegal), when the side to move
  has only pawns and a king (zugzwang breaks the "having the move is worth something"
  assumption, which is exactly what null move relies on), and directly after another null.

- **LMR + killers/history: +46 Elo, 53-39-36 (56.6%), CI [-1, 53].** Measured together on
  purpose: LMR reduces moves BY THEIR POSITION in the move list, so its value is entangled with
  ordering quality, and quiet moves previously had no ordering at all. Killers remember the two
  quiet refutations per ply; history scores (from, to) pairs across the search weighted by
  depth^2. LMR reduces quiet, non-check moves from index 3 onward at depth >= 3, re-searching
  at full depth if the reduced search beats alpha.

Cumulative speed since the start of the session, at fixed depth: numba -36.8%, TT -8.3%,
capture gen -10.8%, null move -10.0% -> **~2.15x faster to the same depth**.

**Absolute gauntlet after the pruning work (336 games): candidate 1661 [1593, 1714] vs
`prev_tt` 1646 [1565, 1707] -- +15, i.e. NOTHING.** The pruning batch summed to +140 in
self-play and transferred ~11%. The eval batch, by contrast, transferred ~47% (+224 -> +105).
The likely reason: capture-gen, null move and LMR all buy depth, and extra depth converts
directly into wins against an opponent that shares your eval and therefore your blind spots.
Stockfish dials fail differently, so marginal depth buys much less. Dials were poorly
calibrated again (RMS 150, sf1800 fitted 1916 ABOVE sf2000's 1880), and `prev_tt` -- identical
code -- read 1613 last gauntlet and 1646 here, so +-150 and ~30 Elo of cross-run drift.
**Weight self-play deltas at roughly half, and less than that for pure-depth changes.**

Step 5 -- eval speed, search refinement, pondering (2026-09-03). Seven changes measured
together (pondering excluded, see below): **+60 Elo, 44-62-22 (58.6%), CI [12, 52] --
SIGNIFICANT** vs `opponents/prev_prune`. At depth 5: -21% nodes, -57% time.
- **Encoding folded into the jitted forward pass.** `board_to_codes` cost 30us against the
  net's 11us, because `piece_map()` allocates a dict and a Piece per piece. `forward_board`
  takes python-chess's bitboards directly: eval 41.3 -> 14.0 us/call, **2.89x**. Bitboards
  with bit 63 set exceed int64, which numba rejects, so `evalnet.to_signed` reinterprets them
  as two's-complement negatives and the de Bruijn bit scan masks with 0x3F to undo the shift's
  sign extension. Verified identical to the codes path over 3,000 positions (1.9e-6).
- **SEE** prunes captures that lose material once the square is fought over, so quiescence
  stops searching queen-takes-defended-pawn. Approximate by design (ignores pins), which is
  why it only ever prunes clearly losing captures, and promotions are exempt.
- **Check extensions**, **futility + reverse futility**, **PVS**, **aspiration windows**,
  and **position-aware time management** (fewer legal moves or in check -> spend less; a
  crowded position -> spend more).
- **Pondering** searches the expected reply on the opponent's clock in a background thread.
  It MUST be a thread: doing it inline spends our own clock, since the referee charges us for
  the whole of get_move. Stopped first thing in the next get_move, via BOTH a past deadline
  (to unwind negamax) and an explicit `_ponder_stop` flag -- without the flag the thread can
  sail past the old deadline and pick up the new one the real search just installed. Measured
  join cost 1.0-3.5ms.
- **Pondering's Elo is unmeasurable locally and was excluded from the +60.** `elo.py` runs
  both agents in ONE process, so our background search would steal the core from the
  opponent's thinking; on the platform each agent has its own container. `PONDER_ENABLED`
  exists to switch it off for local A/B runs. **It ships True.**

**Rule compliance fixes from re-reading the contract** (do not trust the summary in this file
for "what is allowed" -- fetch the URLs):
- Weights ship as `weights/net.pt`. The contract names ".onnx, .safetensors and .pt"; `.npz`
  is NOT listed, and a validator that whitelists extensions would reject the whole submission.
- `torch.set_num_threads(1)` in evalnet.py, since torch now loads the weights at import.
- Contract confirms pondering explicitly: "The process keeps its dedicated core after
  `get_move` returns, and pondering is allowed."
- Finalists must explain how the agent was built, and disqualification can be retroactive --
  which is why the comments explain WHY, not what.

`training/test_stress.py` plays full games asserting every move is legal, parses, fits 4 KB,
and returns inside the clock: 189 and 99 moves, zero violations, both with and without
pondering.

Next: the feature encoding is now the most expensive part of an evaluation. `board_to_codes`
allocates a dict and a Piece object per piece via `piece_map()`, then loops in Python.
Prototype in `training/test_encode_speed.py` folds encoding into the jitted forward pass by
passing bitboards straight to numba. Given Step 2 showed speed is worth far more than eval
accuracy here, that is the highest-value remaining item.

Backlog (highest value first), all pointing the same way -- **buy depth, not accuracy**:
- Fold `board_to_codes` into the jitted forward pass (bitboards -> numba, no `piece_map()`,
  no intermediate array). 58us of the 78.5us eval. Prototype: `training/test_encode_speed.py`.
- Incremental feature updates: push/pop changes at most a few pieces, so the first-layer
  accumulator can be updated rather than rebuilt -- the standard NNUE trick, and the natural
  follow-on once encoding is jitted.
- Killer-move and history heuristics for quiet-move ordering; the TT only orders nodes it has
  already seen.
- A larger or better-trained net LAST. The eval's ordering quality (80.6% pairwise) is the
  ceiling on accuracy, but accuracy has repeatedly been worth less than speed here.

Measurement discipline (earned the hard way):
- 32 games resolves nothing under ~60 Elo; 128 games still only reaches ~+-35. Budget games
  accordingly, and quote significance, not just the point estimate.
- For search changes use `training/test_ordering.py` node counts instead -- it resolved a 55%
  effect that 32 games reported as zero.
- Validation loss is not strength. A better val MAE has twice now failed to predict the game
  result, and cp-space MAE actively misleads for a win-probability model.
- Snapshot into `opponents/` BEFORE each change (`prev`, `prev_qs`, `prev_mvv`, ...), one per
  step, so attribution stays clean and baselines are re-measurable on the same machine.

Gotchas: the harness fails on Windows (`selectors` on pipes, WinError 10038) -- use the
`training/` drivers. Training-only deps `zstandard` + `onnx` were `uv pip install`ed and are NOT
in pyproject (which mirrors the platform), so `uv sync` removes them -- reinstall to retrain.
Eval speed is machine-dependent AND load-dependent: 39us/call on the 2026-08-31 box, but 85us
then 54us on the 2026-09-03 box from identical code (the 85us reading was taken while `uv sync`
was still settling). Call it ~1.4-2.2x slower here. Benchmark on an idle machine, compare agents
only within one pool, and never carry an absolute Elo across sessions. More detail in `memory/`.
