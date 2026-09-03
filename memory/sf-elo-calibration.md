---
name: sf-elo-calibration
description: The candidate agent's absolute Elo from the local Stockfish gauntlet (~1613 after the eval backlog)
metadata:
  type: project
---

**Current: ~1773 absolute Elo** (2026-09-03, third gauntlet, 336 games, dials sf1400..sf2200
plus `opponents/prev_prune`). CI [1717, 1841], 49.0% against the pool, 37-20-39 — above
sf1600, just under sf1800. **This is the most trustworthy run of the three**: the dials came
out monotonic for the first time (1521 < 1737 < 1802 < 1915 < 2025), residual RMS 119. Earlier
runs had inversions (sf1600 above sf1800; sf1800 above sf2000), so treat those as ±150.

Session progression, each measured with the predecessor in the same pool:

| gauntlet | baseline | candidate | within-pool gain |
|---|---|---|---|
| after quiescence/MVV-LVA | prev 1321 | 1407 | +86 |
| after eval backlog | prev_mvv 1508 | 1613 | **+105** |
| after pruning | prev_tt 1646 | 1661 | **+15** |
| after 7-change batch | prev_prune 1689 | **1773** | **+84** |

**Self-play transfer is unpredictable — measured 47%, 11%, and 140%.** The pruning batch
(+140 self-play → +15) was pure depth, which converts into wins against an opponent sharing
your blind spots but not against Stockfish. The seven-change batch (+60 → +84) was mostly eval
speed, which helps against everyone. A self-play number alone predicts almost nothing; only a
gauntlet with the predecessor seated settles it.

Note the absolute scale still drifts between runs for identical code: `prev_tt` read 1613 then
1646, `prev_prune` read 1661 then 1689. Never compare absolute numbers across gauntlets.
The pre-eval-backlog agent re-measured at **1508** in the SAME pool, so the three steps
(win-prob eval, numba forward pass, transposition table) are worth **+105 Elo measured**, not
the +224 the self-play deltas summed to. The candidate now wins 25 games off the dial pool
where the old agent won 8.

**Always put the previous agent in the gauntlet.** That same agent measured 1407 on
2026-09-01 and 1508 here — ~100 Elo of drift for identical code, from machine state and dial
noise. Only a within-pool delta is trustworthy; cross-run absolute comparisons are not.

**The dials are compressed, so 1613 is soft.** Fitted range 1503-2038 (535 wide) against a
nominal 1400-2200 (800 wide): sf1400 reads +103 high, sf2200 -162 low, residual RMS 123.
Ordering was at least monotonic this time (the 2026-09-01 run had sf1600 fitting above
sf1800). Real strength sits between sf1400 and sf1600. Treat the absolute figure as +-100 and
prefer the within-pool delta whenever a decision depends on it.

2026-09-03, after adding quiescence search (`training/elo.py`, 252 games, 6 openings,
3000ms+100ms, dials sf1400..sf2200 plus `opponents/prev`): candidate **~1407 absolute Elo**
(95% CI [1288, 1485], score 21.5%, 6-19-47) vs the pre-quiescence agent at **1321**
([1200, 1389], 13.9%, 0-20-52). Candidate won 6 games off the dial pool; prev won zero.

**Trust the head-to-head, not the absolute number.** Candidate vs prev directly (32 games) was
+150 Elo, 13-19-0, zero losses — a clean A/B on one machine. The gauntlet's absolute scale was
noisy at 72 games/player: residual RMS 147 Elo and sf1600 fitted 1832, ABOVE sf1800's 1750,
which is an ordering inversion that should not happen. Use `--rounds 2` or more openings for a
tighter anchor. The 2026-09-01 run (180 games) was better behaved at RMS 91.

**32 games cannot resolve anything under ~60 Elo** — learned the hard way. The second search
round (MVV-LVA + ply-adjusted mate) measured -10 Elo, 8-15-9, CI [-53, +39] against
`opponents/prev_qs`, i.e. nothing, despite `training/test_ordering.py` showing the SAME build
searching 55% fewer nodes (77% fewer in tactical positions). Node counts are the sharper
instrument for search changes; save games for things expected to clear ~60 Elo, and budget
100+ games otherwise. Changes were kept on the node evidence.

**Numba forward pass (2026-09-03) vs `opponents/prev_wp`: +170 Elo, 71-44-13 (72.7%),
CI [64, 113]** — the biggest win since quiescence. Identical node counts, 36.8% less time.
The lesson that generalises: at this strength **search speed is worth far more than eval
quality**. A 2.2x faster eval bought +170; a better-fitting eval bought +24. When choosing
what to work on, prefer anything that buys depth.

Transposition table (2026-09-03) vs `opponents/prev_numba`: **+30 Elo, 43-53-32 (54.3%),
CI [-10, 39]** — positive, not significant, kept on the node evidence (-8.6% cold).

**Treat self-play deltas as inflated.** Summing the three steps onto the old 1407 gives ~1630,
but Step 2's +170 came from a 2.2x speedup, and the standard heuristic (doubling the time
control ≈ +50-70 Elo) says that is worth ~+70-80 absolute. Agents sharing an eval and search
share blind spots, so beating your own predecessor overstates the gain against a diverse pool.
Predicted absolute before measuring: ~1500-1550.

Eval step (win-probability target, 2026-09-03) vs `opponents/prev_mvv`, 128 games each:
first attempt **+8 Elo, 27-77-24**; after fixing EVAL_SCALE, **+24 Elo, 27-83-18 (53.5%),
CI [-5, 32], p ~ 0.18 — positive but NOT significant**. Even 128 games only resolves to about
+-35 Elo, so quote significance rather than the point estimate. Kept on converging evidence
(pairwise ordering 80.6% vs 78.7%, neutral node cost), not on the game result alone.
See [[eval-net-pipeline]] for the EVAL_SCALE/TRAIN_SCALE split that mattered here.

Eval speed is machine- AND load-dependent: 39us/call on the 2026-09-01 box; the 2026-09-03 box
measured 85us and later 54us from identical code (the 85us run was taken while `uv sync` was
still settling, and the gauntlet above ran under that slower regime). So this box is ~1.4-2.2x
slower, not a clean 2x. Absolute Elo is NOT comparable across sessions — always re-measure the
baseline in the same pool, which is why `opponents/prev*/` snapshots exist. Benchmark only on an
idle machine. See [[eval-net-pipeline]], [[windows-testing]].

Gotcha: the SF calibration runs HANG ON EXIT on Windows — `python-chess` `SimpleEngine.quit()`
via the `atexit` handler in `opponents/sf_engine.py` gets stuck shutting down each stockfish.exe.
The gauntlet finishes and prints the full table first; only cleanup hangs. Because of this the
process never exits, so watch the output file for the final `Scale:` line rather than waiting on
a completion notification, then `Stop-Process` the leftover stockfish/python. Packaging is clean:
`harness/package.py` only globs root `*.py` + `weights/`, so `opponents/` and `training/` never
enter the zip (verified: submission = agent.py + features.py + weights/model.onnx, 768 KB).
