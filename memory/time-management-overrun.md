---
name: time-management-overrun
description: Instability-aware search overrun in _search_root — the safe design, and why the first (banking) version was reverted
metadata:
  type: project
---

2026-09-04: added instability-aware time overrun to `_search_root` (agent.py). The search
normally stops at the soft budget from `_budget`. After each completed iteration at depth >=
OVERRUN_MIN_DEPTH (5), if the best move CHANGED vs the prior iteration, the deadline is extended
to `soft * OVERRUN_FACTOR` (1.6, clamped to time_left - OVERHEAD_MS); if it did not change (or
was extended before and has now settled), the deadline snaps back to soft. So a still-moving
choice buys another ply, a settled one does not. Constants: OVERRUN_FACTOR=1.6,
OVERRUN_MIN_DEPTH=5.

**KEY property: this only ever ADDS search, never removes it** (never stops below the soft
budget), so it cannot make a move worse — only spend extra clock where the choice is unstable.
Flag-safe: hard budget is clamped to time_left - OVERHEAD, and the stress test showed 6s of
clock margin at platform TC.

**First version (REVERTED) did banking and it backfired.** It ALSO stopped stable positions
EARLY (SOFT_STOP_STABLE=0.5 of soft) to bank clock for the unstable ones. cploss @800ms went
26.4 -> 30.7 (mean up) while SF-agreement ROSE 56->58.7% and median stayed ~8 — the signature
of a few TAIL BLUNDERS: positions that looked settled at depth 5-6 but flip deeper got cut off.
Lesson: early-stopping on apparent stability is dangerous; the asymmetric overrun-only version
removes that risk.

**Measurement caveat (important):** cploss CANNOT judge time management. It gives each position
an independent fixed budget, so the "bank on easy, spend on hard" benefit has no carryover to
show — only the per-move cost is visible. cploss also can't resolve the overrun's benefit
(within its ±3-4cp noise at 150 pos). The ONLY fair test is a full-game A/B with a shared clock
(training/elo.py). Being strictly non-harmful, it is safe to ship even if the game A/B reads
neutral. See [[ab-measurement-2400-push]] for the cploss-can't-see-search-changes rule.
