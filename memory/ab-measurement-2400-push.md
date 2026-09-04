---
name: ab-measurement-2400-push
description: How to A/B nets reliably and the tooling built for the 2400-Elo push (Sep 2026)
metadata:
  type: project
---

Goal this session (set 2026-09-03): push the agent from ~1773 to **2400 absolute Elo**,
working independently. See [[sf-elo-calibration]] and [[eval-net-pipeline]].

**Reliable A/B across net/architecture changes** — the old `opponents/` snapshots shared the
ROOT `evalnet.py`/`features.py` and `evalnet` always loads ROOT `weights/net.pt`, so two
different nets could not be compared in one `elo.py` process (they'd use the same net).
Fixed with `training/snapshot.py <name>`: freezes agent.py+evalnet.py+features.py+weights into
`opponents/<name>/` as a **package** with relative imports, so a frozen net stays frozen.
`elo.py` now imports package snapshots via `importlib.import_module`. `base_1773` is the
frozen starting point for this push.

- `elo.py` now disables pondering for all in-process agents by default (`--ponder` to keep it
  on). Pondering in-process steals the shared core and flatters whoever ponders.
- Higher SF dials added: `opponents/sf2400`, `sf2600`, `sf2800` (UCI_Elo capped at 3190), to
  calibrate near the 2400 target. SF still hangs on exit on Windows — kill leftover
  python/stockfish after a gauntlet.
- `training/netquality.py` ranks a `.pt` net cheaply by **pairwise ordering** (the metric that
  tracks strength here) + win-prob MAE + logit/cp slope (to set EVAL_SCALE). Arch-agnostic.
- `train.py`/`export.py` now take/infer `--hidden1/--hidden2`; the numba forward pass already
  reads layer sizes from weight shapes, so a bigger first layer needs no inference change.

**Eval cost vs width** (this machine, warm numba): 256->32 = 6.8us, 512->32 = 13.7us,
768->32 = 20.9us, 1024->32 = 27.9us, 512->16 = 8.0us. Layer 2 (h1xh2 dense) dominates, and
eval runs at every quiescence leaf, so a bigger net really does cost search time — the
incremental dual-perspective accumulator is what would make a big first layer cheap.

New training data: `training/data.npz` re-downloaded at **10M positions** (was 2M), min-depth
12, clamp 2000, scatter over the whole dump. `zstandard` is installed but NOT in pyproject
(uv sync removes it).

**Progress:** 5x-data retrain of the current 256 arch lifted pairwise ordering 80.8%->83.5%
and won +66 self-play Elo vs `base_1773` (significant, CI [5,59]); shipped, EVAL_SCALE 600->530.
Frozen as `base_data10m`.

**Dual-perspective NNUE built** (`training/model2.py`, `train2.py`, `export2.py`;
`features.codes_to_dual_features`; `evalnet._forward_dual_bb` auto-selected when w2 has 2*H1
rows). Numba dual forward matches torch to 2.5e-08. Trains from existing mover-POV codes (the
opponent perspective is sq^56 + our/their swap).

**Incremental accumulator: decided NOT worth it for a 256-wide net.** In quiescence (high
branching, eval at every node) the per-move accumulator update (~8 row-adds x2 persp) roughly
cancels the from-scratch layer-1 saving (~64 row-adds), because you pay the update on every
move explored but only save on nodes that actually eval. It only pays for a much larger first
layer. So evals stay from-scratch; the depth lever is faster movegen, not incremental eval.

**The wall:** `list(board.legal_moves)` ~35us (measured on a 48-move middlegame; was quoted
109us earlier) dominates interior nodes and caps depth.

**MEASUREMENT BREAKTHROUGH — fast game A/B hides depth gains.** At base 2000ms+100ms the agent
only reaches depth ~6 (it reaches depth 8-9 at the platform's 4s/move), so any change that
trades accuracy for depth (bigger eval, LMP, more LMR) shows as break-even/loss because the
depth never materialises. Both the single-512 net (-8) and LMP (-14) LOST that fast A/B.
Use **`training/cploss.py`** instead: mean centipawn loss vs a local-Stockfish reference at a
realistic fixed per-move budget (~800ms). It is fast (~1.2 pos/s), low-variance, and REWARDS
depth. It has a `--set NAME=VAL` override to sweep agent constants without re-editing.
`training/gen_positions.py` makes the test FENs; the SF reference is cached to ref_d*.json.

Verdicts so far at 800ms/move, 256/10M net, 150 pos:
- **LMP is GOOD**: cp_loss 26.9 (off) -> 21.8 (on). Kept. (Constants LMP_MAX_DEPTH=3, LMP_BASE=3
  in agent.py.) The fast game A/B had said the opposite — trust cp_loss for depth changes.
- single-512 eval: cp/quality gain cancelled by 2x eval cost; reverted to 256.
- **LMP**: monotonically WORSE at 800ms (off 26.3, default 26.5, aggressive 29.1, extend-d4
  32.5). Reverted. Depth-buying pruning does not pay at measurable budgets.
- **Staged move generation: GOOD, shipped.** `agent._staged_moves` yields TT move, then
  captures/queen-promos (MVV-LVA), then killers, then quiets (history) -- generating quiets
  LAST and only when reached, so cutoff nodes skip the bulk of legal-move generation. cp_loss
  26.3 -> 24.7, SF-agree 54.4% -> 57.6%, and a confirming 3000ms game A/B won **+44 Elo**
  (56.2%). Correct over 16k+ positions incl checks/promotions/ep/castling; test_rules/mate/
  stress all pass. GOTCHA: castling is filtered by python-chess `to_mask` on the ROOK square
  (occupied), so `to_mask=~occupied` misses it -- added explicitly via generate_castling_moves.
  Frozen as `base_staged`. This is a FREE-depth win (same search, faster), which is the lever
  that pays at measurable budgets -- unlike accuracy/depth tradeoffs.
- **History malus** (penalise quiet moves tried before a cutoff): cp_loss 24.7 -> 27.9, WORSE.
  Reverted.

**Measurement gotchas learned:**
- cp_loss at 250 positions has ~+-3cp noise, so small search tweaks (LMP, malus, ordering) are
  NOT reliably distinguishable. Trust it only for effects >~4cp, or corroborate with a game
  A/B. netquality (pairwise ordering over 200k pairs) is far lower variance -> EVAL changes are
  reliably measurable; SEARCH changes are not, without games or 500+ positions.
- **Absolute-Elo gauntlet is impractical on Windows**: (a) at 3000ms TC a single game can crawl
  toward the 300-ply cap for 15+ min; (b) running cp_loss with a Stockfish DIAL as the agent
  hangs on process exit (sf_engine's atexit engine.quit hang), burning the full timeout per
  dial. Use relative self-play + candidate cp_loss + netquality; run only a lean, faster-TC
  game gauntlet if an absolute number is truly needed. Current estimate ~1830 absolute (1773 +
  ~half of +110 self-play from data+staged).

**Eval arch/speed curve** (this machine, warm): 256x32=6.8us, 512x16=8.0us, 384x32~10us,
512x32=13.7us. A bigger FIRST layer with a SMALLER second (512x16) costs barely more than
256x32 -- testing whether it keeps most of 512x32's quality (netquality 84.78% vs 256's 83.5%)
at ~1.2x cost instead of 2x. If so it is a near-free eval win.
