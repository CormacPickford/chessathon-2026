---
name: eval-is-bottleneck
description: With staged movegen shipped, eval (not movegen) is the #1 search cost — correcting the earlier note
metadata:
  type: project
---

cProfile of a depth-7 negamax (3 middlegame FENs, 2.95s total, 512x16 net) on 2026-09-04:

- `evalnet.forward_board` (the net): **0.668s / ~23%** — the single biggest `tottime`.
  With `evaluate` wrapper + `to_signed` it is ~0.78s / **26%** of search time.
- `board.push`: 0.27s (+`_remove_piece_at` 0.10)
- `generate_pseudo_legal_moves` 0.17s, `attackers_mask` 0.16s, `generate_legal_moves` cumtime 0.53s
- Most evals happen in quiescence (77.5k forward_board vs 52.8k quiesce calls).

**This corrects [[ab-measurement-2400-push]]'s "the wall is `list(board.legal_moves)`".** That
predated staged move generation, which now skips full quiet-move gen at cutoff nodes — so movegen
dropped and **eval is now the dominant per-node cost**. Consequences:

- A retrain at the SAME architecture (e.g. 512x16) is a **free quality win** — no depth cost.
- Bigger/slower nets cost depth directly (eval is the bottleneck), so they must clear a real bar.
- **Incremental accumulator updates would now pay** here (they didn't for the pre-staged 256
  net), IF paired with a wide first layer + tiny second layer + the dual-perspective net
  (single-perspective can't be incremental — every move flips side-to-move → full re-encode).

**DECISIVE (2026-09-04): eval SPEED beats eval QUALITY at platform-ish budget.** cploss @
2500ms/move, 150 pos, ref d14: the 256 net (7.2us eval, 83.5% pairwise) scored **cp_loss 18.9**
vs the 512x16 net (8.6us, 84.57% pairwise) at **26.3** — a 7.4cp gap FAVOURING the faster net,
despite 512x16's higher pairwise ordering. netquality (pairwise) mis-predicted strength here:
the slower eval loses depth and depth dominates. **512x16 was reverted; the 256 net ships.**
Consequence for the 2400 push: chase a FASTER eval at equal/slightly-lower quality (smaller
net, or a faster forward pass), NOT a wider/better one. Incremental accumulator is de-prioritised
too — it only enables WIDE nets, and wide nets just lost. Snapshots: `base_staged` = shipped 256;
`base_512x16` = the reverted slower net (keep for reference).

**WIN SHIPPED (2026-09-04): 3x faster forward pass, identical output.** The numba layer-2 loop
was j-outer/k-inner, so the inner loop strode down a w2 *column* (stride n2) — cache-hostile and
un-vectorisable. Rewrote it k-outer so the inner loop walks the contiguous `w2[k, :]` row, and
skip rows where the post-ReLU `h1[k]==0` (NNUE sparsity). Result: **256x32 eval 7.2us -> 2.29us
microbench (3.1x), ~3.0us in-agent (2.4x)**, verified vs torch to 1.9e-6 over 300 positions;
ruff/mypy clean; stress/mate/rules pass. Applied to `_forward` and `_forward_bb` in evalnet.py.
Shipped as snapshot `base_optfwd`. Layer-1 was already contiguous (per-feature rows), so it was
left alone; only layer-2 had the strided access.

Caveats: cploss @800ms moved only 23.9 -> 23.6 (within noise) because eval was ~26% of node
time, so a 2.4x eval speedup is only ~15% total throughput — real but small, and cploss can't
resolve it per-move. It compounds over a game (depth), zero risk (identical output), so kept.
**Eval is no longer the bottleneck; movegen + board.push now dominate (~70%).** Further eval
speed has diminishing returns; the next depth lever is faster movegen (risky/long) — see
[[ab-measurement-2400-push]]. Note this also means bigger nets are STILL slower (the 256->512
gap is layer-1, which the optimization did not touch — both nets share the same layer-2 size).

Same k-outer/sparsity rewrite applied to `_forward_dual_bb` (its layer 2 also branched
per-element on white_to_move): dual-256 eval **14.06us -> 3.10us (4.5x)**, verified vs torch
2.9e-6. Dual now costs only +0.8us over single-256 (2.31us), so the dual quality bet
(84.71% pairwise vs 83.80%) became cheap enough to test properly.

2026-09-04 continued: retrained 256x32 for 35 epochs -> `training/model_256x32_v2.pt`,
pairwise 83.50% -> **83.80%**, EVAL_SCALE 532, same speed; shipped as `base_v2net` on the
netquality rule (same-speed eval changes are decided by pairwise, cploss ±3cp can't see 0.3%).
Its cploss @800ms read 25.8 vs the old net's 23.6 — within noise, noted as ambiguous.
Also deduped the double `evaluate()` in negamax (RFP + futility both called it): static_eval
computed lazily at most once per node. Provably identical search; ruff/mypy/stress clean.

Also: search ordering/pruning is already well-tuned. Counter-move heuristic + a log-LMR table
both measured neutral-to-worse (cploss 26.4 vs ~24.8 baseline; node counts flat). Reverted.
Depth-buying changes keep losing at the 800ms local budget — see [[ab-measurement-2400-push]].

**DUAL-256 NET BUILT, TESTED, REVERTED (2026-09-04).** Exported the dual-perspective 256 net
(pairwise 84.71% vs v2 single-256's 83.80%, EVAL_SCALE 516) and shipped it briefly. 128-game
self-play vs base_v2net came back **50.4%, CI [-20,+22] — dead even**; cploss flat within noise.
And it is ~1us slower per eval (3.4us vs 2.3us), which is real depth cost at a 4s budget.
By the session's own "eval SPEED beats eval QUALITY" rule, **reverted to the faster v2 single
net** (model_256x32_v2.pt, export.py --numba-out weights/net.pt, EVAL_SCALE 532). The dual net's
only remaining rationale was incremental accumulators, already decided not-worth-it for 256
width. Snapshot `base_dual256` keeps it for reference. GOTCHA in export.py: `--out` is the ONNX
path and `--numba-out` is weights/net.pt — passing `--out weights/net.pt` overwrites the good
numba .pt with an ONNX file and torch.load then fails (weights_only can't unpickle it).
