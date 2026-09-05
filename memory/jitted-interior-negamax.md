---
name: jitted-interior-negamax
description: Jitted interior negamax delegated for shallow subtrees (qsearch.py) — the big depth lever; how it's built, verified, and tuned
metadata:
  type: project
---

2026-09-05: the interior search now runs jitted for the shallowest subtrees, the biggest depth
lever after jitted quiescence+movegen (see [[jitted-qsearch]]). agent.negamax hands any subtree
at `depth <= JIT_LEAF_DEPTH` to a fully jitted negamax in qsearch.py that runs the whole subtree
(make, generation, legality, quiescence) with no python-chess object.

**Safety property (why this was approachable):** the ROOT move is still chosen in Python
(_search_root) over python-chess's verified legal list, so a jitted interior search — like
jitted quiescence — can only affect SCORES, never emit an illegal move. Bugs cost strength, not
games. This is why the whole interior port is far less dangerous than the old "movegen is where
illegal-move bugs come from" fear.

**Built in layers, each verified before the next:**
1. Castling-aware make (`_make_full`, tracks castling-rights bitboard mirroring python-chess)
   + full generator (`_gen_all` = captures + quiets + legal castling). Verified by **perft
   matching python-chess exactly** across the standard suite (kiwipete, positions 3-6 to 3.9M
   nodes, castling-rights-rook-capture edge cases) — training/test_perft.py.
2. Plain alpha-beta (`_negamax_plain`): verified **score-identical** to a python-chess negamax
   with the same ordering + same jitted leaf eval, 406 positions, depths 2-4 —
   training/test_negamax.py. This proves the recursion/make/legality/mate handling.
3. Pruned negamax (`_negamax_pruned`): adds RFP, null move, futility, PVS, LMR, check extensions
   — mirroring agent.negamax. NOT score-parity-testable (pruning diverges by design); validated
   by game A/B + stress/perft. **No TT, no killers/history** (moves come in _gen_all's
   MVV-LVA-then-piece order). Pruning constants are HARDCODED in qsearch (_RFP_MARGIN etc.) and
   must be kept in sync with agent.py by hand.

**Speed (depth-8, 3 middlegames):** plain JIT=2 = 1.49s; pruned JIT=3 = **0.36s (4.1x)**;
JIT=4/5 plateau ~0.35s. Pruning kills the un-pruned node explosion that made plain JIT=3 slow.
JIT_LEAF_DEPTH=3 is the sweet spot: depth<=3 is already the great majority of nodes, so higher
barely adds speed while removing the TT from more of the tree. (Plain depth-7: JIT=0 2.27s ->
JIT=2 0.79s = 2.9x.)

**A/B results (self-play, 96 games @ 6000ms+100ms, seat predecessor each time):**
- Plain JIT=2 vs base_speed: **+35 Elo, CI [12, 60]**, 59.9%. Committed (f63c007).
- Pruned JIT=3 vs base_jitleaf(plain JIT=2): **+58 Elo, CI [31, 87]**, 66.1% (W-D-L 44-39-13).
  Committed. `can_null` is threaded through the delegation so a jitted child of a Python null
  move does not forfeit a second move.

**Session cumulative (self-play, each vs predecessor):** base_v2net -> base_speed +64 (jitted
quiescence+movegen) -> base_jitleaf +35 (plain interior JIT=2) -> base_jitpruned +58 (pruned
interior JIT=3). Depth keeps converting straight to Elo.

**Notes/traps:** import time 2.2s -> ~17.8s (all the jitted compiles; still fine vs 60s).
`is_quiet` in the pruned loop = tier==0 (from the packed sort key) AND promo==0, so captures,
queen-promos and under-promos are excluded from futility/LMR, matching agent. gives_check is
computed after the functional make (no cheap pre-make check in bitboards). LMR/futility use a
`moves_done` legal-move counter, not the raw loop index (which includes skipped illegals).
Snapshots: base_jitleaf (plain JIT=2), base_jitpruned (pruned JIT=3).

**Next lever from here:** a jitted TT (+ killers/history) inside the jitted negamax would let
JIT_LEAF_DEPTH go higher without the ordering/cutoff quality loss, and is the last big depth
lever. See [[eval-is-bottleneck]] for why depth keeps beating eval quality here.
