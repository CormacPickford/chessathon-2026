---
name: jitted-qsearch
description: qsearch.py — quiescence AND interior movegen jitted in numba; score/set-identical; 1.59x search speedup; how it was validated
metadata:
  type: project
---

2026-09-04: the hot search paths moved into `qsearch.py` (@njit), in two waves.

**Wave 1 — whole quiescence subtree jitted:** capture/evasion generation, functional make
(no unmake — scalars passed down the recursion), legality via make+own-king-attacked, SEE,
delta pruning, and the eval call, all compiled. agent.quiesce is a one-line delegation.

**Wave 2 — interior-node movegen jitted:** `qsearch.legal_noisy` / `legal_quiets` (pseudo-gen
+ king-safety filter + MVV-LVA sort) back agent._noisy_moves and the quiet phase of
_staged_moves. The jitted generators hand back packed ints (from | to<<6 | promo<<12);
`agent._cached_move` memoises int -> chess.Move (constructing a Move cost more than the gen).
Castling is NOT jitted — python-chess supplies it via generate_castling_moves, only when
someone still has the right (rare). En passant + queen push-promo come from the noisy set;
under-promos + pushes from quiets; the test proves noisy|quiets == all-legal-minus-castling.

**Result: depth-7 on 3 middlegames 2.79s (base_v2net, pure Python search) -> 1.75s = 1.59x**,
same net, output-identical. Isolated quiescence trees are ~11x (196us -> 18us). This is the
session's core win: pure depth, zero risk (identical output), pays most at the platform's 4s
budget where depth matters (cf. staged movegen's +44 Elo @3000ms).

**Why safe to jit these:** quiescence returns only a SCORE (no illegal-move risk). The interior
LEGAL generators DO feed board.push, so they are game-critical — hence the exhaustive set
parity test. numba doesn't bounds-check: move buffers are 256 wide (max ~218 + promo slack).

**Validation (training/test_qsearch.py), 7629 random-game + 14 trap positions:**
- noisy movegen set == python-chess: 7319/7319; evasions == legal_moves: 310/310.
- legal_noisy | legal_quiets == all legal minus castling: 7629/7629 (the game-critical one).
- quiescence score parity vs a Python reference using the jitted deterministic order key
  ((tier,mvv-lva,packed) desc, ties impossible): 1500/1500 bit-identical.
- Trap FENs: en-passant pins, promotion fans, castling-exclusion, in-check evasions.
Traps hit while building: EVAL_CAP is 20_000 not 10_000 (parity didn't catch — random pos
never saturate; check constants vs agent.py by eye). python-chess SEE quirk replicated
deliberately: attack masks vs ORIGINAL occupancy (no x-ray reveal).

**Consequences:** deadline NOT checked inside the jitted quiescence tree (bounded by
QS_MAX_DEPTH + capture exhaustion; negamax checks before entry). Import time 2.2s -> 6s
(compilation, fine vs 60s). snapshot.py now freezes qsearch.py with import-rewriting; snapshots
before base_speed have none. Packaged zip = agent+evalnet+features+qsearch+weights, 898 KB
unzipped, imports clean in isolation. `np.empty(256)` per interior node is allocated fresh (a
shared buffer is a recursion-clobber risk not worth ~3%).

**What's left on speed:** board.push/pop (python-chess) and the staged-move plumbing now
dominate interior nodes. A fully jitted interior negamax (with TT) is the next big lever but
much riskier. See [[eval-is-bottleneck]], [[ab-measurement-2400-push]], [[time-management-overrun]].
