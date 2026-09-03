---
name: agent-architecture
description: What is in agent.py/evalnet.py and the non-obvious reasons behind each piece
metadata:
  type: project
---

The submission is `agent.py` + `evalnet.py` + `features.py` + `weights/net.pt` (~0.84 MB).
Finalists must explain how the agent was built and disqualification can be retroactive, so the
comments in those files explain WHY rather than what. This note records the traps.

**Eval.** A 768->256->32->1 MLP emitting a LOGIT, run by a numba forward pass over raw
bitboards (`evalnet.forward_board`). Layer 1 is a sparse row gather, not a matmul: at most 32
of 768 inputs are non-zero. See [[eval-net-pipeline]] for the TRAIN_SCALE/EVAL_SCALE split,
which is the single easiest thing to get wrong on a retrain.

- Bitboards with bit 63 set exceed int64 and numba refuses them. `to_signed()` reinterprets
  them as two's-complement negatives; the de Bruijn bit scan then masks `>> 58` with `0x3F`
  to undo the arithmetic shift's sign extension. Change either half and the eval breaks
  silently on positions with pieces on Black's back rank.
- `njit(cache=True)` is NOT used anywhere: numba writes its cache next to the source and the
  platform filesystem is read-only outside /tmp. Compilation lands at import instead.
- Weights are `.pt`, not `.npz`. The contract names ".onnx, .safetensors and .pt"; .npz is
  not listed and a whitelist validator would reject the submission.

**Search.** Iterative deepening alpha-beta with: quiescence (captures + quiet queen promotions,
all evasions when in check), TT with depth-preferred probe, MVV-LVA + killers + history
ordering, null move, LMR, PVS, aspiration windows, check extensions, futility and reverse
futility, delta and SEE pruning in quiescence.

Traps that cost real Elo if broken:
- **Delta pruning's margin is in absolute centipawns**, so it silently stops working if the
  eval's scale changes. This cost 43% more nodes until EVAL_SCALE was recalibrated.
- **Mate scores are ply-relative** (`-MATE + ply`) and must never be stored in the TT.
- **TT moves are membership-checked** against the legal list — a 64-bit hash collision
  returning an illegal move loses the game outright.
- **Null move needs its zugzwang guard** (`_has_non_pawn_material`): the technique assumes
  having the move is worth something, which is exactly false in pawn endgames.
- `chess.polyglot.zobrist_hash` costs 74us, as much as a whole eval. The TT key is a hash of
  the public bitboard tuple instead, at 4us.

**Pondering** runs the expected reply on the opponent's clock in a daemon thread. It must be a
thread: inline would spend our own clock, since the referee charges us for all of get_move.
Stopped at the top of the next get_move by BOTH a past deadline (unwinds negamax) and
`_ponder_stop` (stops the deepening loop) — without the flag the thread can outlive the old
deadline and adopt the new one. `PONDER_ENABLED` switches it off for local A/B runs, because
`elo.py` runs both agents in one process where a background search steals the opponent's core.
**It ships True.** Its Elo cannot be measured locally at all.

Verify with `training/test_stress.py` (full games, asserts legality/clock/size on every move),
`test_rules.py`, `test_mate.py`, `test_noisy.py`. See [[agent-test-drivers]].
