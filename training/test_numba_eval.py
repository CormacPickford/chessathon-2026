"""Verify the numba forward pass matches the ONNX model, then time both.

    uv run python training/test_numba_eval.py

A faster eval that computes something slightly different is worse than no change at all: the
games would move and the cause would be invisible. So check agreement on real positions first
and only then look at the speed. Not part of the submission.
"""

import sys
import time
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evalnet
from features import board_to_codes, codes_to_features

N = 3000

# Real positions from random legal play, so the comparison covers the input distribution the
# agent actually meets rather than random bit patterns.
rng = np.random.default_rng(0)
boards = []
board = chess.Board()
while len(boards) < N:
    if board.is_game_over() or board.ply() > 120:
        board = chess.Board()
    moves = list(board.legal_moves)
    board.push(moves[rng.integers(len(moves))])
    boards.append(board.copy())

codes = np.stack([board_to_codes(b) for b in boards])

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(
    str(Path(__file__).resolve().parent.parent / "training" / "model.onnx"),
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
name = sess.get_inputs()[0].name

onnx_out = np.asarray(sess.run(None, {name: codes_to_features(codes)})[0]).reshape(-1)
numba_out = np.array([evalnet.forward(c) for c in codes], dtype=np.float64)

diff = np.abs(onnx_out - numba_out)
print(f"{N:,} positions")
print(f"max |onnx - numba| = {diff.max():.3e}")
print(f"mean|onnx - numba| = {diff.mean():.3e}")
# Both paths are float32; agreement to ~1e-4 is reordered-arithmetic noise, not a bug.
print(f"agreement: {'OK' if diff.max() < 1e-3 else 'MISMATCH -- do not ship'}")

# Centipawn-level agreement is what actually reaches the search.
from features import EVAL_SCALE  # noqa: E402

cp_diff = np.abs(onnx_out - numba_out) * EVAL_SCALE
print(f"max centipawn difference: {cp_diff.max():.3f}cp\n")

single = codes[0]
t0 = time.perf_counter()
for _ in range(N):
    evalnet.forward(single)
numba_s = time.perf_counter() - t0

feats = codes_to_features(single).reshape(1, -1)
t0 = time.perf_counter()
for _ in range(N):
    sess.run(None, {name: feats})
onnx_s = time.perf_counter() - t0

print(f"onnx : {onnx_s / N * 1e6:7.1f} us/call  ({N / onnx_s:9,.0f}/s)")
print(f"numba: {numba_s / N * 1e6:7.1f} us/call  ({N / numba_s:9,.0f}/s)")
print(f"speedup: {onnx_s / numba_s:.2f}x")
