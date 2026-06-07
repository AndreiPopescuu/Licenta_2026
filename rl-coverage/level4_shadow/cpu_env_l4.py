"""Gym env for the 5615-bin Level 6 benchmark.

Action space: MultiDiscrete([30, 32, 32, 32, 5])  -- (op, rd, rs1, rs2, imm_bucket).
Observation packs coverage + per-register writer/age + last-2 instructions for
K=3 chain planning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_cpu_l4 import (
    Op, N_BINS, WRITERS, READERS, IMM_BUCKETS, RAW_MAX_DIST,
    L6History, bins_for_step, advance_history,
)

N_OPS = len(Op)                           # 30
N_WRITERS = len(WRITERS)                  # 27
WRITER_TO_IDX = {w: i for i, w in enumerate(WRITERS)}
OP_TO_IDX = {op: i for i, op in enumerate(Op)}

OBS_DIM = (
    N_BINS                               # coverage vector
    + 32 * (N_WRITERS + 1)              # per-reg writer one-hot (+none)
    + 32 * (RAW_MAX_DIST + 1)           # per-reg age one-hot (1..MAX, other)
    + 2 * (N_OPS + 32)                  # last 2 (op one-hot, rd one-hot)
)


class IbexL4Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 4096, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.MultiBinary(OBS_DIM)
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.hist = L6History()
        self.step_idx = 0

    def _obs(self) -> np.ndarray:
        w_flat = np.zeros(32 * (N_WRITERS + 1), dtype=np.int8)
        a_flat = np.zeros(32 * (RAW_MAX_DIST + 1), dtype=np.int8)
        for r in range(32):
            w = self.hist.writer[r]; a = self.hist.age[r]
            if w is not None and 1 <= a <= RAW_MAX_DIST:
                w_flat[r * (N_WRITERS + 1) + WRITER_TO_IDX[w]] = 1
                a_flat[r * (RAW_MAX_DIST + 1) + (a - 1)] = 1
            else:
                w_flat[r * (N_WRITERS + 1) + N_WRITERS] = 1  # "none"
                a_flat[r * (RAW_MAX_DIST + 1) + RAW_MAX_DIST] = 1  # "other"
        last_flat = np.zeros(2 * (N_OPS + 32), dtype=np.int8)
        for slot, entry in enumerate(self.hist.last_two):
            op_e, rd, _, _ = entry
            last_flat[slot * (N_OPS + 32) + OP_TO_IDX[op_e]] = 1
            last_flat[slot * (N_OPS + 32) + N_OPS + rd] = 1
        return np.concatenate([self.covered, w_flat, a_flat, last_flat])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.hist.reset()
        self.step_idx = 0
        return self._obs(), {}

    def step(self, action):
        op = int(action[0]); rd = int(action[1])
        rs1 = int(action[2]); rs2 = int(action[3])
        imm_b = int(action[4])
        new_hits = 0
        for b in bins_for_step(op, rd, rs1, rs2, imm_b, self.hist):
            if not self.covered[b]:
                self.covered[b] = 1
                new_hits += 1
        advance_history(op, rd, rs1, rs2, self.hist)
        self.step_idx += 1
        terminated = bool(self.covered.sum() == N_BINS)
        truncated = self.step_idx >= self.episode_steps
        return self._obs(), float(new_hits), terminated, truncated, {
            "covered": int(self.covered.sum()),
        }
