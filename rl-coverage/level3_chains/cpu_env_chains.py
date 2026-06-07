"""Gym env for the 1739-bin chain benchmark.

Observation must expose enough history for the policy to *plan* K=3 chains:
- per-register writer + age (for RAW_DIST bins, same as rich env)
- the last 2 (op, rd) encoded (for K=3 chain detection)

Since K=3 chains require consciously chaining writes across 3 instructions, the
policy must condition on "what did I write 1-2 steps ago and at what register."
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_cpu_chains import (
    Op, N_BINS, WRITERS, READERS, bins_for_step, advance_history, ChainHistory
)

N_OPS = len(Op)                              # 13 (no JAL in this benchmark)
N_WRITERS = len(WRITERS)                     # 10
WRITER_TO_IDX = {w: i for i, w in enumerate(WRITERS)}
OP_TO_IDX = {op: i for i, op in enumerate(Op)}
# Observation layout:
#   [0 : N_BINS]                                  coverage vector
#   [N_BINS : +32*N_WRITERS]                      per-reg writer one-hot
#   [... : +32*4]                                 per-reg age one-hot (1,2,3,other)
#   [... : +2*(N_OPS+32)]                         last 2 (op_one_hot, rd_one_hot)
OBS_DIM = N_BINS + 32 * N_WRITERS + 32 * 4 + 2 * (N_OPS + 32)


class IbexChainsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 2048, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32])
        self.observation_space = spaces.MultiBinary(OBS_DIM)
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.hist = ChainHistory()
        self.step_idx = 0

    def _obs(self) -> np.ndarray:
        writer_flat = np.zeros(32 * N_WRITERS, dtype=np.int8)
        age_flat = np.zeros(32 * 4, dtype=np.int8)
        for r in range(32):
            w = self.hist.writer[r]; a = self.hist.age[r]
            if w is not None and 1 <= a <= 3:
                writer_flat[r * N_WRITERS + WRITER_TO_IDX[w]] = 1
                age_flat[r * 4 + (a - 1)] = 1
            else:
                age_flat[r * 4 + 3] = 1

        last_two_flat = np.zeros(2 * (N_OPS + 32), dtype=np.int8)
        for slot, entry in enumerate(self.hist.last_two):
            op_e, rd, _, _ = entry
            last_two_flat[slot * (N_OPS + 32) + OP_TO_IDX[op_e]] = 1
            last_two_flat[slot * (N_OPS + 32) + N_OPS + rd] = 1
        return np.concatenate([self.covered, writer_flat, age_flat, last_two_flat])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.hist.reset()
        self.step_idx = 0
        return self._obs(), {}

    def step(self, action):
        op = int(action[0]); rd = int(action[1]); rs1 = int(action[2]); rs2 = int(action[3])
        new_hits = 0
        for b in bins_for_step(op, rd, rs1, rs2, self.hist):
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
