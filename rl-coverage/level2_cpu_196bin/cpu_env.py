"""
Gymnasium env for the 196-bin CPU coverage model.

Why this is harder than the decoder env:
- The environment has persistent state -- the previous instruction's rd.
- 143/196 bins are RAW hazards, only reachable by chaining writer->reader with
  matching register. Random picks the wrong register 31/32 of the time.
- The observation must tell the policy what the previous instruction did, or it
  cannot plan hazards. We include (prev_writer_one_hot, prev_rd_one_hot).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_cpu import Op, N_BINS, bins_for_step, WRITERS

N_OPS = len(Op)  # 14
N_WRITERS = len(WRITERS)  # 11


class IbexCpuEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 512, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        # action = (op, rd, rs1, rs2, imm_sign_bucket)
        # imm_sign_bucket: 0 = negative, 1 = zero, 2 = positive (only matters for JAL)
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, 3])
        # observation = coverage vector + prev_writer one-hot(+1 for "none") + prev_rd one-hot(+1 for "none"/x0)
        obs_dim = N_BINS + (N_WRITERS + 1) + 33
        self.observation_space = spaces.MultiBinary(obs_dim)
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.prev_writer: Op | None = None
        self.prev_rd: int | None = None
        self.step_idx = 0

    def _obs(self) -> np.ndarray:
        prev_w = np.zeros(N_WRITERS + 1, dtype=np.int8)
        if self.prev_writer is None:
            prev_w[-1] = 1
        else:
            prev_w[WRITERS.index(self.prev_writer)] = 1
        prev_r = np.zeros(33, dtype=np.int8)
        if self.prev_rd is None:
            prev_r[-1] = 1
        else:
            prev_r[self.prev_rd] = 1
        return np.concatenate([self.covered, prev_w, prev_r])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.prev_writer = None
        self.prev_rd = None
        self.step_idx = 0
        return self._obs(), {}

    def step(self, action):
        op_i, rd, rs1, rs2, imm_bucket = [int(x) for x in action]
        imm_sign = {0: -1, 1: 0, 2: +1}[imm_bucket]
        new_hits = 0
        for b in bins_for_step(op_i, rd, rs1, rs2, imm_sign, self.prev_writer, self.prev_rd):
            if not self.covered[b]:
                self.covered[b] = 1
                new_hits += 1
        # Update sequential state for the *next* step.
        op = Op(op_i)
        from shadow_cpu import WRITERS as W_SET
        if op in W_SET:
            self.prev_writer = op
            self.prev_rd = rd
        else:
            # stores don't write a register; preserve prior write? In the real pipeline,
            # `last_insn` is the most recent retired instruction regardless. LLM4DV's monitor
            # tracks only ops that actually set rd for raw-hazard purposes -- we match that.
            # (A store's rd is not meaningful, so we leave prev_writer/prev_rd alone so the
            # next reader can still hit the hazard from two-back, but the paper's definition
            # requires *immediately* prior. We follow the stricter rule: non-writers clear.)
            self.prev_writer = None
            self.prev_rd = None
        self.step_idx += 1
        terminated = bool(self.covered.sum() == N_BINS)
        truncated = self.step_idx >= self.episode_steps
        return self._obs(), float(new_hits), terminated, truncated, {
            "covered": int(self.covered.sum()),
        }
