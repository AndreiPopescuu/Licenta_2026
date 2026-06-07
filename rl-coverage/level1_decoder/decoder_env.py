"""
Gymnasium environment wrapping the shadow decoder.

Design decisions (the interesting part of doing RL on this):

- Observation: the 2107-dim coverage bit vector. High-dim but sparse. A small MLP
  handles it. Alternative we did not take: only feed which bins are unhit (mask
  inverted) -- equivalent information, same shape.

- Action: MultiDiscrete([26, 32, 32, 32]) = (op_type, rd, rs1, rs2). Structured
  decomposition of the instruction, mirroring how the decoder actually factors
  the 32-bit word. Naive 2^32 is infeasible; this gives ~850K actions that the
  agent can learn over.

- Reward: number of *new* bins this step. Pure coverage-delta reward is sparse
  but unambiguous -- the agent only gets credit for discovery. Dense variants
  (e.g., -1 per repeat) encourage exploration but distort the objective.

- Episode: fixed horizon (default 256 steps). On reset, coverage clears. Long
  enough that a greedy policy can be evaluated; short enough that PPO sees many
  episodes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_decoder import N_BINS, N_OP_TYPES, bins_for_action


class IbexDecoderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 256, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.observation_space = spaces.MultiBinary(N_BINS)
        self.action_space = spaces.MultiDiscrete([N_OP_TYPES, 32, 32, 32])
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.step_idx = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.step_idx = 0
        return self.covered.copy(), {}

    def step(self, action):
        op, rd, rs1, rs2 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
        new_hits = 0
        for b in bins_for_action(op, rd, rs1, rs2):
            if not self.covered[b]:
                self.covered[b] = 1
                new_hits += 1
        self.step_idx += 1
        terminated = bool(self.covered.sum() == N_BINS)  # 100% is the natural stop
        truncated = self.step_idx >= self.episode_steps
        return self.covered.copy(), float(new_hits), terminated, truncated, {
            "covered": int(self.covered.sum()),
        }
