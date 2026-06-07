"""Gymnasium env that trains against real Verilator-compiled Ibex.

Each episode:
  1. The agent picks N actions (op, rd, rs1, rs2, imm_bucket).
  2. We encode them to RISC-V machine code via the Level-6 codec.
  3. We write the program to JSON and invoke Vtop with MODULE=test_run_for_l5.
  4. Verilator's --coverage flag writes coverage.dat as a side effect.
  5. We parse coverage.dat and the reward = newly-covered toggle points
     (relative to a per-episode baseline measured at reset).

Action / observation shapes mirror Level 6 exactly so the warm-start from
ppo_l6.zip is a drop-in. The shadow runs alongside (cheaply) just to provide
the observation; the reward signal is real-RTL only.
"""

import os, sys, json, subprocess, tempfile
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Pull in Level-6 shadow + codec + env scaffolding
THIS_DIR = Path(__file__).resolve().parent
LEVEL6_DIR = THIS_DIR.parent / "level4_full_5615bin"
sys.path.insert(0, str(LEVEL6_DIR))

from shadow_cpu_l6 import (
    Op, N_BINS, WRITERS, IMM_BUCKETS as N_IMM_BUCKETS, RAW_MAX_DIST,
    L6History, bins_for_step, advance_history,
)
from cpu_env_l6 import IbexL6Env, OBS_DIM, N_OPS
from codec_l6 import encode

import cov_parser

ML4DV_DIR = (THIS_DIR.parent.parent / "cpu").resolve()
VTOP = ML4DV_DIR / "sim_build" / "Vtop"
COVDAT = ML4DV_DIR / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l5_program.json"
WFI = 0x10500073


class RealRTLToggleEnv(gym.Env):
    """One Vtop invocation per episode. Reward = Δ toggle covered."""

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 1024, seed: int | None = None,
                 reward_scale: float = 1.0, kind: str = "toggle"):
        super().__init__()
        if not VTOP.exists():
            raise FileNotFoundError(f"Prebuilt Vtop not found at {VTOP}")
        self.episode_steps = episode_steps
        self.reward_scale = reward_scale
        self.kind = kind
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, N_IMM_BUCKETS])
        self.observation_space = spaces.MultiBinary(OBS_DIM)

        # Inner Level-6 env supplies the observation (coverage + history)
        self._inner = IbexL6Env(episode_steps=episode_steps, seed=seed)

        self._action_buffer: list[tuple[int, int, int, int, int]] = []
        self._step_idx = 0
        self._baseline_hits: set[str] = set()
        self._cum_real_hits: set[str] = set()
        self._episode_count = 0

    def reset(self, *, seed=None, options=None):
        obs, _ = self._inner.reset(seed=seed)
        self._action_buffer.clear()
        self._step_idx = 0
        # baseline = empty set per-episode; the reward is "what this episode covers"
        self._baseline_hits = set()
        return obs, {"episode_count": self._episode_count}

    def step(self, action):
        op = int(action[0]); rd = int(action[1])
        rs1 = int(action[2]); rs2 = int(action[3]); ib = int(action[4])

        # Drive the inner env so the next obs reflects shadow state
        obs, _, _, _, _ = self._inner.step(action)
        self._action_buffer.append((op, rd, rs1, rs2, ib))
        self._step_idx += 1

        truncated = self._step_idx >= self.episode_steps
        terminated = False
        reward = 0.0
        info: dict = {}

        if truncated:
            real_hits, n_total = self._run_and_measure()
            new_hits = real_hits - self._baseline_hits
            reward = self.reward_scale * len(new_hits)
            self._cum_real_hits |= real_hits
            self._episode_count += 1
            info.update({
                "ep_real_covered": len(real_hits),
                "ep_real_total": n_total,
                "ep_real_pct": 100.0 * len(real_hits) / n_total if n_total else 0.0,
                "ep_new_hits": len(new_hits),
                "cum_real_covered": len(self._cum_real_hits),
            })

        return obs, reward, terminated, truncated, info

    def _run_and_measure(self) -> tuple[set[str], int]:
        machine = [encode(*a) for a in self._action_buffer]
        with open(PROGRAM_JSON, "w") as f:
            json.dump({
                "n": len(machine),
                "agent": "rl",
                "seed": 0,
                "machine_code": [int(m) for m in machine],
            }, f)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/usr/lib/x86_64-linux-gnu:/home/andre/anaconda3/lib"
            + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        env["MODULE"] = "test_run_for_l5"
        env["RL_L5_JSON"] = PROGRAM_JSON
        # Run Vtop from cpu/ so coverage.dat lands in the expected place
        proc = subprocess.run(
            [str(VTOP)],
            cwd=str(ML4DV_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if proc.returncode != 0:
            return set(), 0

        summary = cov_parser.parse(str(COVDAT))
        covered, total = summary.by_kind.get(self.kind, (0, 0))
        # We want the SET of point keys hit (so we can diff against baseline)
        prefix = f"\x01page\x02v_{self.kind}/"
        hits = {k for k, v in summary.points.items() if v > 0 and prefix in ("\x01" + k)}
        # Sanity: |hits| should equal `covered` from by_kind
        return hits, total
