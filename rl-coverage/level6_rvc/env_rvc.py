"""Gym env for Level 6 (RVC-extended action space).

Identical subprocess pattern to level5_real_rtl/env_l5.py: each episode
the agent produces N actions, we encode them with codec_rvc, write a JSON
program, run Vtop (which dumps coverage.dat), and parse the result.

The only difference from L5 is the codec — everything downstream (cocotb
driver, coverage parser, memory layout at 0x100080) is shared.
"""

import os, sys, json, subprocess
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS_DIR = Path(__file__).resolve().parent
L5_DIR = (THIS_DIR.parent / "level5_real_rtl").resolve()
sys.path.insert(0, str(L5_DIR))
import cov_parser  # noqa: E402  — shared with L5

from codec_rvc import N_OPS, IMM_BUCKETS, emit_program

ML4DV_DIR = (THIS_DIR.parent.parent / "cpu").resolve()
VTOP = ML4DV_DIR / "sim_build" / "Vtop"
COVDAT = ML4DV_DIR / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_rvc_program.json"


def run_program(actions: list[tuple[int, int, int, int, int]]) -> cov_parser.CovSummary | None:
    """Encode the actions and run on real Verilator Ibex.

    Returns None if Vtop fails (signal, nonzero exit). Coverage.dat is
    overwritten each run, so the summary reflects only this program.
    """
    machine = emit_program(actions)
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(machine), "agent": "rvc",
                   "machine_code": [int(m) for m in machine]}, f)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:/home/andre/anaconda3/lib"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["MODULE"] = "test_run_for_l5"
    env["RL_L5_JSON"] = PROGRAM_JSON  # same var name the L5 cocotb test reads
    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


class IbexRVCEnv(gym.Env):
    """One Vtop invocation per episode. Reward defaults to total coverage;
    set reward_mode="novelty" to reward only newly-discovered bins.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 1024, seed: int | None = None,
                 kind: str = "toggle", reward_mode: str = "total"):
        super().__init__()
        self.episode_steps = episode_steps
        self.kind = kind
        self.reward_mode = reward_mode
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self._actions: list[tuple[int, int, int, int, int]] = []
        self._step_idx = 0
        self._cum_hits: set[str] = set()
        self._n_episodes = 0

    def _obs(self) -> np.ndarray:
        return np.array([
            self._step_idx / self.episode_steps,
            min(1.0, self._n_episodes / 100)
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        return self._obs(), {}

    def step(self, action):
        self._actions.append(tuple(int(x) for x in action))
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_steps
        reward = 0.0
        info: dict = {}
        if truncated:
            summary = run_program(self._actions)
            if summary is None:
                info["vtop_failed"] = True
            else:
                covered, total = summary.by_kind[self.kind]
                prefix = f"\x01page\x02v_{self.kind}/"
                hits = {k for k, v in summary.points.items()
                        if v > 0 and prefix in ("\x01" + k)}
                new_hits = hits - self._cum_hits
                self._cum_hits |= hits
                reward = float(len(new_hits) if self.reward_mode == "novelty" else len(hits))
                info.update({
                    "ep_covered": covered,
                    "ep_total": total,
                    "ep_pct": 100.0 * covered / total if total else 0.0,
                    "new_hits_vs_cum": len(new_hits),
                    "cum_covered": len(self._cum_hits),
                    "branch_pct": 100.0 * summary.by_kind["branch"][0] / max(summary.by_kind["branch"][1], 1),
                    "line_pct":   100.0 * summary.by_kind["line"][0]   / max(summary.by_kind["line"][1], 1),
                })
            self._n_episodes += 1
        return self._obs(), reward, terminated, truncated, info
