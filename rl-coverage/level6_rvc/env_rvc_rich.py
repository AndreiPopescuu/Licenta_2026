"""Rich-observation RVC env — PPO sees per-module coverage so it can target
undercovered RTL modules. Mirrors env_l5_rich.py but with the 61-op codec.
"""

import os, sys, json, subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS_DIR = Path(__file__).resolve().parent
L5_DIR = (THIS_DIR.parent / "level5_real_rtl").resolve()
sys.path.insert(0, str(L5_DIR))
import cov_parser  # noqa: E402

from codec_rvc import N_OPS, IMM_BUCKETS, emit_program

ML4DV_DIR = (THIS_DIR.parent.parent / "cpu").resolve()
VTOP = ML4DV_DIR / "sim_build" / "Vtop"
COVDAT = ML4DV_DIR / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_rvc_program.json"


MODULES = [
    "ibex_core", "ibex_cs_registers", "ibex_top", "ibex_if_stage",
    "ibex_top_tracing", "ibex_alu", "ibex_id_stage", "ibex_multdiv_fast",
    "ibex_ex_block", "ibex_tracer", "ibex_prefetch_buffer",
    "ibex_controller", "ibex_compressed_decoder", "ibex_register_file_ff",
    "ibex_load_store_unit", "ibex_decoder", "ibex_fetch_fifo",
    "ibex_counter", "ibex_csr", "ibex_wb_stage",
    "cocotb_ibex", "prim_generic_clock_gating",
    "prim_clock_gating", "prim_buf", "prim_generic_buf",
    "ibex_counter_P1", "prim_cipher_pkg", "prim_secded_pkg",
]
N_MODULES = len(MODULES)
OBS_DIM = 3 + N_MODULES


def _module_coverage(summary: cov_parser.CovSummary) -> np.ndarray:
    out = np.zeros(N_MODULES, dtype=np.float32)
    cov = defaultdict(lambda: [0, 0])
    for page, (c, t) in summary.by_page.items():
        if not page.startswith("v_toggle/"):
            continue
        page_mod = page[len("v_toggle/"):].split("__")[0]
        cov[page_mod][0] += c
        cov[page_mod][1] += t
    for i, m in enumerate(MODULES):
        c, t = cov.get(m, (0, 0))
        out[i] = c / t if t else 0.0
    return out


def run_program(actions):
    machine = emit_program(actions)
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(machine), "machine_code": [int(m) for m in machine]}, f)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:/home/andre/anaconda3/lib"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["MODULE"] = "test_run_for_l5"
    env["RL_L5_JSON"] = PROGRAM_JSON
    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


class IbexRVCRichEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 1024, seed: int | None = None,
                 kind: str = "toggle", reward_mode: str = "novelty"):
        super().__init__()
        self.episode_steps = episode_steps
        self.kind = kind
        self.reward_mode = reward_mode
        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._actions: list[tuple] = []
        self._step_idx = 0
        self._cum_hits: set[str] = set()
        self._module_cov = np.zeros(N_MODULES, dtype=np.float32)
        self._n_episodes = 0
        self._ep_total = 20023

    def _obs(self) -> np.ndarray:
        cum_pct = len(self._cum_hits) / self._ep_total if self._ep_total else 0.0
        head = np.array([
            self._step_idx / self.episode_steps,
            min(1.0, self._n_episodes / 300.0),
            cum_pct,
        ], dtype=np.float32)
        return np.concatenate([head, self._module_cov])

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        return self._obs(), {}

    def step(self, action):
        self._actions.append(tuple(int(x) for x in action))
        self._step_idx += 1
        truncated = self._step_idx >= self.episode_steps
        reward, info = 0.0, {}
        if truncated:
            summary = run_program(self._actions)
            if summary is None:
                info["vtop_failed"] = True
            else:
                covered, total = summary.by_kind[self.kind]
                self._ep_total = total
                prefix = f"\x01page\x02v_{self.kind}/"
                hits = {k for k, v in summary.points.items() if v > 0 and prefix in ("\x01" + k)}
                new_hits = hits - self._cum_hits
                self._cum_hits |= hits
                self._module_cov = _module_coverage(summary)
                reward = float(len(new_hits) if self.reward_mode == "novelty" else len(hits))
                info.update({
                    "ep_covered": covered,
                    "ep_total": total,
                    "ep_pct": 100.0 * covered / total if total else 0.0,
                    "new_hits": len(new_hits),
                    "cum_covered": len(self._cum_hits),
                    "cum_pct": 100.0 * len(self._cum_hits) / total if total else 0.0,
                    "branch_pct": 100.0 * summary.by_kind["branch"][0] / max(summary.by_kind["branch"][1], 1),
                })
            self._n_episodes += 1
        return self._obs(), reward, False, truncated, info
