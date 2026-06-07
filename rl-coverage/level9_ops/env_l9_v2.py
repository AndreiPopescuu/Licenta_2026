"""env_l9_v2.py — L9 env cu obs space extins (32 dims).

Față de env_l9_v1.py:
  - 25 module urmărite (ibex_counter_P1 exclus: MHPMCounterNum=0)
  - n_episodes fraction în obs
  - action history: ultimele 4 opcode-uri (memorie intra-episod)
  - Reward dinamic: weight = 1 / coverage_fraction (neschimbat)

Obs layout (32 dims):
  [0]     step_idx / episode_steps
  [1]     cum_hits / total_toggles
  [2]     min(1, n_episodes / 500)
  [3..27] coverage fraction per modul (25 module)
  [28..31] ultimele 4 opcode-uri alese, normalizate prin N_OPS

Action space: MultiDiscrete([83, 32, 32, 32, 5]) — 83 ops L9.
"""

import os, sys, re, json, subprocess
from collections import Counter, deque
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS   = Path(__file__).resolve().parent
L5     = (THIS.parent / "level5_real_rtl").resolve()
ML4DV  = (THIS.parent.parent / "cpu").resolve()
VTOP   = ML4DV / "sim_build" / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l9_v2.json"

sys.path.insert(0, str(L5))
import cov_parser

from codec_l9 import N_OPS, IMM_BUCKETS, emit_program

MODULES = [
    "ibex_core", "ibex_cs_registers", "ibex_top", "ibex_if_stage",
    "ibex_top_tracing", "ibex_alu", "ibex_id_stage", "ibex_multdiv_fast",
    "ibex_ex_block", "ibex_tracer", "ibex_prefetch_buffer",
    "ibex_controller", "ibex_compressed_decoder", "ibex_register_file_ff",
    "ibex_load_store_unit", "ibex_decoder", "ibex_fetch_fifo",
    "ibex_counter", "ibex_csr", "ibex_wb_stage",
    "cocotb_ibex", "prim_generic_clock_gating",
    "prim_clock_gating", "prim_buf", "prim_generic_buf",
]
N_MODULES   = len(MODULES)
HIST_LEN    = 4
N_OBS       = 3 + N_MODULES + HIST_LEN  # = 32
MAX_EP_NORM = 500.0


_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def _norm_key(key: str) -> str:
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k

def _module_of(key: str) -> str | None:
    m = re.search(r"page\x02v_toggle/([^\x01]+)\x01", key)
    if not m:
        return None
    return m.group(1).split("__")[0]


def run_program(actions):
    machine = emit_program(actions)
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(machine), "agent": "l8",
                   "machine_code": [int(m) for m in machine]}, f)
    env = os.environ.copy()
    cocotb_libs = "/home/andrei/ibex_env/lib/python3.12/site-packages/cocotb/libs"
    env["LD_LIBRARY_PATH"] = (
        cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":/home/andrei/ibex_env/lib/python3.12/site-packages"
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON
    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


class IbexL9V2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 256, seed: int | None = None,
                 initial_hits: set | None = None):
        super().__init__()
        self.episode_steps = episode_steps

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)

        self._actions:    list[tuple] = []
        self._step_idx:   int = 0
        self._n_episodes: int = 0
        self._total_tog:  int = 1

        self._cum_hits:   set = set(initial_hits) if initial_hits else set()
        self._key_to_mod: dict = {}

        self._mod_covered   = {m: 0 for m in MODULES}
        self._mod_total     = {m: 1 for m in MODULES}
        self._mod_total_set = False

        self._action_hist: deque = deque([0] * HIST_LEN, maxlen=HIST_LEN)

        if self._cum_hits:
            for key in self._cum_hits:
                mod = _module_of(key)
                self._key_to_mod[key] = mod
                if mod and mod in self._mod_covered:
                    self._mod_covered[mod] += 1

    def _dynamic_weight(self, mod: str | None) -> float:
        if mod and mod in self._mod_covered and self._mod_total.get(mod, 0) > 0:
            frac = self._mod_covered[mod] / self._mod_total[mod]
            return 1.0 / max(frac, 0.01)
        return 1.0

    def _obs(self) -> np.ndarray:
        step_frac = self._step_idx / self.episode_steps
        cum_cov   = len(self._cum_hits) / max(self._total_tog, 1)
        n_ep_frac = min(1.0, self._n_episodes / MAX_EP_NORM)
        mod_fracs = [self._mod_covered[m] / max(self._mod_total[m], 1)
                     for m in MODULES]
        hist_norm = [h / N_OPS for h in self._action_hist]
        return np.array(
            [step_frac, cum_cov, n_ep_frac] + mod_fracs + hist_norm,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        self._action_hist = deque([0] * HIST_LEN, maxlen=HIST_LEN)
        return self._obs(), {}

    def step(self, action):
        op = int(action[0])
        self._action_hist.append(op)
        self._actions.append(tuple(int(x) for x in action))
        self._step_idx += 1
        truncated = self._step_idx >= self.episode_steps
        reward, info = 0.0, {}

        if truncated:
            summary = run_program(self._actions)
            if summary is None:
                info["vtop_failed"] = True
            else:
                toggle_covered, toggle_total = summary.by_kind["toggle"]
                self._total_tog = max(toggle_total, 1)

                prefix = "\x01page\x02v_toggle/"
                ep_hits  = {_norm_key(k) for k, v in summary.points.items()
                            if v > 0 and prefix in ("\x01" + k)}
                new_hits = ep_hits - self._cum_hits
                self._cum_hits |= ep_hits

                for key in ep_hits:
                    if key not in self._key_to_mod:
                        self._key_to_mod[key] = _module_of(key)

                if not self._mod_total_set:
                    self._mod_total_set = True
                    mod_counts = Counter()
                    for key in summary.points:
                        if key not in self._key_to_mod:
                            self._key_to_mod[key] = _module_of(key)
                        mod = self._key_to_mod[key]
                        if mod:
                            mod_counts[mod] += 1
                    for mod in MODULES:
                        if mod in mod_counts:
                            self._mod_total[mod] = mod_counts[mod]

                for key in new_hits:
                    mod = self._key_to_mod.get(key)
                    if mod and mod in self._mod_covered:
                        self._mod_covered[mod] += 1

                shaped_reward = sum(
                    self._dynamic_weight(self._key_to_mod.get(k))
                    for k in new_hits
                )
                reward = shaped_reward

                cum_pct   = 100.0 * len(self._cum_hits) / self._total_tog
                worst_mod = min(MODULES,
                                key=lambda m: self._mod_covered[m] / max(self._mod_total[m], 1))
                info.update({
                    "ep_pct":          100.0 * toggle_covered / self._total_tog,
                    "cum_covered":     len(self._cum_hits),
                    "cum_pct":         cum_pct,
                    "new_hits_vs_cum": len(new_hits),
                    "shaped_reward":   shaped_reward,
                    "branch_pct":      100.0 * summary.by_kind["branch"][0] /
                                       max(summary.by_kind["branch"][1], 1),
                    "worst_mod":       worst_mod,
                    "worst_pct":       100.0 * self._mod_covered[worst_mod] /
                                       max(self._mod_total[worst_mod], 1),
                    "mod_coverage":    {m: self._mod_covered[m] /
                                        max(self._mod_total[m], 1)
                                        for m in MODULES},
                    "ep_words":        [int(w) for w in emit_program(self._actions)]
                                       if new_hits else None,
                })
            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
