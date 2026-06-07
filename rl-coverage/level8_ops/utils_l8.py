"""utils_l8.py — Simulator interface for L8 training scripts."""

import json, os, subprocess, sys
from pathlib import Path

THIS   = Path(__file__).resolve().parent
L5     = (THIS.parent / "level5_real_rtl").resolve()
ML4DV  = (THIS.parent.parent / "cpu").resolve()
VTOP   = ML4DV / "sim_build" / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_raw.json"

sys.path.insert(0, str(L5))
import cov_parser


def run_raw(machine_code):
    """Run a list of 32-bit instruction words through Vtop and return coverage summary."""
    payload = {
        "n": len(machine_code),
        "agent": "l8",
        "machine_code": [int(w) for w in machine_code],
    }
    with open(PROGRAM_JSON, "w") as f:
        json.dump(payload, f)

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
