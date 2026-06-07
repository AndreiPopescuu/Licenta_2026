# RL and stimulus-engineering experiments for Ibex coverage closure

Seven iterations of making a program generator push Verilator toggle coverage
higher on the real lowRISC Ibex RTL. The final configuration (Level 7) reaches
**66.27% cumulative toggle in 30 random episodes** on a minimal Ibex build with
a reachable ceiling of 75.9% — up from **56.2% at L5 PPO-300-episodes**.

The headline finding is at the bottom of this README.

> **New reader?** [`JOURNEY.md`](JOURNEY.md) walks through the seven levels in
> narrative form — what each level was trying to answer, what happened, what
> it taught, and what the next level was. Recommended before reading the
> per-level code.


## TL;DR

| level | what's new | evaluated on | best result |
|------:|------------|--------------|------------:|
| L1    | Decoder RL (combinational, 2107 bins)                           | shadow decoder | ~100% on reachable bins |
| L2    | CPU RL on 196 LLM4DV bins (shadow only)                         | shadow CPU     | 196/196 |
| L3    | 1739-bin chained shadow                                         | shadow CPU     | 99%+    |
| **L4**| **5615-bin full shadow, PPO trained**                           | shadow + real RTL validation | PPO ≈ **100x** faster than random to saturate shadow; 425/425 and 1499/1499 shadow-to-real match |
| L5    | 45-op encoder, first run on real Verilator toggle coverage      | real Ibex RTL  | 56.20% (PPO rich, 300 eps) |
| L6    | +16 RVC ops (compressed instructions)                           | real Ibex RTL  | 57.48% (random 150 eps); `compressed_decoder` 0% → 97% |
| **L7**| **+AUIPC, ECALL, EBREAK; 29 CSRs; mem prepop + trap handler**   | real Ibex RTL  | **66.27%** (random 30 eps) — ~87% of the 75.9% reachable ceiling |


## Directory map

```
rl-coverage/
├── level1_decoder/          # combinational decoder, 2107 bins
├── level2_cpu_196bin/       # LLM4DV-benchmark shadow env
├── level3_chains/           # 1739-bin chained coverage
├── level4_shadow/           # 5615-bin shadow + PPO (100x speedup story)
├── level5_real_rtl/         # first real Verilator toggle coverage runs (45 ops)
├── level6_rvc/              # +16 RVC ops, characterises reachable ceiling
└── level7_stimulus/         # +AUIPC/ECALL/EBREAK, trap handler, 29 CSRs — the win
```

The cocotb drivers that bridge L5/L6/L7 to real Verilator live in
`../cpu/test_run_for_l5.py` and `../cpu/test_run_for_l7.py`. They share the
prebuilt `../cpu/sim_build/Vtop` that the rest of the repo uses.
`../cpu/test_validate_l6.py` runs a shadow-backed monitor on the real Ibex
retirement stream and compares bin hits against the shadow's prediction; this
is how the "425/425 and 1499/1499 shadow-to-real match" number in the TL;DR
was produced.

Two pieces of shared infrastructure sit inside `level5_real_rtl/` (imported
from L6 and L7 via `sys.path`):

- **`cov_parser.py`** — parses Verilator's binary `coverage.dat` into per-bin
  hit counts, grouped by kind (toggle / branch / line) and by source file.
  Everything downstream of "run Vtop" uses this.
- **`analyze_unreachable.py`** (one copy in L6, one in L7) — walks the
  `coverage.dat` uncovered-bin set and classifies each signal as TIED-OFF
  (constant due to config, e.g. PMP, ICache, debug, RV32B, SecureIbex, MHPM,
  RVFI fanout), NEEDS-this-feature (requires enabling a config option),
  REACHABLE-but-uncovered, or unknown. The **75.9% reachable ceiling** quoted
  for L7 (and the earlier **~69.6%** quoted at L6) comes from subtracting the
  TIED-OFF set from the total. The L7 number is higher because L7's trap
  handler makes `mepc` / `mtval` / `mcause` / controller exception states
  reachable — they were counted as TIED-OFF at L6.


## Dependencies

On top of Verilator 5.x and cocotb (see the top-level README), L1–L7 need:

```
pip install gymnasium stable-baselines3 numpy matplotlib
```

All RL training runs were done on CPU; no GPU required.


## How to reproduce the L7 result (the headline number)

```bash
# One-time build of Verilator Ibex (writes cpu/sim_build/Vtop)
cd cpu/
make

# Measure the 30-episode random baseline on the L7 env
cd ../rl-coverage/level7_stimulus/
python smoke_l7.py               # quick sanity check
python measure_l7_random.py      # ~2 min wall time, writes l7_random_baseline.npz
python analyze_unreachable.py    # partitions uncovered bins into TIED-OFF / NEEDS / REACHABLE?
python plot_l7.py                # regenerates l7_comparison.png
```

Environment requirements:
- Verilator 5.x, cocotb 1.8–1.x, Python 3.11+.
- `LD_LIBRARY_PATH` must include `/usr/lib/x86_64-linux-gnu`. If you run under
  Anaconda Python, also add `$CONDA_PREFIX/lib` so Vtop picks up the right
  `libpython`.
- The driver writes programs to `/tmp/rl_l7_program.json` and expects
  `cpu/sim_build/Vtop` to exist.


## How to reproduce the other levels

Each level is self-contained; scripts follow the same naming convention.

```bash
# L1 — combinational decoder (shadow only, ~1 min)
cd rl-coverage/level1_decoder/
python train.py                     # trains a PPO agent against the shadow decoder
python plot.py                      # re-renders curves.png

# L2 — 196-bin LLM4DV benchmark (shadow only)
cd ../level2_cpu_196bin/
python train_cpu.py                 # PPO vs the 196-bin shadow
python train_and_emit.py            # also dumps a concrete program for real RTL

# L3 — 1739-bin chained shadow
cd ../level3_chains/
python train_chains.py

# L4 — 5615-bin full shadow (the 100x-speedup finding)
cd ../level4_shadow/
python train_l6.py                  # ~minutes of CPU; writes ppo_l6.zip (gitignored, 87 MB)
python rl_emit_l6.py                # emits a program for real-RTL validation

# Validate an L4 agent on the real CPU:
cd ../../cpu/
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_validate_l6 ./sim_build/Vtop

# L5 — first real Verilator toggle coverage (three PPO variants lived here)
cd ../rl-coverage/level5_real_rtl/
python smoke_test.py                # sanity check: one program hits Vtop and returns coverage
python measure_l5_baselines.py      # random + uniform-op baselines
python train_l5.py                  # PPO with basic observation (saves ppo_l5_curve.npz)
python train_l5_rich.py             # PPO with rich observation incl. novelty bonus (ppo_l5_rich.npz)
# ppo_l5_novelty.npz is an intermediate novelty-only variant kept for comparison.

# L6 — +16 RVC ops
cd ../level6_rvc/
python smoke_rvc.py
python measure_rvc_baseline.py      # random baseline (saturates at 57.48% at 150 eps)
python train_rvc.py                 # rich-obs PPO, same as L5_rich but with RVC codec
python analyze_unreachable.py       # first run of the TIED-OFF/NEEDS/REACHABLE classifier (~69.6%)
```

The shadow envs (L1–L4) run in pure Python at roughly **1,500× real-sim
speed**, which is why PPO training is practical there. L5–L7 call out to
Vtop every episode (a few seconds per 1024-instruction program), so they
budget a few dozen to a few hundred episodes rather than thousands.


## What actually moved the number (Level 7 findings)

The L5/L6 work spent time trying to make PPO beat a random baseline and
couldn't — both saturated at ≈56% on the 45-op encoder. The L7 pivot
established the **stimulus vs algorithm** finding: four non-RL changes lifted
cumulative toggle from 56% to 66% with a random agent.

1. **Data-memory prepopulation.** `MemAgent` used to return `0x10500073` (WFI)
   for every unwritten address, so every load saw a constant. Replaced with
   `addr XOR 0xDEADBEEF`. This single change lifted `ibex_load_store_unit`
   toggle from 38% to 92%.

2. **Trap-safe ECALL / EBREAK.** A two-instruction prologue sets
   `mtvec = 0x00200000`, and a four-instruction trap handler at that address
   advances `mepc` by 4 before `MRET`. ECALL/EBREAK now return instead of
   looping, which unlocks the exception path — `mepc_q`, `mtval_q`, `mcause_q`,
   `mstack_*`, and the controller's exception FSM states.

3. **29 CSR addresses instead of 5.** The L6 codec rotated through 5
   hand-picked safe CSRs. L7 covers `mcycle`, `minstret`, `mcycleh`,
   `minstreth`, `mcountinhibit`, `mcause`, `mtval`, `mhpmcounter3..8`,
   `mhpmevent3..8`, plus the user-level cycle/time/instret. CSR addresses are
   now indexed by `(imm_bucket*7 + rs1) % 29` so the 5-bucket action space can
   still reach all 29. `ibex_cs_registers` toggle rose from ~10% to 22% in one
   run.

4. **AUIPC.** Unlocks `imm_u_type_o` and related ALU upper-immediate paths.
   Small on its own but compounds with the other three changes.

**Meta-finding:** on this class of coverage problem, stimulus engineering (the
action space, the memory model, the trap scaffolding) moved the number
substantially more than any RL algorithm change. Random and PPO were both
bounded by the same ceiling — the ceiling was set by what the generator could
emit and what the memory model returned, not by the optimizer. Once the
generator was extended, a random agent in 30 episodes beat the best previous
PPO run at 300 episodes by 10 percentage points.


## What's next (ranked by payoff per hour)

1. **Greedy directed stimulus** for the remaining 9.6 points to 75.9% ceiling.
   The uncovered reachable bins tend to need instruction *sequences*
   (e.g. `LUI 0xFFFFF` → `MUL rd, rs, rs` to toggle `imd_val_q_i[0][*]`).
   Enumerate templates, emit each, keep net-positive ones. Likely closes 3–5
   points in ~1 day. Suggested location: `rl-coverage/level8_directed/`.

2. **Shadow-dense-reward PPO on the L7 env** (~2 hours). Use
   `level4_shadow/shadow_cpu_l6.py` for per-step bin rewards, evaluate each
   episode against real Vtop toggle coverage. First time RL has a real
   frontier to chase — either PPO beats random by 2–5 pts or it ties, and
   either outcome is publishable.

3. **Rebuild Ibex in the opentitan config** (~1–2 days). Flip PMP, ICache,
   debug triggers, writeback stage, and SecureIbex in `cpu/cocotb_ibex.sv`.
   Surface grows from ~20k to 50–100k toggle points and our numbers become
   directly comparable to lowRISC's public 88.7% branch / 90% functional
   baseline.


## Attribution

- Shadow coverage bins for Levels 2–4 follow the LLM4DV methodology
  ([ZixiBenZhang/ml4dv](https://github.com/ZixiBenZhang/ml4dv), Apache-2.0).
- The `instruction_monitor.py` bug fix (`last_insn` clearing on idle cycles)
  from the parent repo applies to L2–L4 shadow work.
