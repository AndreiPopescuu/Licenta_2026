# Level 7 — Stimulus engineering (the real story)

## TL;DR

Random agent on Ibex toggle coverage, by level:

| level | what changed                                                       | eps | cum toggle |
|-------|--------------------------------------------------------------------|----:|-----------:|
| L5    | 45 base ops (R-type, I-type, loads, stores, 5 CSRs, MUL/DIV, BR, JAL) |  300 (PPO rich) | 56.20% |
| L6    | + 16 RVC ops                                                       | 150 (random)    | 57.48% |
| L6    | + 16 RVC ops, PPO rich                                             | 150 (PPO)       | 57.03% |
| **L7**| **+ AUIPC, ECALL, EBREAK; 29 CSRs rotated; memory prepop; trap handler** | **30 (random)** | **66.27%** |

**L7 random at 30 episodes beats L5 PPO at 300 episodes by 10.07 percentage points and reaches ~87% of the 75.9% hard reachable ceiling of this minimal-config Ibex build.**

One random 1024-instruction program on L7 covers **59.72%** of toggle bins — higher than *any* L5/L6 configuration ever achieved cumulatively.

## What actually moved the number

Three changes, none of them RL:

**1. Data-memory prepopulation.** `MemAgent` used to return `0x10500073` (WFI) for every unwritten address, so every load hit the same constant and `data_rdata_i` had the same bit pattern on every read. Replaced with `addr XOR 0xDEADBEEF` for read-misses. This single change lifted `ibex_load_store_unit` toggle from 38% to 92%.

**2. Trap-safe ECALL / EBREAK.** Added a two-instruction prologue that sets `mtvec = 0x00200000`, and placed a 4-instruction trap handler there that advances `mepc` before MRET. The agent can now safely emit ECALL/EBREAK and the CPU returns correctly instead of looping on its own PC. This unlocks the entire exception path: `mepc_q`, `mtval_q`, `mcause_q`, `mstack_*`, the controller's exception FSM states.

**3. 29 CSR addresses instead of 5.** The L6 codec rotated through 5 hand-picked safe CSRs (mscratch + 4 read-only IDs). Expanded to 29 including `mcycle`, `minstret`, `mcycleh`, `minstreth`, `mcountinhibit`, `mcause`, `mtval`, `mhpmcounter3..8`, `mhpmevent3..8`, the user-level cycle/time/instret. CSR-address indexing now combines `imm_bucket` and `rs1` so the 5-bucket action space can still reach all 29. Lifted `ibex_cs_registers` toggle from ~10% to 22% in a single run.

**4. AUIPC.** One new op, trivial encoding, unlocks `imm_u_type_o` and related ALU upper-immediate paths. Small by itself but compounds with the other changes.

## Why this matters

I spent the L5 and L6 work trying to make RL beat random on a fixed environment. Both times PPO barely tied random, and I had reasoned carefully about why (sparse reward, stale observations, novelty saturation). That analysis was correct, but it missed the meta-point: **the environment's ceiling was set by the stimulus distribution, not by the optimization algorithm.** Random and PPO were both bounded by what instructions the encoder could emit and what the memory model would return.

Rephrased: we were tuning the knob labelled "algorithm" while the knob labelled "stimulus" had 10x more travel on it.

## What's left

Reachable ceiling on this Ibex config: ≈ 75.9% (excluding tied-off signals — PMP, ICache, debug triggers, RV32B, SecureIbex, MHPM counters, RVFI fanout, writeback stage).

We're at 66.3%. Gap to close: 9.6 points. The remaining uncovered-but-reachable bins are harder to reach — they tend to need specific instruction *sequences* (e.g. populating high-bit operands via `LUI` before `MUL` to toggle `imd_val_q_i[0][*]`), not just a diverse random mix. This is where a **directed search or an RL agent with dense per-step rewards could genuinely contribute** — the remaining gap has the sequential-planning structure that the original L5/L6 environments lacked.

## Next possible lever (untried)

- **Greedy targeted stimulus**: for each uncovered bin, find the template most likely to hit it, emit it, repeat. Likely closes another 3-5 points cheaply.
- **Shadow-dense-reward PPO on the L7 env**: now that the environment has a larger reachable frontier, PPO's dense-reward advantage over random might actually show up.
- **Rebuild in opentitan config** (PMP, ICache, debug, writeback, SecureIbex): expands total surface from 20k to 50-100k toggle points, and our numbers become directly comparable to lowRISC's public 88.7% branch / 90% functional baseline.

## Files

| file                            | purpose                                                       |
|---------------------------------|---------------------------------------------------------------|
| `codec_l7.py`                   | 64-op encoder with AUIPC/ECALL/EBREAK and 29-CSR rotation     |
| `env_l7.py`                     | gym env using the L7 codec and cocotb driver                  |
| `smoke_l7.py`                   | 256-instruction sanity test                                   |
| `measure_l7_random.py`          | 30-ep random baseline                                         |
| `analyze_unreachable.py`        | TIED-OFF / NEEDS / REACHABLE? partition and ceiling report    |
| `plot_l7.py`                    | L5/L6/L7 comparison chart                                     |
| `l7_random_baseline.npz`        | saved curve                                                   |
| `l7_comparison.png`             | the chart                                                     |
| `../../cpu/test_run_for_l7.py` | cocotb driver with memory prepop + trap handler    |
