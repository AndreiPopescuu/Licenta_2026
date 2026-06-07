# Level 6 — RVC Encoder Extension

## Goal

The Level 5 work plateaued at ~56% cumulative toggle coverage on minimal-config
Ibex and identified the cause as encoder capability rather than learning
capability: the 45-op action space could not emit RV32C (compressed)
instructions, leaving the entire `ibex_compressed_decoder` module idle. Level 6
extends the encoder with 16 RVC opcodes to test whether unblocking that
module moves the cumulative ceiling.

## What was built

- **`codec_rvc.py`** — a 61-op encoder combining Level 5's 45 base ops with
  16 RV32C ops chosen to exercise every CI, CR, CA, CB, CIW, CL, CS format
  the Ibex `compressed_decoder` implements. Every encoding is guaranteed
  legal (the Ibex-specific illegal-pattern checks — `C.ADDI4SPN` with
  `nzuimm=0`, `C.LUI` with `nzimm6=0`, `C.LUI` with `rd∈{0,2}`, etc — are
  all rejected by construction). A self-test encodes every op × every imm
  bucket × four register tuples and asserts opcode bits.
- **Packing strategy** — each action still produces exactly one 32-bit
  program word so memory layout stays simple. RVC ops pack as
  `(C.NOP << 16) | rvc_word`: Ibex executes the RVC at PC, the C.NOP
  filler at PC+2, then advances to the next word.
- **`env_rvc.py` / `env_rvc_rich.py`** — same subprocess pattern and reward
  shape as L5's envs, only the codec differs. The rich variant gives PPO
  per-module coverage as part of the observation.
- **`measure_rvc_baseline.py`** — 30-episode random baseline on the
  61-op space.
- **`train_rvc.py`** — PPO with the same hyperparameters as `train_l5_rich.py`
  so the curves are directly comparable.
- **`analyze_unreachable.py`** — parses the last run's `coverage.dat` and
  classifies every uncovered toggle point as TIED-OFF (unreachable in our
  minimal Ibex config), NEEDS (reachable but requires stimulus we don't
  currently emit — ECALL, AUIPC, wider state diversity), or REACHABLE?
  (likely reachable, unclassified). Reports a hard reachable ceiling.
- **`plot_rvc.py`** — overlays L5 PPO curves and L6 RVC curves.

## Key finding 1 — compressed_decoder activation

With a pure-RVC smoke program (256 instructions cycling all 16 RVC ops),
the `ibex_compressed_decoder` module goes from 0% to **98.6% toggle
coverage** in a single run. Under the mixed 61-op space a single random
episode keeps it above **97%**. The module that was listed as the L5
bottleneck is now effectively saturated.

## Key finding 2 — cumulative ceiling barely moved

| configuration                                 | episodes | cum toggle |
|-----------------------------------------------|---------:|-----------:|
| L5 PPO rich      (45-op base encoder)         |      300 |   56.20%   |
| L6 random        (61-op RVC encoder)          |       30 |   55.90%   |
| L6 PPO rich      (61-op RVC encoder)          |      150 |   57.03%   |
| **L6 random     (61-op RVC encoder)**         |  **150** | **57.48%** |

Both 150-episode L6 runs exceed the L5 300-episode ceiling. The delta from
the 45-op to the 61-op encoder is **~1.3 percentage points** of cumulative
toggle coverage — roughly twice what "70 new compressed_decoder bins out
of 20,023" would predict on its own, because the RVC ops also stress
decoder/fetch-fifo paths in the base modules via the expansion logic.

**PPO does not outperform random on this environment.** L6 random (150 eps)
ends at 57.48%, L6 PPO (150 eps) at 57.03%. This mirrors the L5 finding:
with 1024-step episodes and a wide action space, uniform random already
samples broadly enough to saturate the reachable-without-new-stimulus
surface in under 30 episodes, and novelty reward becomes too sparse for
on-policy PPO to exploit. The contribution of this work is therefore the
**encoder extension and the resulting ceiling lift**, not an RL advance.

## Key finding 3 — the real reachable ceiling is ~70%, not 56%

`analyze_unreachable.py` partitions the ~10,700 uncovered toggle points:

| bucket     | count  | why                                                       |
|------------|--------|-----------------------------------------------------------|
| TIED-OFF   | ~6,100 | unreachable given our minimal config                      |
| NEEDS      |   ~260 | reachable but requires specific stimulus we don't emit    |
| REACHABLE? | ~5,500 | likely reachable, classification unclear                  |

The TIED-OFF set is dominated by:
- PMP (`PMPEnable=0`) — PMP registers and check logic
- ICache (`ICache=0`) — tag / data / scramble signals
- Debug (`DbgTriggerEn=0`) — triggers, dpc, dcsr, mcontext
- Writeback stage (`WritebackStage=0`)
- Branch predictor (`BranchPredictor=0`)
- Secure Ibex (`SecureIbex=0`) — dummy_instr, scramble
- MHPM (`MHPMCounterNum=0`) — performance counters
- RVFI fanout signals
- B extension datapath (`RV32B=RV32BNone`) — butterfly, minmax, shifter reversal

Excluding tied-off gives a **hard reachable ceiling of ~69.6%** on this
config. Our 56% plateau is therefore at ~80% of the reachable frontier,
not at a true ceiling.

## Key finding 4 — what the next lever is

Inside the ~260 NEEDS bucket the biggest structural group is the exception
path: `mepc_q`, `mtval_q`, `mcause_q`, `mstack_epc_q`, plus controller FSM
states like `WAIT_EXC`. These require an `ECALL` or `EBREAK` to fire,
followed by an `MRET` to return. Adding a minimal trap prologue (set
`mtvec` to a small stub that does `MRET`) plus two new ops (ECALL, EBREAK)
would unlock this whole group — estimated +5-10 percentage points to
cumulative toggle on this config.

## Why PPO does not beat random on this environment

Consistent with the L5 rich finding: with 1024-step episodes and 61 ops ×
32³ × 5 imm buckets = ~10M action combinations, uniform random already
samples broadly enough to saturate the reachable-without-new-stimulus
frontier in ~20 episodes. Novelty reward becomes sparse after that. PPO's
on-policy updates can't easily out-plan random without either (a) longer
horizons that let sequential dependencies pay off, or (b) specific reward
shaping that targets individual undercovered bins. This is a property of
the environment, not of the RVC extension.

If we want RL to *meaningfully* out-plan random, the environment needs to
reward sequential planning. One option we haven't tried: per-episode
*initial state* variation — e.g. the first 8 instructions form a fixed
prologue the agent writes a *handler* for. Another: explicit targeting of
individual uncovered bins (contextual bandit on undercovered points).

## Files in this directory

| file                       | purpose                                         |
|----------------------------|-------------------------------------------------|
| `codec_rvc.py`             | 61-op encoder (45 base + 16 RVC) with self-test |
| `env_rvc.py`               | basic gym env                                   |
| `env_rvc_rich.py`          | gym env with per-module obs for PPO             |
| `smoke_rvc.py`             | 1-episode RVC-only + mixed sanity test          |
| `measure_rvc_baseline.py`  | 30-episode random baseline                      |
| `train_rvc.py`             | PPO trainer                                     |
| `analyze_unreachable.py`   | classifies uncovered points into TIED / NEEDS / REACHABLE? |
| `plot_rvc.py`              | comparison chart                                |
| `rvc_random_baseline.npz`  | curve from random baseline                      |
| `ppo_rvc_rich.npz`         | curve from PPO training                         |
| `ppo_rvc_rich.zip`         | trained model                                   |
| `rvc_comparison.png`       | the plot                                        |

## Reproducing

```
# from this directory, with LD_LIBRARY_PATH + anaconda python:
/home/andre/anaconda3/bin/python codec_rvc.py           # self-test
/home/andre/anaconda3/bin/python smoke_rvc.py           # 1-ep sanity check
/home/andre/anaconda3/bin/python measure_rvc_baseline.py   # random 30 eps
/home/andre/anaconda3/bin/python train_rvc.py --episodes 150  # PPO 150 eps
/home/andre/anaconda3/bin/python analyze_unreachable.py     # reachable-ceiling report
/home/andre/anaconda3/bin/python plot_rvc.py             # comparison plot
```

The pre-built Verilator binary and `coverage.dat` sink live at
`../../cpu/sim_build/Vtop` and are shared with Level 5.
