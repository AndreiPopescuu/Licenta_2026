# Ibex Coverage Experiments

Hands-on verification experiments on the lowRISC Ibex RISC-V core,
demonstrating the stimulus-simulate-measure loop that an AI coverage agent
would run. Everything here uses free, open-source tools and real chip RTL.

## What is in this repo

### `decoder/` -- Combinational decoder (isolated)

The Ibex instruction decoder is a purely combinational block (~1800 lines of
SystemVerilog). We extracted it from the Ibex repo and wrapped it so we can
feed it 32-bit instruction words from Python and instantly read what operation
it decoded. No clock, no pipeline, no memory -- just input-output.

Two tests live here:

- **`test_decoder.py`** runs three phases to illustrate the coverage loop:
  1. Hand-crafted instructions (a human picks ADD, SUB, LW, etc.)
  2. Pure random 32-bit values (baseline -- most are illegal)
  3. Encoding-aware random (understands RISC-V bit fields, much more effective)

- **`test_max_coverage.py`** does a systematic sweep: every operation times
  every register (0..31) across every port. Result: **2041 / 2107 bins
  covered**. The remaining ~65 bins are ISA-unreachable (e.g., RISC-V has no
  SUBI instruction, so the subtract-immediate cross-coverage bins can never
  fire).

The coverage model matches the one from the LLM4DV paper (ZixiBenZhang/ml4dv),
which defines 2107 bins across three types:
  - Type 1: which ALU/memory operations were seen (26 bins)
  - Type 2: which register ports were exercised (96 bins)
  - Type 3: cross-coverage, operation x register (1985 bins)

### `cpu/` -- Full Ibex CPU with RTL toggle coverage

The full Ibex RISC-V CPU (~15K lines of SystemVerilog, minimal configuration).
We load machine-code programs into a simulated instruction memory, let the CPU
execute them, and observe which instructions retire through the RVFI (RISC-V
Formal Interface).

- **`test_cpu_coverage.py`** builds a single program that targets all 196
  LLM4DV CPU benchmark bins. These 196 bins cover six categories:

  | Type | What it measures | Bins |
  |------|-----------------|------|
  | SEEN | Each of the 14 operations executed at least once | 14 |
  | ZERO_DST | R-type or JAL with destination register = x0 | 11 |
  | ZERO_SRC | R-type or S-type with a source register = x0 | 13 |
  | SAME_SRC | R-type or S-type with rs1 == rs2 | 13 |
  | BR | JAL forward + JAL backward | 2 |
  | RAW_HAZARD | Read-after-write data dependency between consecutive instructions | 143 |

  **Result: 196/196 = 100%** (the LLM4DV paper's best published result using
  Claude 3.5 Sonnet was 5.61%). The test programs were written with AI
  assistance, given the coverage spec and ISA documentation as context.

- **`instruction_monitor.py`** is the coverage monitor, adapted from LLM4DV
  with a bug fix. The original code cleared `last_insn` on every idle cycle
  (when `rvfi_valid == 0`). In a pipelined CPU there are always idle cycles
  between instruction retirements, so the original monitor could never see two
  consecutive instructions -- making RAW hazard detection impossible. Our fix:
  only update `last_insn` when a new instruction actually retires.

- **Verilator RTL toggle coverage** (`--coverage` flag in Makefile) measured
  how much of the actual hardware was exercised:

  **985 / 5746 toggle points = 17%**

  Per-module breakdown (selected):

  | Module | Covered | Total | Rate |
  |--------|---------|-------|------|
  | ibex_alu.sv | 47 | 119 | 39% |
  | ibex_controller.sv | 87 | 298 | 29% |
  | ibex_cs_registers.sv | 79 | 422 | 18% |
  | ibex_decoder.sv | 68 | 530 | 12% |
  | ibex_compressed_decoder.sv | 9 | 107 | 8% |
  | ibex_multdiv_fast.sv | 38 | 169 | 22% |

  The 17% is expected: our test only uses ~14 instruction types out of 50+.
  Major uncovered areas include branches (BEQ/BNE/...), loads (LB/LH/LW),
  LUI, AUIPC, CSR operations, MUL/DIV, compressed instructions, exceptions,
  and interrupts. A natural next step is writing tests that exercise these
  paths to push toggle coverage significantly higher.

  That follow-on work lives in [`rl-coverage/`](rl-coverage/) — see
  ["RL and stimulus-engineering experiments"](#rl-and-stimulus-engineering-experiments)
  below.

### `rl-coverage/` -- RL and stimulus-engineering experiments

Seven iterations of a program generator (Levels 1–7) pushing Verilator toggle
coverage up on the real Ibex RTL. The arc starts with small shadow
environments and PPO training, moves to real-RTL validation, and ends with a
stimulus-engineering finding that made a random generator beat PPO by 10
coverage points.

**Headline:** Level 7 reaches **66.27% cumulative toggle in 30 random
episodes** on minimal Ibex (reachable ceiling 75.9%), up from L5's 56.20% at
PPO-300-episodes. The lift came from four non-RL changes: data-memory
prepopulation with an address-XOR pattern (LSU 38%→92%), a trap-safe
prologue + handler unlocking ECALL/EBREAK, a 29-CSR rotation, and AUIPC.

See [`rl-coverage/README.md`](rl-coverage/README.md) for a level-by-level
table and reproduction instructions, or
[`rl-coverage/JOURNEY.md`](rl-coverage/JOURNEY.md) for the narrative
walk-through of how each level came about, what it taught, and what it
pointed at next.

### CPU configuration

The cocotb wrapper (`cocotb_ibex.sv`) instantiates Ibex in a minimal
configuration: no PMP, no ICache, no debug triggers, no security hardening,
2-stage pipeline, no branch predictor. All interrupts are tied to zero.
These parameters can be changed to exercise more of the design.

For comparison, lowRISC's own verification of Ibex targets the "opentitan"
configuration (PMP with 16 regions, ICache, debug, security, 3-stage
pipeline) and achieves 88.7% branch coverage / 90% functional coverage
across ~50K-100K coverage bins using 1530 regression tests and commercial
simulators.


## How to set up

### Prerequisites

Tested on WSL2 Ubuntu with Anaconda Python 3.13.

```
# Verilator (simulator) and compression library
sudo apt-get install verilator zlib1g-dev

# cocotb (Python-to-simulator bridge)
pip install 'cocotb>=1.8,<2.0'

# RL experiments under rl-coverage/ additionally need:
pip install gymnasium stable-baselines3 numpy matplotlib
```

Verilator 5.x is required. Check with `verilator --version`.


## How to run

### Decoder demo

```bash
cd decoder/

# Build (first time, or after editing .sv files)
make

# Run the 3-phase demo
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_decoder ./sim_build/Vtop

# Run the systematic max-coverage sweep
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_max_coverage ./sim_build/Vtop
```

### Full CPU test

```bash
cd cpu/

# Build (first time, or after editing .sv files)
make

# Run the 196-bin coverage test
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_cpu_coverage ./sim_build/Vtop

# Analyze RTL toggle coverage (after running the test)
verilator_coverage --annotate /tmp/ibex_cov_annotate coverage.dat
```

The `--annotate` command writes per-file annotated source to
`/tmp/ibex_cov_annotate/`. Lines with `%000000` were never toggled.

### Build troubleshooting

If the build fails at the `verilator_includer` step (Anaconda's libstdc++
conflicting with system libraries), you can manually concatenate the generated
C++ files. From inside `sim_build/`:

```bash
cat Vtop.cpp Vtop___024root__DepSet_*.cpp Vtop__Dpi.cpp \
    Vtop__Trace__0.cpp Vtop___024root__Slow.cpp \
    Vtop_ibex_pkg__Slow.cpp Vtop_ibex_pkg__DepSet_*.cpp \
    Vtop__Syms.cpp Vtop__Trace__0__Slow.cpp \
    Vtop__TraceDecls__0__Slow.cpp > Vtop__ALL.cpp
make -f Vtop.mk
```

You may also need to set `LD_LIBRARY_PATH` to include both
`/usr/lib/x86_64-linux-gnu` (system libstdc++) and your Python installation's
`lib/` directory (for `libpython3.x.so`).


## Where to go from here

Steps 1–3 below are already done and live in [`rl-coverage/`](rl-coverage/):
adding more instruction types (including 16-bit compressed), exceptions and
interrupts via a minimal trap handler, and a diverse data-memory model. The
result on minimal Ibex is **66.27% cumulative toggle** (30 random episodes),
with a characterised reachable ceiling of **75.9%**.

Step 4 — richer configuration — is the next frontier. Concrete queue:

1. **Greedy directed stimulus** targeting the remaining 9.6 points to the
   75.9% ceiling (1 day). The remaining uncovered-but-reachable bins need
   instruction *sequences* rather than a diverse random mix, which is
   exactly what a greedy template search handles. Suggested location:
   `rl-coverage/level8_directed/`.

2. **Rebuild Ibex in the opentitan config** (~1–2 days). Flip `PMPEnable`,
   `ICache`, `DbgTriggerEn`, `WritebackStage`, `SecureIbex` in
   `cpu/cocotb_ibex.sv`. The coverage surface grows from ~20k to 50–100k
   toggle points and our numbers become directly comparable to lowRISC's
   public 88.7% branch / 90% functional baseline.

3. **Shadow-dense-reward PPO on the Level 7 env** (~2 hours). With a real
   9.6-point reachable frontier now visible, PPO finally has something to
   chase that random saturates on. Either it beats random by 2–5 points or
   it ties — both outcomes are publishable.

The annotated coverage output (`verilator_coverage --annotate`) still shows
exactly which RTL lines are uncovered, making it straightforward to target
specific gaps.


## Attribution

- Ibex RTL: [lowRISC/ibex](https://github.com/lowRISC/ibex), Apache-2.0
- Coverage model and benchmark bins: [ZixiBenZhang/ml4dv](https://github.com/ZixiBenZhang/ml4dv) (LLM4DV), Apache-2.0
- `instructions.py`, `shared_types.py`: from LLM4DV, Apache-2.0 (imports adjusted)
- `instruction_monitor.py`: from LLM4DV, Apache-2.0, with bug fix (idle-cycle `last_insn` clearing removed)
- `test_decoder.py`, `test_max_coverage.py`, `test_cpu_coverage.py`: original work, AI-assisted
