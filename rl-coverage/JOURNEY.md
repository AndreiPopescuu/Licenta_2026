# The seven-level journey

This document walks a reader through how Levels 1–7 came to exist, what each
level was trying to answer, what happened, and what it pointed at next. It is
meant as the "why" companion to the per-level table in
[`README.md`](README.md) and the concrete commands in each level's directory.


## The question we started with

> *Can a program-generating agent (random baseline, reinforcement learning,
> or something hybrid) push Verilator coverage on the real lowRISC Ibex RISC-V
> core to a level comparable with the public lowRISC regression suite, using
> only free open-source tools?*

The public lowRISC baseline is 88.7% branch / 90% functional coverage on the
opentitan-configured Ibex, produced by 1530 hand-written regression tests and
commercial simulators. The question was whether something like RL or
stimulus search could approach that number on Verilator + cocotb with orders
of magnitude less hand-effort.

Seven iterations in, the short answer is:

- **56.2% → 66.3% cumulative toggle coverage** on minimal-config Ibex in 30
  random episodes.
- A characterised **reachable ceiling of 75.9%** for that config (i.e. the
  remaining 24.1% is tied off by configuration choices, not by our stimulus).
- A clear meta-finding that **stimulus engineering (action space, memory
  model, trap scaffolding) matters more than RL algorithm tuning** on this
  class of problem.

What follows is the path that got us there.


## Level 1 — Start small: the combinational decoder

**Question.** Can the generate-simulate-measure loop even close on a tiny,
well-bounded piece of RTL?

**What we built.** The Ibex instruction decoder is a purely combinational
block (~1,800 lines of SystemVerilog, no clock, no pipeline). We extracted
it, wrapped it so Python could feed it 32-bit instruction words, and built
a small RL environment around a 2,107-bin coverage model that mirrors the
LLM4DV benchmark.

**What happened.** A PPO agent trained against a Python "shadow" of the
decoder (same bin logic, no RTL) saturates essentially every reachable bin
in minutes. 2,041 / 2,107 bins hit; the remaining ~65 are ISA-unreachable
(e.g. RISC-V has no SUBI, so subtract-immediate cross-bins never fire).

**What it taught.** The loop works. The shadow is fast enough to train on.
Encoding-aware random already does well on a combinational block — but the
next step is a stateful target where sequence matters.

**Files.** `level1_decoder/`.


## Level 2 — The 196-bin CPU shadow

**Question.** Can the same loop handle a stateful target? The LLM4DV paper
publishes a 196-bin functional coverage model for a CPU with RAW hazards
and zero-register crosses; their best published result with Claude 3.5
Sonnet was **5.61%**. Can a trained agent hit 100%?

**What we built.** A shadow of the full CPU retirement stream that emits
the 196-bin coverage in pure Python (no RTL), plus a PPO trainer on top
of it.

**What happened.** PPO saturates all 196 bins. On real RTL, the same
generated program also hits 196/196 when run through Verilator-compiled
Ibex through RVFI (the `cpu/test_cpu_coverage.py` entry point in the
parent repo).

**What it taught.** The shadow-to-real correspondence holds: a program
that hits a bin in the shadow also hits it on real RTL. That shadow-as-a-
proxy property is what makes the rest of the journey tractable — training
against real Vtop is ~1,500× slower than training against the shadow.

**Files.** `level2_cpu_196bin/`.


## Level 3 — 1,739-bin chained coverage

**Question.** The 196-bin model is shallow; real verification teams care
about chains of events (branch, retire, forward) across many cycles. Does
the shadow approach still work when the coverage model is deeper?

**What we built.** A 1,739-bin shadow with chained coverage (multi-cycle
dependencies, broader RAW patterns), plus the PPO training loop against
it.

**What happened.** PPO saturates 99%+ of the reachable set. Random lags
meaningfully here for the first time — the chained bins have enough
structure that sequential planning starts to pay.

**What it taught.** Chained coverage is within reach for on-policy RL on
the shadow side. Moving past 1.7k bins demanded richer encodings.

**Files.** `level3_chains/`.


## Level 4 — The 5,615-bin full shadow and the 100× finding

**Question.** Can we build a **full** Python shadow for RV32I/M with enough
bins to be publishable as a standalone functional coverage artefact — and
does RL meaningfully out-perform random on it?

**What we built.**

- A 5,615-bin shadow covering single-instruction categories, RAW hazards
  (all operand positions), ZERO-register crosses, SAME-source crosses,
  branch patterns, CSR reads/writes, multiply/divide, compressed
  instructions, and memory ordering.
- A gym env (`cpu_env_l6.py`) that exposes the shadow as per-step bin
  rewards.
- A PPO trainer on top.
- A shadow-to-real validation path: take a trained agent's output
  program, run it on real Verilator Ibex via cocotb, monitor RVFI, and
  check that the bins the shadow *claimed* were hit are the same bins the
  real RTL *actually* hit.

**What happened.**

- PPO saturates the shadow roughly **100× faster** than uniform random.
- Shadow-to-real check on two independent trained agents: **425/425** bin
  match on one, **1,499/1,499** on another — zero divergence in either.
- Wall-clock to saturate the full 5,615-bin shadow: ~2 minutes of PPO vs.
  ~53 minutes of uniform random.

**What it taught.**

- The shadow is the right instrument for training: ~1,500× faster than
  real simulation, with no observed loss of fidelity when validated
  bin-by-bin.
- RL meaningfully out-plans random when the reward is dense and the bins
  have structure the policy can learn.
- We now have a published-grade functional coverage artefact that didn't
  exist: an open, Python-native, 5,615-bin RV32I/M model.

**Files.** `level4_shadow/`. Note `ppo_l6.zip` (87 MB trained model) is in
.gitignore — regenerate with `python train_l6.py`.


## Level 5 — First contact with real Verilator toggle coverage

**Question.** The shadow work is compelling; the *real* research target is
Verilator's RTL toggle coverage on the live Ibex. Does RL's 100×
advantage over random hold up when the environment is real RTL, or does
that advantage live in the shadow?

**What we built.**

- `codec_l5.py` — a 45-op encoder covering R-type ALU, I-type ALU,
  loads, stores, 5 hand-picked safe CSRs (mscratch + four read-only IDs),
  MUL/DIV, branches, JAL. Every emitted instruction is legal by
  construction.
- `env_l5.py` / `env_l5_rich.py` — gym envs. The rich variant's
  observation includes per-module coverage (Ibex decoder, controller,
  cs_registers, ALU, ...) so the policy can see which modules still
  need stimulus.
- `cov_parser.py` — parses Verilator's binary `coverage.dat` into per-bin
  hit counts. All of L5/L6/L7 depends on this.
- Three PPO variants: vanilla (`train_l5.py`), novelty-bonus
  (`ppo_l5_novelty.npz`), and rich-obs (`train_l5_rich.py`).
- Random baselines (`measure_l5_baselines.py`).

**What happened.**

- Real-RTL wall time: a 1024-instruction program takes ~2–3 seconds to
  run through Vtop. RL episode budget drops from "thousands" to "hundreds".
- **Random saturates at ≈56% cumulative toggle coverage by episode 30.**
- **All three PPO variants plateau at the same 56%**, even at 300 episodes.
- Per-module breakdown showed the compressed decoder was at 0% (we couldn't
  emit 16-bit RVC), cs_registers was at ~10% (only 5 CSR addresses),
  load_store_unit at ~38% (loads always hit a constant WFI sentinel).

**What it taught.** The plateau wasn't the algorithm — it was the
environment. Random and PPO hit the same ceiling because both were bounded
by the same encoder and the same memory model. The diagnosis pointed at
three levers: the missing instruction families, the narrow CSR set, and
the constant data-memory pattern.

**Files.** `level5_real_rtl/`. `env_l5_rich.py:30` contains the fixed list
of 30-ish modules the rich observation tracks.


## Level 6 — Extend the encoder, characterise the ceiling

**Question.** Does adding the obvious missing instruction family (RVC,
16-bit compressed) prove the "encoder ceiling, not learning ceiling"
hypothesis, and how high could we possibly go on this config?

**What we built.**

- `codec_rvc.py` — a 61-op encoder (45 L5 base + 16 RVC) covering every
  compressed format the Ibex decoder implements (CI, CR, CA, CB, CIW, CL,
  CS). Illegal patterns (`C.ADDI4SPN nzuimm=0`, `C.LUI rd∈{0,2}`, etc.)
  are rejected at encoding time. RVC ops are packed as
  `(C.NOP << 16) | rvc_word` so every action still produces one 32-bit
  program word — memory layout stays trivial.
- `env_rvc.py` / `env_rvc_rich.py` — same subprocess pattern as L5, new
  codec.
- `analyze_unreachable.py` — first appearance of the TIED-OFF / NEEDS /
  REACHABLE? classifier. Walks `coverage.dat`, inspects each uncovered
  signal, and decides whether the signal is unreachable because of config
  (PMP disabled, ICache disabled, debug disabled, etc.) or reachable but
  un-stimulated.

**What happened.**

| configuration                        | episodes | cum toggle |
|--------------------------------------|---------:|-----------:|
| L5 PPO rich (45-op encoder)          |      300 |   56.20%   |
| L6 random   (61-op RVC encoder)      |      150 |   57.48%   |
| L6 PPO rich (61-op RVC encoder)      |      150 |   57.03%   |

- `ibex_compressed_decoder` went from **0% to 97–99% toggle coverage** in
  a single RVC-aware run. The module identified as the L5 bottleneck was
  effectively saturated.
- Cumulative toggle ceiling bumped only ~1.3 percentage points —
  compressed_decoder is 107 toggle points out of ~20,000, so the
  arithmetic upside was small.
- `analyze_unreachable.py` partitioned the remaining ~10,700 uncovered
  points as roughly **6,100 TIED-OFF / 260 NEEDS-stimulus / 5,500
  REACHABLE-but-uncovered**, giving a **reachable ceiling of ~69.6%**
  on this config.
- PPO still did not beat random. The environment was stimulus-bound, not
  algorithm-bound.

**What it taught.**

- Encoder capability was *part* of the L5 plateau but not all of it —
  RVC unblocked a small module, not a big swath.
- The real ceiling (what's reachable given our config) is roughly 70%, not
  56%. Our plateau was at about 80% of the reachable frontier.
- Inside the ~260 NEEDS bucket, the biggest group was the exception path
  (`mepc_q`, `mtval_q`, `mcause_q`, `mstack_*`, controller WAIT_EXC and
  friends). Triggering those requires ECALL/EBREAK plus a return path.
  That became the L7 plan.

**Files.** `level6_rvc/`, plus `STATUS.md` in that directory for the full
detail.


## Level 7 — Stimulus engineering beats algorithm tuning

**Question.** If the ceiling is ~70% and we're at 57%, and PPO ≈ random,
what single intervention closes the most distance?

**What we built.** Four changes, all in the environment, none in the RL
algorithm.

1. **Data-memory prepopulation.** The L5/L6 `MemAgent` returned
   `0x10500073` (WFI) for every unwritten address. Every load therefore
   produced the same constant on `data_rdata_i`. Replaced with
   `addr XOR 0xDEADBEEF` for read-misses. One change. `ibex_load_store_unit`
   toggle rose from **38% to 92%**.
2. **Trap-safe ECALL / EBREAK.** A two-instruction prologue sets
   `mtvec = 0x00200000`. A four-instruction trap handler at that address
   advances `mepc` by 4 and does `MRET`. The agent can now safely emit
   ECALL/EBREAK/illegal-instr — they trap, the handler returns, execution
   continues. This unlocks the entire exception path: `mepc_q`, `mtval_q`,
   `mcause_q`, `mstack_*`, the controller's exception FSM.
3. **29 CSR addresses instead of 5.** The L6 codec rotated through 5
   hand-picked safe CSRs. L7 covers `mcycle`, `minstret`, `mcycleh`,
   `minstreth`, `mcountinhibit`, `mcause`, `mtval`, `mhpmcounter3..8`,
   `mhpmevent3..8`, and the user-level cycle/time/instret. CSR-address
   indexing combines `imm_bucket` and `rs1` so the 5-bucket action space
   still reaches all 29. `ibex_cs_registers` toggle rose from ~10% to 22%
   in a single run.
4. **AUIPC.** One extra opcode. Unlocks `imm_u_type_o` and related ALU
   upper-immediate paths. Small alone, compounds with the rest.

**What happened.**

| configuration                        | episodes | cum toggle |
|--------------------------------------|---------:|-----------:|
| L5 PPO rich (45-op encoder)          |      300 |   56.20%   |
| L6 random   (61-op RVC encoder)      |      150 |   57.48%   |
| **L7 random (64-op, mem/trap/CSRs)** |   **30** |  **66.27%** |

- A **single random 1024-instruction program** on L7 covers **59.72%** of
  toggle bins — higher than any L5/L6 *cumulative* configuration ever
  achieved.
- `analyze_unreachable.py` now classifies more of the exception path as
  reachable, and the reachable ceiling jumps from **~69.6% (L6) to 75.9%
  (L7)** — not because the hardware changed, but because the L7 trap
  scaffolding and wider CSR rotation turn signals that were conservatively
  tagged TIED-OFF into reachable ones.
- We're at **~87% of the 75.9% reachable ceiling** with an *untrained*
  random agent. The remaining 9.6 points need specific instruction
  sequences rather than a broader mix.

**What it taught. The meta-finding.**

On this class of coverage problem, **stimulus engineering moves the number
far more than algorithm tuning**. Random and PPO were bounded by the same
ceiling because the ceiling was set by what the generator could emit and
what the memory model returned, not by the optimiser. When we were tuning
the algorithm knob, we were working on the knob with less travel. Once the
stimulus knob was extended, a random agent in 30 episodes beat the best
previous PPO run at 300 episodes by 10 percentage points.

**Files.** `level7_stimulus/`, plus `STATUS.md` in that directory.


## Where we are, and what's next

The current state:

- **66.27% cumulative toggle** on minimal-config Ibex (30 random episodes,
  L7 environment).
- **75.9% reachable ceiling** for this config (tied-off signals excluded).
- A published-grade **5,615-bin Python shadow** that matches real RTL
  bin-for-bin.
- Infrastructure: cocotb drivers for L5/L7, a Verilator `coverage.dat`
  parser, a tied-off/needs/reachable classifier, per-level random and PPO
  baselines, a memory agent that returns diverse data, a trap handler
  scaffold, and encoders that cover 64 RV32IMC ops including exceptions.

Three ranked next steps (from [`README.md`](README.md) under "What's next"):

1. **Greedy directed stimulus** against the 9.6-point gap (~1 day,
   suggested location `level8_directed/`). The remaining reachable bins
   need specific instruction *sequences* (e.g. `LUI 0xFFFFF` followed by
   `MUL rd, rs, rs` to toggle `imd_val_q_i[0][*]`). A greedy template
   search, rather than uniform random, should close 3–5 points.
2. **Shadow-dense-reward PPO on the L7 env** (~2 hours). With a real
   reachable frontier now visible, PPO finally has something to chase
   that random saturates short of. Either it wins by 2–5 points and RL
   gets back on the map for this problem, or it ties and the stimulus-vs-
   algorithm finding gets a stronger form.
3. **Rebuild in the `opentitan` config** (~1–2 days). Flip `PMPEnable`,
   `ICache`, `DbgTriggerEn`, `WritebackStage`, `SecureIbex`, `RV32B`,
   `BranchPredictor`, `MHPMCounterNum` in `cpu/cocotb_ibex.sv`. Surface
   grows from ~20k toggle points to 50–100k, and our numbers become
   directly comparable to lowRISC's public 88.7% branch / 90% functional
   baseline.

The L7 environment, the shadow, and the analysis tooling all carry over
to each of those three paths.


## If you only read one section

Read Level 7. The findings there — data-memory prepopulation, trap-safe
ECALL/EBREAK, 29-CSR rotation, AUIPC — and the meta-observation that
stimulus engineering beat algorithm tuning by a factor of ten in
coverage-per-episode are the most transferable results in this repo.
