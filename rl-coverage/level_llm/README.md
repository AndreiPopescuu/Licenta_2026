# Level LLM — LLM-based Toggle Coverage

## Context

This folder contains everything needed to continue toggle coverage
of the Ibex RISC-V processor using LLM-generated programs,
starting from the maximum coverage already achieved by RL.

## Starting point

| Metric | Value |
|---|---|
| Baseline coverage | **71.99%** (14,404 / ~20,008 bins) |
| Baseline file | `baseline_hits.pkl` |
| Target bins | **121 bins** in `accessible_bins_for_llm.txt` |
| Potential gain | up to **+0.60%** (72.59% ceiling) |

## Files

```
level_llm/
├── README.md                    ← this file
├── baseline_hits.pkl            ← 14,404 cumulative hits from L8→L9→L10
├── accessible_bins_for_llm.txt  ← 121 uncovered bins proven accessible
└── starter.py                   ← entry point — run this first
```

## How it works

```
accessible_bins_for_llm.txt
        │
        ▼
   LLM generates
   Python actions
        │
        ▼
  starter.py runs
  the program via
  run_program()
        │
        ▼
  check_new_hits()
  compares against
  baseline_hits.pkl
        │
        ▼
  report which of
  the 121 bins got
  covered → feedback
  to LLM
```

## Quick start

```bash
# 1. Make sure the simulation is compiled
cd ../../cpu && make   # only needed once

# 2. Run the starter example
cd ../rl-coverage/level_llm
python starter.py

# 3. Implement your LLM loop in a new file (e.g. llm_agent.py)
#    Use starter.py as the reference for run_program() usage
```

## Key function: run_program()

```python
from starter import run_and_check

# actions = list of (op, rd, rs1, rs2, imm_bucket) tuples
# See Op constants in starter.py
new_bins = run_and_check(actions)
print(f"New bins covered: {len(new_bins)}")
```

## Action format

Each action is a tuple: `(op, rd, rs1, rs2, imm_bucket)`

```python
# Example: lui t1, 0xFFFFF  then  csrw mcountinhibit, t1
actions = [
    (LUI,   1, 0, 0, 4),    # lui t1, 0xFFFFF  (imm_bucket 4 = 0xFFFFF)
    (CSRRW, 0, 1, 0, 22),   # csrw mcountinhibit, t1  (csr bucket 22)
    (ADDI,  1, 0, 0, 0),    # li t1, 0
    (CSRRW, 0, 1, 0, 22),   # csrw mcountinhibit, zero
]
```

## Op constants

See `starter.py` for the full list. Key ones:

| Op | Value | Instruction |
|---|---|---|
| ADDI | 10 | addi rd, rs1, imm |
| LUI | 64 | lui rd, imm |
| AUIPC | 61 | auipc rd, imm |
| CSRRW | 27 | csrw csr, rs1 |
| CSRRS | 28 | csrrs rd, csr, rs1 |
| ECALL | 62 | ecall |
| EBREAK | 63 | ebreak |
| MRET | 70 | mret |
| MUL | 30 | mul rd, rs1, rs2 |
| DIV | 34 | div rd, rs1, rs2 |
| LW | 21 | lw rd, imm(rs1) |
| SW | 26 | sw rs2, imm(rs1) |
| LB | 19 | lb rd, imm(rs1) |
| SB | 24 | sb rs2, imm(rs1) |
| JAL | 44 | jal rd, imm |
| JALR | 65 | jalr rd, rs1, imm |

## IMM_BUCKET values (imm_bucket index → actual value)

```
0 → 0       1 → 1       2 → 4       3 → 8
4 → 0xFFFFF (LUI max)   5 → 0x80000
```
(see codec_l9.py for full list)

## Accessible bins summary

| Group | Bins | How to cover |
|---|---|---|
| mcountinhibit bits | 30 | CSRW 0x320 with various values |
| immediate upper bits | 27 | LUI/AUIPC with large immediates |
| mtvec low bits | 21 | CSRW mtvec with bits [7:1] set |
| CSR read data | 16 | Read CSRs after writing non-zero |
| exception paths | 11 | Illegal instruction / misaligned |
| ALU / decoder | 11 | MUL/DIV, JAL/JALR, illegal CSR |
| data alignment | 3 | LB/SB to odd addresses |
| other | 2 | — |
