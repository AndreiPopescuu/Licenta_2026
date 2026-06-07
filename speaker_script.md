# Speaker Script

---

## Slide 1 — Title
Good morning. Today I'm presenting my thesis on automated hardware verification using reinforcement learning, with experiments on the Ibex RISC-V processor.

---

## Slide 2 — Hardware Verification
Hardware verification is one of the most expensive parts of chip design. A modern CPU has an enormous number of internal states and you need to make sure every relevant one is exercised before the chip goes to manufacturing — bugs in silicon are permanent and costly. The industry standard is thousands of hand-written tests on commercial simulators. lowRISC verifies Ibex with 1530 manual tests and reaches 88.7% coverage. The question of this thesis is: can an automated agent approach that number using only free tools?

---

## Slide 3 — Types of Coverage
There are several ways to measure how well you've exercised a design. Toggle, branch, line, and expression coverage are structural — Verilator extracts them automatically from the RTL without any manual effort. FSM coverage tracks state machine transitions. Functional coverage is different: you define it yourself, specifying the scenarios that matter for correctness — instruction combinations, data hazards, corner cases. This project uses toggle coverage in Levels 1–7, and moves to rich functional coverage in Level 9.

---

## Slide 4 — RISC-V Instruction Types
RISC-V instructions come in several formats. R-type operates register to register. I-type has an immediate operand and includes loads. S-type is stores. B-type is branches. U-type is upper immediate — LUI and AUIPC. J-type is jumps. CSR instructions read and write control/status registers like the cycle counter or cause register. RVC is the compressed 16-bit variant of the above. Each level of the experiments progressively adds more of these types to the generator, which is directly what drives coverage up.

---

## Slide 5 — Ibex & Tech Stack
Ibex is a real open-source RISC-V processor — about 15,000 lines of SystemVerilog, used in Google's OpenTitan chip. Verilator compiles that RTL into a fast C++ simulator. Cocotb bridges Python to the simulator so we can write programs and feed them to the CPU from Python. RVFI is the formal interface that exposes every instruction retiring from the pipeline. Gymnasium and PPO from Stable-Baselines3 provide the RL framework. The shadow environment is a pure-Python replica of the coverage logic — about 1500 times faster than Verilator, which is what makes RL training practical.

---

## Slide 6 — The Loop
The fundamental loop is: generate a program, simulate it on Ibex, measure coverage, reward the agent for new bins hit, repeat. Against real Verilator each episode takes 2–3 seconds. Against the shadow it takes milliseconds. Levels 1–4 train exclusively on the shadow. Levels 5–9 run against real RTL.

---

## Slide 7 — L1 Chart
The first experiment targets the Ibex instruction decoder — purely combinational, no clock, no state. PPO trained on a Python shadow saturates almost all reachable bins: 2041 of 2107. The remaining 66 are ISA-unreachable. The loop works.

---

## Slide 8 — L2 Chart
Level 2 uses the LLM4DV benchmark — 196 bins on the full CPU. The best published result using Claude 3.5 Sonnet was 5.61%. PPO hits 196 of 196, and the same program run on real Verilator Ibex also hits 196 of 196. Shadow-to-real fidelity is confirmed.

---

## Slide 9 — L3 Chart
Level 3 extends to 1739 bins with chained multi-cycle dependencies. PPO hits 99%+. Random lags for the first time — sequential planning starts to matter.

---

## Slide 10 — L4 Chart
Level 4 builds a comprehensive 5615-bin shadow for all of RV32I/M. PPO saturates it in about 2 minutes; random takes 53 minutes. That's the 100× speedup. Shadow-to-real validation shows zero divergence: 425 of 425 bins match on one agent, 1499 of 1499 on another.

---

## Slide 11 — L5 Chart
First run on real Verilator toggle coverage. Surprising result: all three PPO variants plateau at exactly 56.2% — same as random, even at 300 episodes. The per-module breakdown explains why: compressed decoder at 0%, load/store unit at 38%, CSR registers at 10%. The ceiling was the encoder, not the algorithm.

---

## Slide 12 — L6 Chart
Adding 16 RVC compressed instructions fixes the compressed decoder — it jumps from 0% to 97% in one run. But the cumulative gain is only 1.3 percentage points because that module is small. PPO still equals random. The unreachability analysis reveals the real gap: the exception path requires ECALL/EBREAK and a trap handler.

---

## Slide 13 — L7 Chart
This is the main finding. Four changes to the environment, none to the algorithm: memory prepopulated with a diverse pattern instead of a constant — load/store unit 38% to 92%. A trap handler so ECALL and EBREAK return safely — entire exception path unlocked. 29 CSR addresses instead of 5. AUIPC added. Result: random in 30 episodes hits 66.27% — 10 points above the best PPO at 300 episodes on the old environment. PPO on the same L7 environment reaches 67.58% at 300 episodes — a small advantage over random, both hitting the same ceiling. The ceiling is set by what you can generate, not by the optimizer.

---

## Slide 14 — L9
Level 9 shifts the target to rich functional coverage — about 15,000 bins covering instruction sequences, operand pairs, RAW hazards, and corner cases. A full RISC-V ISA simulator was built in Python with correct control flow. A critical bug was found and fixed: the initial version executed instructions sequentially ignoring branches, reporting 3× more coverage than Verilator. After the fix the shadow-to-Verilator ratio stabilised at about 0.84. The constrained random baseline is 45.10%. PPO training is ongoing.

---

## Slide 15 — Conclusions
The main contributions: a 5615-bin Python shadow with zero RTL divergence, a characterised reachable ceiling of 75.9%, and the core finding that stimulus engineering dominates algorithm tuning by roughly a factor of 10 on this class of problem. The L7 PPO experiment confirms that once the stimulus space is rich enough, RL holds a small advantage over random but both are bounded by the same ceiling.

---

## Slide 16 — Q&A
Thank you. Happy to take questions.
