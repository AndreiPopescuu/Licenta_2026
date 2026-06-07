#!/bin/bash
set -e

PYTHON=/home/andrei/ibex_env/bin/python3
ROOT=/home/andrei/IBEX/ibex-coverage-experiments/rl-coverage

echo "=== STEP 1: L8 V3 pana la saturatie ==="
$PYTHON $ROOT/level8_ops/train_l8_v3_ppo.py \
  --episodes 2000 --saturation 50 \
  --save-hits $ROOT/level8_ops/l8_final_hits.pkl \
  --out $ROOT/level8_ops/l8_v3_pipeline_curve.npz

echo "=== STEP 2: L9 V2 de la hits L8 ==="
$PYTHON $ROOT/level9_ops/train_l9_v2_ppo.py \
  --hits $ROOT/level8_ops/l8_final_hits.pkl \
  --episodes 5000 --steps 512 --ent-coef 0.15 \
  --out $ROOT/level9_ops/l9_v2_pipeline_curve.npz

echo "=== STEP 3: L10 de la hits L9 ==="
$PYTHON $ROOT/level10_ops/train_l10_ppo.py \
  --hits $ROOT/level9_ops/l9_v2_checkpoint_hits.pkl \
  --episodes 3600 --steps 512 --ent-coef 0.15 \
  --out $ROOT/level10_ops/l10_ppo_curve.npz

echo "=== PIPELINE DONE ==="
