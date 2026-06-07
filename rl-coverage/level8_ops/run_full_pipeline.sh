#!/bin/bash
# Antrenează L8, construiește corpus, rulează regression test.
set -e

EPISODES=${1:-1200}
STEPS=${2:-256}
EXPECTED=${3:-69.0}

cd "$(dirname "$0")"

echo "=== [1/3] Antrenare L8 PPO ($EPISODES episoade, $STEPS pași/ep) ==="
python3 train_l8_v3_ppo.py --episodes "$EPISODES" --steps "$STEPS"

echo ""
echo "=== [2/3] Construire corpus ==="
python3 build_corpus.py --episodes 300 --model l8_v3_ppo_model.zip --out corpus_l8.json

echo ""
echo "=== [3/3] Regression test (prag: ${EXPECTED}%) ==="
python3 replay_corpus.py --corpus corpus_l8.json --expected "$EXPECTED"
