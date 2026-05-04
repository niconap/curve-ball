#! /bin/bash

source /opt/conda/bin/activate
conda activate hypeflow
export PYTHONPATH="$(pwd):$(pwd)/src/models/hyperbolic_nn_plusplus:$PYTHONPATH"

EXPERIMENT=${1:-comm20_hyp}
DATASET=${2:-comm20}
CHECKPOINT_PATH=${3}

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Usage: $0 <experiment> <dataset> <checkpoint_path>"
  echo "Example: $0 comm20_hyp comm20 /path/to/last.ckpt"
  exit 1
fi

python src/evaluate_vqvae_test_metrics.py \
  +experiment="${EXPERIMENT}" \
  dataset="${DATASET}" \
  loss=VQVAE \
  general.test_only="${CHECKPOINT_PATH}" \
  general.wandb=disabled
