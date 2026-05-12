#!/bin/bash

# 1. Environment
module purge
module load 2024
module load Miniconda3/24.7.1-0
module load Mamba/24.9.0-0

# source activate HypeFlow
# export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src/models/hyperbolic_nn_plusplus

conda run -n HypeFlow --no-capture-output \
  env PYTHONPATH="$(pwd):$(pwd)/src/models/hyperbolic_nn_plusplus" \
  python /home/scur0096/cuddly-doodle2/src/train_map.py +model._target_=src.models.FlowMaps.HypeFlowMap.FlowMapModule \
  +model.underlying_loss="esd" \
  +model.in_shape=[1024,3] \
  +data._target_="data.so3_datamodule.SO3DataModule" \
  +data.batch_size=32 \
  +paths.output_dir="/home/scur0096/cuddly-doodle2/logs" \
  +trainer.fast_dev_run=True
# python /home/scur0096/cuddly-doodle2/src/train_map.py +model._target_=src.models.FlowMaps.HypeFlowMap.FlowMapModule     
# +model.underlying_loss="esd"     +model.in_shape=[1024,3]     +data._target_="data.so3_datamodule.SO3DataModule"     +data.batch_size=32     
# +paths.output_dir="/home/scur0096/cuddly-doodle2/logs"     +trainer.fast_dev_run=True