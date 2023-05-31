#!/bin/bash
set -x
# Setup the output directory
rm -rf timm_logs
mkdir timm_logs

# Commands for timm_models for device=cuda, dtype=float32 for training and for performance testing
python benchmarks/dynamo/timm_models.py --performance --float32 -dcuda --output=timm_logs/inductor_timm_models_float32_training_cuda_performance.csv --training --inductor  --output-directory=./ --no-skip --dashboard -x gluon_xception65 -x levit_128 -x selecsls42b --cold-start-latency

# Commands for timm_models for device=cuda, dtype=float32 for training and for accuracy testing
python benchmarks/dynamo/timm_models.py --accuracy --float32 -dcuda --output=timm_logs/inductor_timm_models_float32_training_cuda_accuracy.csv --training --inductor  --output-directory=./ --no-skip --dashboard -x gluon_xception65 -x levit_128 -x selecsls42b

