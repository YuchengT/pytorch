#!/bin/bash
set -x
# Setup the output directory
rm -rf torchbench_logs
mkdir torchbench_logs

# Commands for torchbench for device=cuda, dtype=float32 for training and for performance testing
python benchmarks/dynamo/torchbench.py --performance --float32 -dcuda --output=torchbench_logs/inductor_torchbench_float32_training_cuda_performance.csv --training --inductor  --output-directory=./ --no-skip --dashboard -x detectron2_fasterrcnn_r_50_dc5 -x pyhpc_turbulent_kinetic_energy -x detectron2_maskrcnn_r_101_c4 -x detectron2_maskrcnn_r_50_fpn -x detectron2_fasterrcnn_r_101_c4 -x maml -x fambench_xlmr -x detectron2_maskrcnn_r_101_fpn -x pyhpc_isoneutral_mixing -x pyhpc_equation_of_state -x detectron2_maskrcnn -x detectron2_fasterrcnn_r_101_dc5 -x opacus_cifar10 -x detectron2_fasterrcnn_r_50_fpn -x detectron2_fasterrcnn_r_50_c4 -x detectron2_fasterrcnn_r_101_fpn --cold-start-latency

# Commands for torchbench for device=cuda, dtype=float32 for training and for accuracy testing
python benchmarks/dynamo/torchbench.py --accuracy --float32 -dcuda --output=torchbench_logs/inductor_torchbench_float32_training_cuda_accuracy.csv --training --inductor  --output-directory=./ --no-skip --dashboard -x detectron2_fasterrcnn_r_50_dc5 -x pyhpc_turbulent_kinetic_energy -x detectron2_maskrcnn_r_101_c4 -x detectron2_maskrcnn_r_50_fpn -x detectron2_fasterrcnn_r_101_c4 -x maml -x fambench_xlmr -x detectron2_maskrcnn_r_101_fpn -x pyhpc_isoneutral_mixing -x pyhpc_equation_of_state -x detectron2_maskrcnn -x detectron2_fasterrcnn_r_101_dc5 -x opacus_cifar10 -x detectron2_fasterrcnn_r_50_fpn -x detectron2_fasterrcnn_r_50_c4 -x detectron2_fasterrcnn_r_101_fpn

