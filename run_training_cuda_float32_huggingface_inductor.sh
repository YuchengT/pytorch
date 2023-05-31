#!/bin/bash
set -x
# Setup the output directory
rm -rf huggingface_logs
mkdir huggingface_logs

# Commands for huggingface for device=cuda, dtype=float32 for training and for performance testing
python benchmarks/dynamo/huggingface.py --performance --float32 -dcuda --output=huggingface_logs/inductor_huggingface_float32_training_cuda_performance.csv --training --inductor   --no-skip --dashboard -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForCausalLM -x Reformer -x GPTNeoForSequenceClassification --find-all-batch-sizes --cold-start-latency

# Commands for huggingface for device=cuda, dtype=float32 for training and for accuracy testing
python benchmarks/dynamo/huggingface.py --accuracy --float32 -dcuda --output=huggingface_logs/inductor_huggingface_float32_training_cuda_accuracy.csv --training --inductor   --no-skip --dashboard -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForCausalLM -x Reformer -x GPTNeoForSequenceClassification --find-all-batch-sizes

