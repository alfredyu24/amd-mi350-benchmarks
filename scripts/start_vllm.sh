#!/bin/bash
# Start vLLM inference server for GRPO training
# Uses GPUs 0-1 for inference generation

set -e

# Configuration
MODEL=${1:-"Qwen/Qwen2.5-Math-7B"}
PORT=${2:-8001}
TP_SIZE=${3:-2}
MAX_MODEL_LEN=${4:-8192}

echo "=========================================="
echo "Starting vLLM Server for GRPO Training"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPUs: 0,1"
echo "=========================================="

# ROCm environment variables
export NCCL_MIN_NCHANNELS=112
export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# Restrict to GPUs 0-1
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16

# Alternative: Use TRL's vllm-serve
# trl vllm-serve --model "$MODEL" --port "$PORT"
