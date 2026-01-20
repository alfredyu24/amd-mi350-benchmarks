#!/bin/bash
# SFT Training Script for MI350X
# Uses all 8 GPUs for supervised fine-tuning

set -e

# Configuration
CONFIG=${1:-"/home/asrr/amd_test/configs/sft_mi350x.yaml"}
NUM_GPUS=${2:-8}

echo "=========================================="
echo "Starting SFT Training on MI350X"
echo "=========================================="
echo "Config: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Change to open-r1 directory
cd /home/asrr/amd_test/open-r1

# ROCm environment variables
export NCCL_MIN_NCHANNELS=112
export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Logging
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

# Use all GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch training
accelerate launch \
    --config_file /home/asrr/amd_test/configs/accelerate_zero2_mi350x.yaml \
    --num_processes "$NUM_GPUS" \
    src/open_r1/sft.py \
    --config "$CONFIG"

echo "=========================================="
echo "SFT Training Complete!"
echo "=========================================="
