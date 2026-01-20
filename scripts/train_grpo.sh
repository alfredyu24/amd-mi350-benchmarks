#!/bin/bash
# GRPO Training Script for MI350X
# Uses GPUs 2-7 for training (GPUs 0-1 reserved for vLLM)

set -e

# Configuration
CONFIG=${1:-"/home/asrr/amd_test/configs/grpo_mi350x.yaml"}
NUM_GPUS=${2:-6}
VLLM_URL=${3:-"http://localhost:8001/v1"}

echo "=========================================="
echo "Starting GRPO Training on MI350X"
echo "=========================================="
echo "Config: $CONFIG"
echo "Training GPUs: 2-7 ($NUM_GPUS GPUs)"
echo "vLLM Server: $VLLM_URL"
echo "=========================================="

# Check if vLLM server is running
echo "Checking vLLM server..."
if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
    echo "vLLM server is running!"
else
    echo "ERROR: vLLM server not responding at $VLLM_URL"
    echo "Please start vLLM first with: ./scripts/start_vllm.sh"
    exit 1
fi

# Change to open-r1 directory
cd /home/asrr/amd_test/open-r1

# ROCm environment variables
export NCCL_MIN_NCHANNELS=112
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Logging
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

# Use GPUs 2-7 for training (0-1 used by vLLM)
export HIP_VISIBLE_DEVICES=2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Create modified accelerate config for 6 GPUs
cat > /tmp/accelerate_grpo.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Launch training
accelerate launch \
    --config_file /tmp/accelerate_grpo.yaml \
    --num_processes "$NUM_GPUS" \
    src/open_r1/grpo.py \
    --config "$CONFIG" \
    --vllm_server_url "$VLLM_URL"

echo "=========================================="
echo "GRPO Training Complete!"
echo "=========================================="
