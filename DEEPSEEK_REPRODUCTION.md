# DeepSeek R1 Reproduction Guide

## Overview

This guide documents the reproduction of DeepSeek R1 using the [open-r1](https://github.com/huggingface/open-r1) framework on AMD MI350X infrastructure.

## Infrastructure

| Component | Specification |
|-----------|---------------|
| GPUs | 8× AMD MI350X |
| VRAM per GPU | 288 GB |
| Total VRAM | ~2.3 TB |
| Runtime | ROCm 7.0.0 |
| Base Image | `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` |

## The DeepSeek R1 Training Pipeline

DeepSeek R1 follows a three-phase training approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepSeek R1 Training Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Distillation (SFT)                                       │
│  ─────────────────────────                                         │
│  • Extract reasoning patterns from DeepSeek-R1                     │
│  • Train smaller models on distilled reasoning traces              │
│  • Dataset: Mixture-of-Thoughts (350k samples)                     │
│  • Output: Base model with reasoning capabilities                  │
│                                                                     │
│                          ↓                                         │
│                                                                     │
│  Phase 2: Reinforcement Learning (GRPO)                            │
│  ──────────────────────────────────────                            │
│  • Group Relative Policy Optimization                              │
│  • Rewards: accuracy, format, reasoning structure                  │
│  • Dataset: OpenR1-Math-220k, CodeForces-CoTs                      │
│  • Output: RL-tuned model with improved reasoning                  │
│                                                                     │
│                          ↓                                         │
│                                                                     │
│  Phase 3: Multi-Stage Composition                                  │
│  ────────────────────────────────                                  │
│  • Combine SFT and RL into end-to-end pipeline                     │
│  • Iterative refinement with curriculum learning                   │
│  • Output: Final DeepSeek R1-like model                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## GRPO: Group Relative Policy Optimization

GRPO is the core RL algorithm used instead of PPO. Key differences:

| Aspect | PPO | GRPO |
|--------|-----|------|
| Critic Network | Required (separate model) | Not required |
| Advantage Estimation | Value function baseline | Group comparison |
| Memory Usage | ~2× model size | ~1× model size |
| Samples per Prompt | 1 | 16 (configurable) |

### How GRPO Works

1. **Generate**: Produce N completions per prompt (default: 16)
2. **Evaluate**: Score each completion with reward functions
3. **Rank**: Compare completions within each group
4. **Update**: Use relative ranking for policy gradient

### Reward Functions

```python
# Combined reward calculation
total_reward = (
    accuracy_reward × 1.0 +    # Mathematical correctness
    format_reward × 1.0 +       # <think>/<answer> structure
    tag_count_reward × 1.0      # Proper tag placement
)
```

| Reward Function | Description |
|-----------------|-------------|
| `accuracy_reward` | Verifies answer correctness via LaTeX parsing |
| `format_reward` | Checks `<think>...</think><answer>...</answer>` format |
| `tag_count_reward` | Scores tag placement (0.25 points each) |
| `len_reward` | Encourages concise correct answers |
| `reasoning_steps_reward` | Rewards structured step-by-step reasoning |
| `code_reward` | Executes code against test cases |

## AMD MI350X Configuration

### GPU Allocation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    8× MI350X Layout                         │
├─────────────────────────────────────────────────────────────┤
│  GPU 0-1: vLLM Inference Server (576 GB)                   │
│           └─ Generates completions for GRPO rollouts       │
│           └─ HIP_VISIBLE_DEVICES=0,1                       │
│           └─ tensor-parallel-size=2                        │
├─────────────────────────────────────────────────────────────┤
│  GPU 2-7: GRPO Training (1,728 GB)                         │
│           └─ Policy gradient updates with DeepSpeed        │
│           └─ HIP_VISIBLE_DEVICES=2,3,4,5,6,7               │
│           └─ 6-way data parallelism                        │
└─────────────────────────────────────────────────────────────┘
```

### ROCm Environment Variables

```bash
# Required for optimal performance
export NCCL_MIN_NCHANNELS=112
export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# Optional: Debug and logging
export NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
```

### Known ROCm Issues & Workarounds

| Issue | Symptom | Workaround |
|-------|---------|------------|
| GRPO + vLLM hanging | Training freezes after vLLM init | Run vLLM as separate process |
| CUDA graph capture | Crashes on multi-node | Disable CUDA graphs |
| AITER MoE regression | Slow performance with DeepSeek-R1 | Use vllm 0.11.1 for MoE models |

## Directory Structure

```
/home/asrr/amd_test/
├── CLAUDE.md                      # Claude Code instructions
├── DEEPSEEK_REPRODUCTION.md       # This document
├── open-r1/                       # Cloned open-r1 repository
│   ├── src/open_r1/
│   │   ├── sft.py                 # Supervised fine-tuning
│   │   ├── grpo.py                # GRPO training
│   │   ├── rewards.py             # Reward functions
│   │   └── generate.py            # Data generation
│   ├── recipes/
│   │   ├── OpenR1-Distill-7B/     # 7B model configs
│   │   ├── accelerate_configs/    # DeepSpeed configs
│   │   └── ...
│   └── slurm/                     # Cluster job scripts
├── configs/                       # Custom MI350X configs
│   ├── sft_mi350x.yaml
│   └── grpo_mi350x.yaml
├── scripts/                       # Launch scripts
│   ├── start_vllm.sh
│   ├── train_sft.sh
│   └── train_grpo.sh
└── outputs/                       # Training outputs
    ├── sft-7b/
    └── grpo-7b/
```

## Quick Start

### 1. Start Docker Container

```bash
# Start ROCm Docker container
docker run -d --name open-r1-sft \
  --device /dev/kfd \
  --device /dev/dri \
  --privileged \
  --network=host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NCCL_MIN_NCHANNELS=112 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v /home/asrr/amd_test:/workspace \
  -w /workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  sleep infinity
```

### 2. Install Dependencies (Inside Container)

```bash
docker exec -it open-r1-sft bash

# Install dependencies WITHOUT overwriting PyTorch
pip install --no-deps trl==0.18.0
pip install datasets accelerate deepspeed transformers tokenizers \
    sentencepiece peft einops liger-kernel wandb safetensors huggingface-hub

# Install open-r1
cd /workspace/open-r1
pip install --no-deps -e .
```

### 3. SFT Training (Phase 1)

```bash
# Run inside Docker container
docker exec -d open-r1-sft bash -c "
cd /workspace/open-r1

# Set environment variables
export NCCL_MIN_NCHANNELS=112
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch SFT training
accelerate launch \
    --config_file /workspace/configs/accelerate_zero2_mi350x.yaml \
    --num_processes 8 \
    src/open_r1/sft.py \
    --config /workspace/configs/sft_mi350x.yaml \
    2>&1 | tee /workspace/outputs/sft_training.log
"
```

### 4. Monitor Training

```bash
# Check training progress
docker exec open-r1-sft tail -10 /workspace/outputs/sft_training.log

# Check GPU utilization
rocm-smi --showuse

# Watch live
docker exec -it open-r1-sft tail -f /workspace/outputs/sft_training.log
```

### 5. GRPO Training (Phase 2 - Not Yet Verified)

**Terminal 1: Start vLLM Server (GPUs 0-1)**
```bash
docker exec -d open-r1-sft bash -c "
export HIP_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Math-7B \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --port 8001
"
```

**Terminal 2: Run GRPO Training (GPUs 2-7)**
```bash
docker exec -d open-r1-sft bash -c "
cd /workspace/open-r1
export HIP_VISIBLE_DEVICES=2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=112
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

accelerate launch \
  --config_file /workspace/configs/accelerate_zero2_mi350x.yaml \
  --num_processes 6 \
  src/open_r1/grpo.py \
  --config /workspace/configs/grpo_mi350x.yaml \
  2>&1 | tee /workspace/outputs/grpo_training.log
"
```

## Training Configurations

### Accelerate Config (DeepSpeed ZeRO-2)

```yaml
# configs/accelerate_zero2_mi350x.yaml
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
num_processes: 8
rdzv_backend: static
same_network: true
use_cpu: false
```

### SFT Configuration (MI350X Optimized - Verified Working)

```yaml
# configs/sft_mi350x.yaml
model_name_or_path: Qwen/Qwen2.5-Math-7B
model_revision: main
torch_dtype: bfloat16
attn_implementation: eager  # Use eager for ROCm compatibility

# Dataset
dataset_name: open-r1/Mixture-of-Thoughts
dataset_config: all
dataset_num_proc: 16

# Training - High throughput settings
packing: true                        # CRITICAL: 38x speedup
max_length: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 4.0e-05
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr_rate: 0.1
warmup_ratio: 0.03
num_train_epochs: 1
max_grad_norm: 0.2

# Optimization
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
use_liger_kernel: true
bf16: true

# Output
output_dir: /workspace/outputs/sft-7b
save_strategy: epoch
save_total_limit: 2
report_to: []                        # Disable wandb/tensorboard
push_to_hub: false
```

### GRPO Configuration (MI350X - Not Yet Verified)

```yaml
# configs/grpo_mi350x.yaml
model_name_or_path: Qwen/Qwen2.5-Math-7B
model_revision: main
torch_dtype: bfloat16
attn_implementation: eager  # Use eager for ROCm compatibility

# Dataset
dataset_name: open-r1/OpenR1-Math-220k
dataset_prompt_column: problem

# GRPO specific
num_generations: 16
max_prompt_length: 1024
max_completion_length: 4096

# Rewards
reward_funcs:
  - accuracy
  - format
  - tag_count
reward_weights:
  - 1.0
  - 1.0
  - 1.0

# Training - MI350X optimized
per_device_train_batch_size: 8       # Larger for MI350X
gradient_accumulation_steps: 4
gradient_checkpointing: false        # MI350X has enough VRAM
learning_rate: 2.0e-05
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_train_epochs: 1

# vLLM
use_vllm: true
vllm_server_url: http://localhost:8001/v1

# Output
output_dir: /workspace/outputs/grpo-7b
report_to: []                        # Disable for initial testing
push_to_hub: false
```

## Datasets

| Dataset | Size | Purpose | HuggingFace Link |
|---------|------|---------|------------------|
| Mixture-of-Thoughts | 350k | SFT distillation | `open-r1/Mixture-of-Thoughts` |
| OpenR1-Math-220k | 220k | Math GRPO | `open-r1/OpenR1-Math-220k` |
| CodeForces-CoTs | 100k | Code GRPO | `open-r1/codeforces-cots` |
| NuminaMath-CoT | 860k | Extended math | `AI-MO/NuminaMath-CoT` |

## Training Performance (MI350X Benchmarks)

### SFT Training Speed

| Configuration | Throughput | Per GPU | Step Time |
|---------------|------------|---------|-----------|
| Without packing | 970 tok/s | 121 tok/s | ~540s |
| **With packing** | **37,500 tok/s** | **4,681 tok/s** | **~14s** |

**Improvement: 38x faster with packing enabled**

### Optimal SFT Configuration

```yaml
# High-throughput settings for MI350X
packing: true                        # CRITICAL: 10-20x speedup
max_length: 4096                     # Reduced from 8192
per_device_train_batch_size: 4       # Increased from 2
gradient_accumulation_steps: 4       # Reduced from 8
gradient_checkpointing: true         # Required for long sequences
use_liger_kernel: true               # Additional optimization
```

### Training Time Estimates (SFT)

| Dataset | Examples | Config | Time (8× MI350X) |
|---------|----------|--------|------------------|
| Mixture-of-Thoughts | 349k | packing=false | ~79 days |
| Mixture-of-Thoughts | 494k (packed) | packing=true | **~15 hours** |

### Key Optimization Lessons

1. **`packing: true`** is essential - eliminates padding waste (10-20x speedup)
2. **Reduce `max_length`** if most samples are shorter - attention is O(n²)
3. **Increase batch size** - MI350X has 288GB VRAM, use it
4. **Use Liger kernels** - fused operations for better throughput

## Expected Results

| Model | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench |
|-------|-----------|----------|--------------|---------------|
| DeepSeek-R1-Distill-Qwen-7B | 51.3% | 83.9% | 49.0% | - |
| OpenR1-Distill-7B | 52.7% | 83.3% | 46.7% | - |
| **Target (MI350X)** | ≥52% | ≥83% | ≥45% | - |

## Evaluation

```bash
# Run evaluation on trained model
python -m lighteval accelerate \
  --model_args "pretrained=./outputs/grpo-7b" \
  --tasks "aime_2024,math_500,gpqa_diamond" \
  --output_dir ./eval_results
```

## Monitoring

### Weights & Biases Metrics

Key metrics to track during GRPO training:

- `reward/accuracy` - Mathematical correctness rate
- `reward/format` - Format compliance rate
- `reward/total` - Combined reward (should increase)
- `policy/kl` - KL divergence from reference (should stay bounded)
- `tokens_per_second` - Training throughput

### GPU Monitoring

```bash
# Monitor GPU utilization
watch -n 1 rocm-smi

# Check memory usage
rocm-smi --showmeminfo vram
```

## Troubleshooting

### Training Hangs with vLLM

**Symptom**: Training freezes after "vLLM initialized" message

**Solution**: Run vLLM as a separate process instead of colocated mode

```bash
# Instead of --vllm_mode colocate, use separate server
HIP_VISIBLE_DEVICES=0,1 trl vllm-serve --model Qwen/Qwen2.5-Math-7B &
```

### Out of Memory

**Symptom**: CUDA OOM errors during training

**Solutions**:
1. Reduce `per_device_train_batch_size`
2. Enable `gradient_checkpointing: true`
3. Use ZeRO-3 instead of ZeRO-2
4. Reduce `max_seq_length`

### Slow Training

**Symptom**: Low tokens/second throughput

**Solutions**:
1. Increase batch size (you have 288GB VRAM)
2. Disable gradient checkpointing
3. Use `use_liger_kernel: true`
4. Check NCCL environment variables

## References

- [open-r1 GitHub](https://github.com/huggingface/open-r1)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [AMD GRPO Blog](https://rocm.blogs.amd.com/software-tools-optimization/llm-grpo-rocm/README.html)
- [ROCm vLLM Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
- [TRL Documentation](https://huggingface.co/docs/trl)
