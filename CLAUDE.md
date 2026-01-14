# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository documents and tests DeepSeek R1 running on AMD MI350X GPU infrastructure via Docker and vLLM.

## Environment

- **GPU:** 8x AMD MI350X (288 GB VRAM each, ~2.3 TB total)
- **Runtime:** ROCm 7.0.0
- **Inference Server:** vLLM 0.11.2
- **Model:** deepseek-ai/DeepSeek-R1

## vLLM Docker Configuration

### Docker Run Command
```bash
docker run -d \
  --name deepseek-r1 \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  -e NCCL_MIN_NCHANNELS=112 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --block-size 1 \
    --port 8000
```

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--tensor-parallel-size` | 8 | Distribute model across 8 GPUs |
| `--max-model-len` | 8192 | Maximum context length |
| `--block-size` | 1 | KV cache block size |
| `--port` | 8000 | API server port |

### ROCm Environment Variables
- `VLLM_ROCM_USE_AITER=1` - Enable AMD AIter optimizations
- `NCCL_MIN_NCHANNELS=112` - NCCL communication channels
- `HIP_FORCE_DEV_KERNARG=1` - HIP kernel optimization
- `TORCH_BLAS_PREFER_HIPBLASLT=1` - Use hipBLASLt for better performance

## Key Commands

### Check DeepSeek R1 Status
```bash
docker ps | grep deepseek
docker logs deepseek-r1 --tail 50
```

### Test API
```bash
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-R1", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'
```

### Monitor Throughput
```bash
docker logs deepseek-r1 2>&1 | grep "throughput" | tail -5
```

## API Endpoint

The vLLM server exposes an OpenAI-compatible API at `http://localhost:8000/v1/`.

## Command-Line Chat

Interactive chat without web UI:
```bash
python3 chat.py
```

Commands within chat:
- `quit` - exit the chat
- `clear` - reset conversation history
- `Ctrl+C` - exit

Single question one-liner:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-R1","messages":[{"role":"user","content":"Your question"}],"max_tokens":100}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## Training Benchmarks

### Performance Summary

| Benchmark | Result |
|-----------|--------|
| Matrix Multiply (FP16) | **1,022 TFLOPS** |
| Inference (DeepSeek R1) | **~70 tokens/sec** |
| Single GPU Training (7B) | **578 TFLOPS** |
| Multi-GPU Training (7B × 8) | **3,182 TFLOPS** |

### Quick Benchmark Commands

```bash
# Simple training benchmark (matrix multiply + MLP)
docker exec deepseek-r1 python3 /tmp/simple_train_benchmark.py

# Transformer training benchmark (GPT-style)
docker exec deepseek-r1 python3 /tmp/transformer_benchmark.py

# Full transformer benchmark suite (all model sizes)
docker exec deepseek-r1 python3 /tmp/transformer_benchmark.py --full

# Large models benchmark (1.5B+) - requires stopping DeepSeek first
docker stop deepseek-r1
docker run --rm --device /dev/kfd --device /dev/dri \
  -v /home/asrr/amd_test:/workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  python3 /workspace/transformer_benchmark.py --model 7b --batch-size 8 --seq-len 1024
docker start deepseek-r1
```

### Transformer Training Results (Small Models)

| Model | Params | Tokens/sec | TFLOPS | Memory |
|-------|--------|------------|--------|--------|
| Small | 30M | 950,586 | 171.8 | 9.7 GB |
| Medium | 124M | 298,957 | 222.4 | 9.8 GB |
| Large | 354M | 107,626 | 228.7 | 11.3 GB |
| XL | 773M | 40,054 | 185.8 | 12.6 GB |

### Large Model Training Results (1.5B - 13B)

| Model | Params | Batch | Seq Len | Tokens/sec | TFLOPS | Memory |
|-------|--------|-------|---------|------------|--------|--------|
| 1.5B | 1.44B | 4 | 512 | 24,195 | 208.6 | 20.7 GB |
| 3B | 2.65B | 2 | 512 | 11,980 | 190.3 | 26.9 GB |
| 7B | 5.24B | 8 | 1024 | 18,381 | **578.3** | 107.9 GB |
| 13B | 10.12B | 4 | 1024 | 7,941 | 482.2 | 119.7 GB |

### Benchmark Scripts

| Script | Description |
|--------|-------------|
| `simple_train_benchmark.py` | Matrix multiply + simple MLP training |
| `transformer_benchmark.py` | GPT-style transformer with attention (30M - 13B) |

### Benchmark Options

```bash
# Available model sizes
--model {small,medium,large,xl,1.5b,3b,7b,13b}

# Custom batch size and sequence length
--batch-size 8 --seq-len 1024

# Run all small models
--full

# Run large models only (1.5B+)
--large-models
```

### Multi-GPU Training (DDP)

| Config | Tokens/sec | Total TFLOPS | Memory/GPU |
|--------|------------|--------------|------------|
| 7B × 8 GPUs | 101,124 | **3,182** | 92 GB |
| 13B × 8 GPUs | 50,595 | **3,072** | 157 GB |

```bash
# Multi-GPU benchmark (requires stopping DeepSeek first)
docker stop deepseek-r1
docker run --rm --device /dev/kfd --device /dev/dri --ipc=host \
  -v /home/asrr/amd_test:/workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  python3 /workspace/multigpu_benchmark.py --model 7b --gpus 8 --batch-size 8

# Scaling test (1,2,4,8 GPUs)
python3 /workspace/multigpu_benchmark.py --model 7b --scaling-test
```

### MAD Framework (Optional)

ROCm/MAD for official AMD training benchmarks:

```bash
cd MAD
pip install -r requirements.txt
export HF_TOKEN=your_hugging_face_token

# Discover available models
madengine discover --tags training

# Run Llama 3.1 8B training
madengine run --tags pyt_train_llama-3.1-8b --live-output --timeout 3600
```

### Training Modes (MAD)
- `pretrain` - Pre-training from scratch
- `finetune_lora` - LoRA fine-tuning
- `finetune_fw` - Full weight fine-tuning
