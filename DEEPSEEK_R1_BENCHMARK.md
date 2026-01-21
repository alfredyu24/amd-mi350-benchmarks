# DeepSeek R1 Benchmark Results - MI350X

**Date:** 2025-01-20
**Hardware:** 8x AMD MI350X (288 GB VRAM each)
**Software:** ROCm 7.0.0, vLLM 0.11.2

## Docker Command

```bash
docker run -d \
  --name deepseek-r1 \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --shm-size 64G \
  -v /home/asrr/models/hub/:/root/.cache/huggingface/hub \
  -e NCCL_MIN_NCHANNELS=112 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MHA=1 \
  -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=1 \
  -e VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
  -e AMDGCN_USE_BUFFER_OPS=1 \
  -e VLLM_USE_AITER_TRITON_ROPE=1 \
  -e VLLM_USE_AITER_TRITON_SILU_MUL=0 \
  -e VLLM_USE_V1=1 \
  -e VLLM_USE_TRITON_FLASH_ATTN=1 \
  -e USE_FASTSAFETENSOR=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  vllm serve deepseek-ai/DeepSeek-R1 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 16384 \
    --block-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 65536 \
    --distributed-executor-backend mp \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --async-scheduling \
    --dtype auto
```

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Model | deepseek-ai/DeepSeek-R1 (671B, FP8) |
| Tensor Parallel Size | 8 |
| Max Model Length | 16,384 |
| GPU Memory Utilization | 95% |
| Max Batched Tokens | 65,536 |
| Max Sequences | 256 |
| Block Size | 1 |
| Distributed Backend | mp (multiprocessing) |
| Prefix Caching | Disabled |
| Async Scheduling | Enabled |
| vLLM Version | V1 |

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| VLLM_USE_V1 | 1 | Enable vLLM V1 engine |
| VLLM_ROCM_USE_AITER | 1 | Enable AITER optimizations |
| VLLM_ROCM_USE_AITER_MHA | 1 | AITER multi-head attention |
| VLLM_ROCM_USE_AITER_PAGED_ATTN | 0 | Disable AITER paged attention |
| VLLM_ROCM_USE_AITER_RMSNORM | 1 | AITER RMSNorm |
| VLLM_V1_USE_PREFILL_DECODE_ATTENTION | 1 | Prefill/decode attention optimization |
| VLLM_USE_TRITON_FLASH_ATTN | 1 | Triton flash attention |
| VLLM_USE_AITER_TRITON_ROPE | 1 | AITER Triton RoPE |
| VLLM_USE_AITER_TRITON_SILU_MUL | 0 | Disable AITER Triton SiLU multiply |
| AMDGCN_USE_BUFFER_OPS | 1 | AMD buffer operations |
| USE_FASTSAFETENSOR | 1 | Fast safetensor loading |
| SAFETENSORS_FAST_GPU | 1 | GPU-accelerated safetensor loading |
| HIP_FORCE_DEV_KERNARG | 1 | HIP kernel optimization |
| TORCH_BLAS_PREFER_HIPBLASLT | 1 | Prefer hipBLASLt |
| NCCL_MIN_NCHANNELS | 112 | NCCL communication channels |

## Model Loading

| Metric | Value |
|--------|-------|
| Memory Used | 81.7 GiB |
| Loading Time | 394.9 seconds |
| Safetensor Shards | 163 files |
| VRAM per GPU | ~295 GB (95% of 309 GB) |

## Throughput Results

### Concurrent Request Benchmark

| Concurrency | Total Tokens | Time (s) | Throughput (tok/s) |
|-------------|--------------|----------|-------------------|
| 1 | 512 | 23.28 | 22.00 |
| 4 | 2,048 | 16.94 | 120.88 |
| 8 | 4,096 | 10.60 | 386.49 |
| 16 | 8,192 | 13.03 | 628.64 |
| 32 | 8,192 | 8.26 | 991.47 |
| 64 | 16,384 | 9.26 | **1,770.06** |
| 128 | 32,768 | 32.00 | 1,023.99 |
| 256 | 17,510 | 6.59 | **2,655.89** |

### Peak Performance

| Metric | Value |
|--------|-------|
| **Peak Throughput** | **2,656 tokens/sec** |
| Optimal Concurrency | 64-256 requests |
| vLLM Reported Peak | 2,339 tokens/sec |
| Per GPU Throughput | ~332 tokens/sec |

### Single Request Performance

| Metric | Value |
|--------|-------|
| Latency (1024 tokens) | ~22 seconds |
| Single Request Throughput | ~46 tokens/sec |

## Comparison with H200

| Hardware | Backend | Throughput | Source |
|----------|---------|------------|--------|
| **MI350X x8** | **vLLM V1** | **2,656 tok/s** | This benchmark |
| H200 x8 | vLLM | 2,200 tok/s/GPU | vLLM Blog |
| H200 x8 | TensorRT-LLM | 4,176 tok/s | dstack |
| H200 x8 | SGLang | 6,311 tok/s | dstack |
| H200 (HGX) | NIM | 3,872 tok/s | NVIDIA Blog |
| MI300X x8 | vLLM | 4,574 tok/s | dstack |

## Potential Optimizations

1. **Disable NUMA balancing** (warnings in logs):
   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
   ```

2. **Try SGLang** - showed better results on AMD hardware in benchmarks

3. **Enable prefix caching** - currently disabled

4. **Increase batch sizes** - dstack benchmarks used up to 1,024

5. **Try different AITER settings**:
   - `VLLM_ROCM_USE_AITER_PAGED_ATTN=1`
   - `VLLM_USE_AITER_TRITON_SILU_MUL=1`

## Notes

- Model uses FP8 quantization automatically
- Using Triton backend for FP8 MoE
- AITER MLA backend enabled
- Chunked prefill enabled with 65,536 max batched tokens
- KV cache usage remained low (~0.4% max) during benchmarks
