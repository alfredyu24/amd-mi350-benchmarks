# AMD MI350X Training Benchmark Results

## System Configuration

| Property | Value |
|----------|-------|
| GPU | AMD Instinct MI350X |
| Number of GPUs | 8 |
| Memory per GPU | 309 GB |
| Total VRAM | ~2.5 TB |
| ROCm Version | 7.0.0 |
| PyTorch Version | 2.9.0 |

---

## Benchmark 1: Matrix Multiplication (FP16)

Raw GPU compute performance measuring TFLOPS.

| Matrix Size | Time (20 iter) | Throughput | Efficiency |
|-------------|----------------|------------|------------|
| 4096x4096 | 0.004s | 775 TFLOPS | ~60% |
| 8192x8192 | 0.022s | 1,019 TFLOPS | ~78% |
| 16384x16384 | 0.172s | 1,022 TFLOPS | ~78% |

**Peak Performance: ~1,022 TFLOPS (FP16)**

### Comparison with Other GPUs

| GPU | FP16 TFLOPS |
|-----|-------------|
| NVIDIA A100 | ~312 |
| NVIDIA H100 | ~990 |
| **AMD MI350X** | **~1,022** |

---

## Benchmark 1b: GEMM Performance (SemiAnalysis Methodology)

Comprehensive GEMM benchmark based on [SemiAnalysis MI300X vs H100/H200 methodology](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training), using real-world Llama 70B training shapes.

### Methodology

- **Timing**: OpenAI do_bench style (warmup=30, rep=200, median selection)
- **Initialization**: Random normal distribution (mean=0, variance=1)
- **Shapes**: Production Llama 70B training workloads

### FP16 Results

| Shape | Description | Time (ms) | TFLOPS |
|-------|-------------|-----------|--------|
| (8192, 8192, 8192) | Standard Square (8K) | 1.044 | **1,053** |
| (16384, 8192, 7168) | FFN GEMM 1 | 1.852 | 1,039 |
| (16384, 16384, 16384) | Large Square (16K) | 8.624 | 1,020 |
| (16384, 1024, 8192) | Attention Output | 0.280 | 981 |
| (16384, 3584, 8192) | FFN GEMM 2 | 0.983 | 979 |
| (4096, 4096, 4096) | Standard Square (4K) | 0.146 | 940 |
| (16384, 8192, 1280) | Fused QKV Projection | 0.398 | 863 |

**FP16 Average: 982 TFLOPS | Peak: 1,053 TFLOPS**

### BF16 Results

| Shape | Description | Time (ms) | TFLOPS |
|-------|-------------|-----------|--------|
| (8192, 8192, 8192) | Standard Square (8K) | 0.996 | **1,104** |
| (16384, 8192, 7168) | FFN GEMM 1 | 1.757 | 1,095 |
| (16384, 16384, 16384) | Large Square (16K) | 8.309 | 1,059 |
| (16384, 3584, 8192) | FFN GEMM 2 | 0.927 | 1,038 |
| (16384, 1024, 8192) | Attention Output | 0.269 | 1,023 |
| (4096, 4096, 4096) | Standard Square (4K) | 0.142 | 969 |
| (16384, 8192, 1280) | Fused QKV Projection | 0.381 | 901 |

**BF16 Average: 1,027 TFLOPS | Peak: 1,104 TFLOPS**

### Comparison with SemiAnalysis Results

| GPU | BF16 TFLOPS | FP16 TFLOPS | Notes |
|-----|-------------|-------------|-------|
| **AMD MI350X** | **1,104** | **1,053** | This benchmark |
| AMD MI300X | ~620 | - | SemiAnalysis |
| NVIDIA H100 | ~720 | - | SemiAnalysis |
| NVIDIA H200 | ~720 | - | SemiAnalysis |

**MI350X achieves ~78% higher GEMM throughput than MI300X and ~53% higher than H100/H200.**

### Library Comparison (torch.matmul vs F.linear)

| Shape | matmul (hipBLASLt) | F.linear (rocBLAS) | Difference |
|-------|--------------------|--------------------|------------|
| (8192, 8192, 8192) | 1,046 | 1,206 | -13.3% |
| (16384, 8192, 1280) | 863 | 970 | -11.0% |
| (4096, 4096, 4096) | 939 | 1,053 | -10.8% |

Note: F.linear (rocBLAS) shows ~11-13% higher performance than torch.matmul in this environment, opposite to SemiAnalysis findings on MI300X, suggesting AMD has improved rocBLAS optimization.

### GEMM Benchmark Command

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -v /home/asrr/amd_test:/workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  python3 /workspace/gemm_benchmark.py
```

---

## Benchmark 2: Simple Training Loop

Simple 3-layer neural network training (134M parameters).

### Model Architecture

```
Input [256, 4096]
    ↓
Linear(4096 → 8192) + ReLU    # 33M params
    ↓
Linear(8192 → 8192) + ReLU    # 67M params
    ↓
Linear(8192 → 4096)           # 33M params
    ↓
Output [256, 4096]
```

### Results

| Metric | Value |
|--------|-------|
| Batch Size | 256 |
| Iterations | 100 |
| Total Time | 0.206s |
| **Throughput** | **124,517 samples/sec** |
| **Time per Step** | **2.06 ms** |

---

## Benchmark 3: Transformer Training (GPT-style)

Full transformer training with multi-head attention, feed-forward networks, and causal masking.

### Model Configurations

| Model | Layers | Heads | Dim | FF Dim | Parameters |
|-------|--------|-------|-----|--------|------------|
| Small | 6 | 6 | 384 | 1536 | 30M |
| Medium | 12 | 12 | 768 | 3072 | 124M |
| Large | 24 | 16 | 1024 | 4096 | 354M |
| XL | 36 | 20 | 1280 | 5120 | 773M |
| 1.5B | 48 | 24 | 1536 | 6144 | 1.44B |
| 3B | 32 | 32 | 2560 | 10240 | 2.65B |
| 7B | 32 | 32 | 4096 | 11008 | 5.24B |
| 13B | 40 | 40 | 5120 | 13824 | 10.12B |

### Results Summary (Small to XL)

| Model | Params | Batch | Seq Len | Tokens/sec | TFLOPS | Memory | ms/step |
|-------|--------|-------|---------|------------|--------|--------|---------|
| Small | 30M | 32 | 512 | **950,586** | 171.8 | 9.7 GB | 17.2 |
| Medium | 124M | 16 | 512 | **298,957** | 222.4 | 9.8 GB | 27.4 |
| Large | 354M | 8 | 512 | **107,626** | 228.7 | 11.3 GB | 38.1 |
| XL | 773M | 4 | 512 | **40,054** | 185.8 | 12.6 GB | 51.1 |

### Large Model Results (1.5B to 13B)

| Model | Params | Batch | Seq Len | Tokens/sec | TFLOPS | Memory | ms/step |
|-------|--------|-------|---------|------------|--------|--------|---------|
| 1.5B | 1.44B | 4 | 512 | **24,195** | 208.6 | 20.7 GB | 84.7 |
| 3B | 2.65B | 2 | 512 | **11,980** | 190.3 | 26.9 GB | 85.5 |
| 7B | 5.24B | 8 | 1024 | **18,381** | **578.3** | 107.9 GB | 445.7 |
| 13B | 10.12B | 4 | 1024 | **7,941** | 482.2 | 119.7 GB | 515.8 |

### Key Observations

1. **Peak TFLOPS**: **578 TFLOPS** on 7B model with batch size 8
2. **Best Throughput**: 950K tokens/sec on Small model
3. **Large Model Efficiency**: 7B and 13B models achieve 480-580 TFLOPS
4. **Memory Capacity**: 13B model uses only 120 GB of 288 GB available per GPU
5. **Scaling**: Larger batch sizes significantly improve TFLOPS utilization

### Transformer Architecture Details

```
┌─────────────────────────────────────────┐
│          Input Tokens [B, T]            │
├─────────────────────────────────────────┤
│  Token Embedding + Position Embedding   │
├─────────────────────────────────────────┤
│                                         │
│   ┌─────────────────────────────────┐   │
│   │    Transformer Block (×N)       │   │
│   │  ┌───────────────────────────┐  │   │
│   │  │ Layer Norm                │  │   │
│   │  │ Multi-Head Self-Attention │  │   │
│   │  │ + Residual Connection     │  │   │
│   │  ├───────────────────────────┤  │   │
│   │  │ Layer Norm                │  │   │
│   │  │ Feed-Forward Network      │  │   │
│   │  │ + Residual Connection     │  │   │
│   │  └───────────────────────────┘  │   │
│   └─────────────────────────────────┘   │
│                                         │
├─────────────────────────────────────────┤
│            Final Layer Norm             │
├─────────────────────────────────────────┤
│         Output Logits [B, T, V]         │
└─────────────────────────────────────────┘
```

---

## Benchmark Scripts

### Simple Benchmark
```bash
docker exec deepseek-r1 python3 /tmp/simple_train_benchmark.py
```

### Transformer Benchmark (Single)
```bash
docker exec deepseek-r1 python3 /tmp/transformer_benchmark.py
```

### Transformer Benchmark (Full Suite)
```bash
docker exec deepseek-r1 python3 /tmp/transformer_benchmark.py --full
```

### Script Locations

- `/home/asrr/amd_test/simple_train_benchmark.py`
- `/home/asrr/amd_test/transformer_benchmark.py`

---

---

## Benchmark 4: Multi-GPU Training (DDP)

Data-parallel training across multiple MI350X GPUs using PyTorch DistributedDataParallel.

### 8-GPU Results

| Model | Params | Batch/GPU | Global Batch | Tokens/sec | Total TFLOPS | TFLOPS/GPU | Memory/GPU |
|-------|--------|-----------|--------------|------------|--------------|------------|------------|
| 7B | 5.24B | 8 | 64 | **101,124** | **3,182** | 398 | 92 GB |
| 13B | 10.12B | 8 | 64 | **50,595** | **3,072** | 384 | 157 GB |

### Scaling Efficiency (7B Model)

| GPUs | Tokens/sec | Total TFLOPS | Scaling Efficiency |
|------|------------|--------------|-------------------|
| 1 | 18,324 | 577 | 100% (baseline) |
| 2 | 28,451 | 895 | 77.6% |
| 4 | 62,813 | 1,976 | 85.7% |
| 8 | 70,763 | 2,226 | 48.3% |

### Key Observations

1. **Peak Multi-GPU Performance**: 3,182 TFLOPS on 8 GPUs (7B model)
2. **13B Model**: Fits comfortably on 8 GPUs with DDP (157 GB/GPU)
3. **30B+ Models**: Require model parallelism (ZeRO/FSDP) instead of data parallelism
4. **Scaling**: Good efficiency up to 4 GPUs, communication overhead increases at 8 GPUs

### Multi-GPU Benchmark Commands

```bash
# Run 8-GPU benchmark
docker run --rm --device /dev/kfd --device /dev/dri --ipc=host \
  -v /home/asrr/amd_test:/workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  python3 /workspace/multigpu_benchmark.py --model 7b --gpus 8 --batch-size 8

# Run scaling test (1,2,4,8 GPUs)
docker run --rm --device /dev/kfd --device /dev/dri --ipc=host \
  -v /home/asrr/amd_test:/workspace \
  rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 \
  python3 /workspace/multigpu_benchmark.py --model 7b --scaling-test
```

---

## Benchmark 5: MAD Framework Training (Qwen3-8B)

Training benchmarks using AMD's MAD framework with torchtune.

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B |
| Batch Size | 14 |
| Sequence Length | 8192 |
| GPUs | 8x MI350X |
| Framework | torchtune (PyTorch 2.10.0) |
| Compile | Enabled |
| Activation Checkpointing | Enabled |

### Results Summary

| Mode | Precision | Tokens/s/GPU | Total Tokens/s | Memory/GPU |
|------|-----------|--------------|----------------|------------|
| **Full Fine-tune** | **FP8** | **20,497** | **163,976** | 76.35 GB |
| Full Fine-tune | BF16 | 14,471 | 115,768 | 76.35 GB |
| LoRA Fine-tune | BF16 | 16,312 | 130,496 | 72.65 GB |

### Full Fine-tuning (FP8) - Best Performance

| Metric | Value |
|--------|-------|
| **Throughput (per GPU)** | **20,497 tokens/s** |
| **Total Throughput** | **~163,976 tokens/s** |
| **Max Memory per GPU** | 76.35 GB |
| Precision | FP8 |

### Full Fine-tuning (BF16)

| Metric | Value |
|--------|-------|
| **Throughput (per GPU)** | **14,471 tokens/s** |
| **Total Throughput** | **~115,768 tokens/s** |
| **Max Memory per GPU** | 76.35 GB |
| Precision | BF16 |

### LoRA Fine-tuning (BF16)

| Metric | Value |
|--------|-------|
| **Throughput (per GPU)** | **16,312 tokens/s** |
| **Total Throughput** | **~130,496 tokens/s** |
| **Max Memory per GPU** | 72.65 GB |
| Precision | BF16 |

### Training Progress (LoRA)

| Epoch | Step | Loss |
|-------|------|------|
| 1 | 1 | 2.011 |
| 1 | 5 | 1.889 |
| 1 | 10 | 1.785 |
| 1 | 11 | 1.765 |

### Key Observations

1. **FP8 Speedup**: FP8 provides **42% speedup** over BF16 for full fine-tuning (20,497 vs 14,471 tokens/s)
2. **LoRA Efficiency**: LoRA uses slightly less memory (72.65 GB vs 76.35 GB) with competitive throughput
3. **Memory Efficient**: Only ~25% of available GPU memory used (76 GB of 309 GB)
4. **Long Context**: Successfully trained with 8192 token sequences
5. **Peak Throughput**: 163,976 tokens/s total (FP8 full fine-tune)

### MAD Benchmark Command

```bash
cd MAD
madengine run --tags pyt_train_qwen3-8b --live-output --timeout 3600
```

---

## Date

Tested: 2026-01-08
