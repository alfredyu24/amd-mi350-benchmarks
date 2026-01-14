# AMD GPU Training Benchmark Plan

## System Configuration

- **GPU:** 8x AMD Instinct MI350X (gfx950)
- **ROCm:** 7.1.1
- **VRAM:** 288 GB per GPU (~2.3 TB total)

## Options for Testing MI350X Training Performance

### 1. ROCm/MAD (Model Automation and Dashboarding) - Recommended

AMD's official benchmark framework with pre-configured training scripts.

```bash
# Clone the repository
git clone https://github.com/ROCm/MAD
cd MAD

# Install madengine CLI
pip install -r requirements.txt

# Set Hugging Face token (required for gated models)
export HF_TOKEN=your_hugging_face_token

# Discover available training models
madengine discover --tags training

# Run training benchmark with madengine
madengine run --tags pyt_train_llama-3.1-8b --live-output --timeout 3600
```

#### Available Training Models (via madengine)

| Model | Tag | Type |
|-------|-----|------|
| Llama 3.1 8B | `pyt_train_llama-3.1-8b` | PyTorch |
| Llama 3.1 70B | `pyt_train_llama-3.1-70b` | PyTorch |
| Llama 3.3 70B | `pyt_train_llama-3.3-70b` | PyTorch |
| Qwen 2.5 72B | `pyt_train_qwen2.5-72b` | PyTorch |
| DeepSeek V3 16B | `primus_pyt_train_deepseek-v3-16b` | Primus |

#### Direct Script Usage

```bash
cd MAD/scripts/pytorch_train

# Pretrain Llama 3.1 8B with BF16
./pytorch_benchmark_report.sh -t pretrain -m Llama-3.1-8B -p BF16 -s 8192

# LoRA fine-tuning Llama 3.1 70B
./pytorch_benchmark_report.sh -t finetune_lora -m Llama-3.1-70B -p BF16 -s 8192

# Full weight fine-tuning
./pytorch_benchmark_report.sh -t finetune_fw -m Llama-3.1-8B -p BF16 -s 8192
```

### 2. Primus Framework

AMD's unified LLM training framework.

```bash
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt

export MAD_SECRETS_HFTOKEN="your_hf_token"
madengine run --tags primus_pyt_train_llama-3.1-8b --keep-model-dir
```

### 3. PyTorch Training Docker (Quick Start)

AMD provides pre-built Docker images.

```bash
docker run --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -it rocm/pytorch:rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

### 4. TorchBench

General PyTorch benchmark suite that works with ROCm.

```bash
git clone https://github.com/pytorch/benchmark
cd benchmark
python install.py
python run_benchmark.py
```

### 5. MLPerf Training Benchmarks

AMD's MLPerf Training v5.0 focused on Llama 2 70B LoRA fine-tuning.

## References

- [Training a model with PyTorch on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html)
- [AMD's MLPerf Training Debut](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.0/README.html)
- [Training a model with Primus and PyTorch](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-pytorch.html)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
