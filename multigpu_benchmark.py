#!/usr/bin/env python3
"""
Multi-GPU Transformer Training Benchmark for AMD MI350X
Uses PyTorch DistributedDataParallel (DDP) for data-parallel training across multiple GPUs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import math
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    vocab_size: int = 50257
    max_seq_len: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    d_model: int = 4096
    d_ff: int = 11008
    dropout: float = 0.0
    bias: bool = False


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_configs():
    return {
        "7b": TransformerConfig(
            n_layers=32, n_heads=32, d_model=4096, d_ff=11008,
            max_seq_len=2048
        ),
        "13b": TransformerConfig(
            n_layers=40, n_heads=40, d_model=5120, d_ff=13824,
            max_seq_len=2048
        ),
        "30b": TransformerConfig(
            n_layers=60, n_heads=52, d_model=6656, d_ff=17920,
            max_seq_len=2048
        ),
        "70b": TransformerConfig(
            n_layers=80, n_heads=64, d_model=8192, d_ff=28672,
            max_seq_len=2048
        ),
    }


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def benchmark_worker(rank, world_size, model_name, batch_size, seq_len, iterations, results_dict):
    setup(rank, world_size)

    configs = get_model_configs()
    config = configs[model_name]
    config.max_seq_len = seq_len

    device = torch.device(f"cuda:{rank}")

    # Create model
    model = GPTModel(config).to(device).to(torch.float16)
    model = DDP(model, device_ids=[rank])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        print("=" * 70)
        print(f"Multi-GPU Transformer Training Benchmark: {model_name.upper()}")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
        print(f"Layers: {config.n_layers}")
        print(f"Heads: {config.n_heads}")
        print(f"Model Dim: {config.d_model}")
        print(f"Batch Size per GPU: {batch_size}")
        print(f"Global Batch Size: {batch_size * world_size}")
        print(f"Sequence Length: {seq_len}")
        print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Warmup
    if rank == 0:
        print("Warming up...")

    for _ in range(3):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if rank == 0:
        print(f"Running {iterations} iterations...")

    total_tokens = 0

    torch.cuda.synchronize()
    dist.barrier()
    start_time = time.time()

    for i in range(iterations):
        optimizer.zero_grad()
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += batch_size * seq_len * world_size

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - start_time

    # Gather results on rank 0
    if rank == 0:
        tokens_per_sec = total_tokens / elapsed
        global_batch = batch_size * world_size
        samples_per_sec = (global_batch * iterations) / elapsed
        time_per_step = (elapsed / iterations) * 1000

        # TFLOPS calculation
        flops_per_token = 6 * n_params
        total_flops = flops_per_token * total_tokens
        tflops = total_flops / elapsed / 1e12

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Time per Step: {time_per_step:.2f} ms")
        print(f"Global Tokens/sec: {tokens_per_sec:,.0f}")
        print(f"Global Samples/sec: {samples_per_sec:.2f}")
        print(f"Tokens/sec per GPU: {tokens_per_sec/world_size:,.0f}")
        print(f"Estimated Total TFLOPS: {tflops:.2f}")
        print(f"TFLOPS per GPU: {tflops/world_size:.2f}")
        print()

        # Memory usage per GPU
        for i in range(world_size):
            mem = torch.cuda.max_memory_allocated(i) / 1e9
            print(f"GPU {i} Memory: {mem:.2f} GB")

        results_dict['tokens_per_sec'] = tokens_per_sec
        results_dict['tflops'] = tflops
        results_dict['time_per_step'] = time_per_step

    cleanup()


def run_benchmark(model_name, num_gpus, batch_size, seq_len, iterations):
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        benchmark_worker,
        args=(num_gpus, model_name, batch_size, seq_len, iterations, results),
        nprocs=num_gpus,
        join=True
    )

    return dict(results)


def run_scaling_test(model_name, batch_size, seq_len, iterations):
    """Test scaling across different GPU counts"""
    max_gpus = torch.cuda.device_count()

    print("=" * 70)
    print(f"MULTI-GPU SCALING TEST: {model_name.upper()}")
    print(f"Max GPUs available: {max_gpus}")
    print("=" * 70)
    print()

    results = []
    gpu_counts = [1, 2, 4, 8][:max_gpus.bit_length()]  # Only test up to available GPUs
    gpu_counts = [g for g in gpu_counts if g <= max_gpus]

    for num_gpus in gpu_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {num_gpus} GPU(s)...")
        print(f"{'='*70}\n")

        try:
            result = run_benchmark(model_name, num_gpus, batch_size, seq_len, iterations)
            result['num_gpus'] = num_gpus
            results.append(result)
        except Exception as e:
            print(f"Error with {num_gpus} GPUs: {e}")
            continue

        torch.cuda.empty_cache()

    # Print scaling summary
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"{'GPUs':<8} {'Tokens/sec':<15} {'TFLOPS':<12} {'Scaling Eff':<12}")
    print("-" * 70)

    base_throughput = results[0]['tokens_per_sec'] if results else 0
    for r in results:
        ideal = base_throughput * r['num_gpus']
        efficiency = (r['tokens_per_sec'] / ideal * 100) if ideal > 0 else 0
        print(f"{r['num_gpus']:<8} {r['tokens_per_sec']:>12,.0f} {r['tflops']:>10.2f} {efficiency:>10.1f}%")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Transformer Training Benchmark")
    parser.add_argument("--model", type=str, default="7b",
                        choices=["7b", "13b", "30b", "70b"],
                        help="Model size")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of training iterations")
    parser.add_argument("--scaling-test", action="store_true",
                        help="Run scaling test across 1,2,4,8 GPUs")

    args = parser.parse_args()

    num_gpus = args.gpus if args.gpus else torch.cuda.device_count()

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    if args.scaling_test:
        run_scaling_test(args.model, args.batch_size, args.seq_len, args.iterations)
    else:
        run_benchmark(args.model, num_gpus, args.batch_size, args.seq_len, args.iterations)
