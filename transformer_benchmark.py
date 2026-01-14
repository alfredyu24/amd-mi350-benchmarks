#!/usr/bin/env python3
"""
Advanced Transformer Training Benchmark for AMD MI350X

This benchmark implements a GPT-style transformer model with:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Causal masking
- Gradient accumulation
- Mixed precision training (FP16/BF16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    vocab_size: int = 50257      # GPT-2 vocabulary size
    max_seq_len: int = 1024      # Maximum sequence length
    n_layers: int = 12           # Number of transformer blocks
    n_heads: int = 12            # Number of attention heads
    d_model: int = 768           # Model dimension
    d_ff: int = 3072             # Feed-forward dimension (4x d_model)
    dropout: float = 0.1         # Dropout rate
    bias: bool = False           # Use bias in linear layers


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""

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
    """Single transformer block with pre-norm architecture"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (like GPT-2)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style transformer model"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (share embedding weights with output)
        self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
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

        # Get embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos)  # (1, T, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_configs():
    """Create different model size configurations"""
    return {
        "small": TransformerConfig(
            n_layers=6, n_heads=6, d_model=384, d_ff=1536,  # ~30M params
            max_seq_len=512
        ),
        "medium": TransformerConfig(
            n_layers=12, n_heads=12, d_model=768, d_ff=3072,  # ~125M params (GPT-2 small)
            max_seq_len=1024
        ),
        "large": TransformerConfig(
            n_layers=24, n_heads=16, d_model=1024, d_ff=4096,  # ~350M params (GPT-2 medium)
            max_seq_len=1024
        ),
        "xl": TransformerConfig(
            n_layers=36, n_heads=20, d_model=1280, d_ff=5120,  # ~760M params (GPT-2 large)
            max_seq_len=1024
        ),
        "1.5b": TransformerConfig(
            n_layers=48, n_heads=24, d_model=1536, d_ff=6144,  # ~1.5B params (GPT-2 XL)
            max_seq_len=1024
        ),
        "3b": TransformerConfig(
            n_layers=32, n_heads=32, d_model=2560, d_ff=10240,  # ~3B params
            max_seq_len=2048
        ),
        "7b": TransformerConfig(
            n_layers=32, n_heads=32, d_model=4096, d_ff=11008,  # ~7B params (Llama-style)
            max_seq_len=2048
        ),
        "13b": TransformerConfig(
            n_layers=40, n_heads=40, d_model=5120, d_ff=13824,  # ~13B params (Llama-style)
            max_seq_len=2048
        ),
    }


def benchmark_transformer(config_name: str = "medium", batch_size: int = 8,
                          seq_len: int = 512, iterations: int = 50,
                          gradient_accumulation: int = 1,
                          use_fp16: bool = True):
    """Run transformer training benchmark"""

    print("=" * 70)
    print(f"Transformer Training Benchmark: {config_name.upper()}")
    print("=" * 70)

    device = torch.device("cuda:0")
    configs = create_model_configs()
    config = configs[config_name]
    config.max_seq_len = seq_len

    # Create model
    model = GPTModel(config).to(device)
    dtype = torch.float16 if use_fp16 else torch.float32
    model = model.to(dtype)

    n_params = model.count_parameters()
    print(f"Model: GPT-style Transformer")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Layers: {config.n_layers}")
    print(f"Heads: {config.n_heads}")
    print(f"Model Dim: {config.d_model}")
    print(f"FF Dim: {config.d_ff}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Gradient Accumulation: {gradient_accumulation}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Warmup
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

    # Benchmark
    print(f"Running {iterations} iterations...")

    total_tokens = 0
    losses = []

    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(iterations):
        optimizer.zero_grad()

        accum_loss = 0
        for _ in range(gradient_accumulation):
            # Generate random input (simulating tokenized text)
            idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

            # Forward pass
            logits = model(idx)

            # Compute loss (cross-entropy for language modeling)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                targets.view(-1)
            ) / gradient_accumulation

            # Backward pass
            loss.backward()
            accum_loss += loss.item() * gradient_accumulation
            total_tokens += batch_size * seq_len

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(accum_loss)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Calculate metrics
    tokens_per_sec = total_tokens / elapsed
    samples_per_sec = (batch_size * iterations * gradient_accumulation) / elapsed
    time_per_step = (elapsed / iterations) * 1000  # ms
    avg_loss = sum(losses) / len(losses)

    # Estimate TFLOPS (approximate)
    # For transformer: ~6 * n_params * tokens_processed for forward+backward
    flops_per_token = 6 * n_params
    total_flops = flops_per_token * total_tokens
    tflops = total_flops / elapsed / 1e12

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Time per Step: {time_per_step:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"Samples/sec: {samples_per_sec:.2f}")
    print(f"Estimated TFLOPS: {tflops:.2f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print()

    # Memory usage
    memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    memory_reserved = torch.cuda.max_memory_reserved() / 1e9
    print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
    print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
    print()

    return {
        "config": config_name,
        "parameters": n_params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
        "time_per_step_ms": time_per_step,
        "tflops": tflops,
        "memory_gb": memory_allocated,
    }


def run_all_benchmarks():
    """Run benchmarks for different model sizes"""

    print("=" * 70)
    print("AMD MI350X TRANSFORMER TRAINING BENCHMARK SUITE")
    print("=" * 70)
    print()

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GPUs Available: {torch.cuda.device_count()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    results = []

    # Test different configurations
    benchmarks = [
        # (config, batch_size, seq_len, iterations)
        ("small", 32, 512, 100),    # ~30M params
        ("medium", 16, 512, 50),    # ~125M params
        ("large", 8, 512, 30),      # ~350M params
        ("xl", 4, 512, 20),         # ~760M params
    ]

    for config_name, batch_size, seq_len, iters in benchmarks:
        try:
            result = benchmark_transformer(
                config_name=config_name,
                batch_size=batch_size,
                seq_len=seq_len,
                iterations=iters,
                gradient_accumulation=1,
                use_fp16=True
            )
            results.append(result)
        except RuntimeError as e:
            print(f"ERROR running {config_name}: {e}")
            continue

        # Clear cache between runs
        torch.cuda.empty_cache()
        print("\n" + "=" * 70 + "\n")

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<10} {'Params':<12} {'Batch':<8} {'Tokens/s':<15} {'TFLOPS':<10} {'Memory':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<10} {r['parameters']/1e6:.1f}M{'':<6} {r['batch_size']:<8} "
              f"{r['tokens_per_sec']:>12,.0f} {r['tflops']:>8.2f} {r['memory_gb']:>8.2f} GB")
    print()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Transformer Training Benchmark")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["small", "medium", "large", "xl", "1.5b", "3b", "7b", "13b"],
                        help="Model size to benchmark")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (auto-selected if not specified)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of training iterations")
    parser.add_argument("--full", action="store_true",
                        help="Run all model sizes")
    parser.add_argument("--large-models", action="store_true",
                        help="Run only large models (1.5B+)")

    args = parser.parse_args()

    if args.full:
        run_all_benchmarks()
    elif args.large_models:
        # Run larger model benchmarks
        print("=" * 70)
        print("LARGE MODEL BENCHMARK SUITE (1.5B+)")
        print("=" * 70)

        large_benchmarks = [
            # (config, batch_size, seq_len, iterations)
            ("1.5b", 4, 512, 20),    # ~1.5B params
            ("3b", 2, 512, 15),      # ~3B params
            ("7b", 1, 512, 10),      # ~7B params
        ]

        results = []
        for config_name, batch_size, seq_len, iters in large_benchmarks:
            try:
                result = benchmark_transformer(
                    config_name=config_name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    iterations=iters,
                    use_fp16=True
                )
                results.append(result)
            except RuntimeError as e:
                print(f"ERROR running {config_name}: {e}")
                continue
            torch.cuda.empty_cache()
            print("\n" + "=" * 70 + "\n")

        # Summary
        print("=" * 70)
        print("LARGE MODEL BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Model':<10} {'Params':<12} {'Batch':<8} {'Tokens/s':<15} {'TFLOPS':<10} {'Memory':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<10} {r['parameters']/1e9:.2f}B{'':<5} {r['batch_size']:<8} "
                  f"{r['tokens_per_sec']:>12,.0f} {r['tflops']:>8.2f} {r['memory_gb']:>8.2f} GB")
    else:
        # Single model benchmark
        # Auto-select batch size if not specified
        if args.batch_size is None:
            batch_sizes = {
                "small": 32,
                "medium": 16,
                "large": 8,
                "xl": 4,
                "1.5b": 4,
                "3b": 2,
                "7b": 1,
                "13b": 1,
            }
            batch_size = batch_sizes.get(args.model, 4)
        else:
            batch_size = args.batch_size

        benchmark_transformer(
            config_name=args.model,
            batch_size=batch_size,
            seq_len=args.seq_len,
            iterations=args.iterations,
            use_fp16=True
        )
