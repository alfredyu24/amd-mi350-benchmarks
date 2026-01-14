#!/usr/bin/env python3
"""Simple GPU Training Benchmark for AMD MI350X"""

import torch
import torch.nn as nn
import time

def benchmark_matmul():
    """Benchmark matrix multiplication (simulates training compute)"""
    print("=" * 60)
    print("AMD MI350X Training Benchmark")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        return

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Benchmark different sizes
    sizes = [4096, 8192, 16384]

    for size in sizes:
        print(f"Matrix size: {size}x{size}")

        # Create random matrices
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        iterations = 20
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate TFLOPS
        flops = 2 * size * size * size * iterations
        tflops = flops / elapsed / 1e12

        print(f"  Time: {elapsed:.3f}s ({iterations} iterations)")
        print(f"  Throughput: {tflops:.2f} TFLOPS (FP16)")
        print()

def benchmark_simple_training():
    """Benchmark simple model training"""
    print("=" * 60)
    print("Simple Training Loop Benchmark")
    print("=" * 60)

    device = torch.device("cuda:0")

    # Simple model
    model = nn.Sequential(
        nn.Linear(4096, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 4096),
    ).to(device).half()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    batch_size = 256

    # Warmup
    for _ in range(5):
        x = torch.randn(batch_size, 4096, device=device, dtype=torch.float16)
        y = torch.randn(batch_size, 4096, device=device, dtype=torch.float16)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        x = torch.randn(batch_size, 4096, device=device, dtype=torch.float16)
        y = torch.randn(batch_size, 4096, device=device, dtype=torch.float16)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    samples_per_sec = (batch_size * iterations) / elapsed

    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"Time per step: {elapsed/iterations*1000:.2f} ms")
    print()

if __name__ == "__main__":
    benchmark_matmul()
    benchmark_simple_training()
    print("Benchmark complete!")
