#!/usr/bin/env python3
"""
GEMM Performance Benchmark for AMD MI350X
Based on SemiAnalysis methodology (MI300X vs H100/H200 comparison)

Tests real-world matrix shapes from Llama 70B training workloads
Using OpenAI do_bench style timing with proper warmup and repetitions
"""

import torch
import torch.nn.functional as F
import time
import argparse
from typing import List, Tuple

# Ensure we're using HIP/ROCm
assert torch.cuda.is_available(), "CUDA/ROCm not available"

def get_gpu_info():
    """Get GPU information"""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "num_gpus": torch.cuda.device_count(),
    }

def do_bench(fn, warmup=30, rep=200):
    """
    OpenAI-style benchmark function
    - warmup: number of warmup iterations
    - rep: number of timed repetitions
    Returns median time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times.sort()
    return times[len(times) // 2]  # Return median

def compute_tflops(m: int, n: int, k: int, time_ms: float) -> float:
    """Compute TFLOPS for a GEMM operation"""
    # GEMM: 2*M*N*K FLOPs (multiply-add = 2 ops)
    flops = 2 * m * n * k
    tflops = (flops / (time_ms / 1000)) / 1e12
    return tflops

def benchmark_gemm(m: int, n: int, k: int, dtype: torch.dtype, use_linear: bool = False) -> Tuple[float, float]:
    """
    Benchmark a single GEMM operation
    Returns: (time_ms, tflops)
    """
    # Initialize with normal distribution (matches NN weight distribution)
    a = torch.randn(m, k, dtype=dtype, device='cuda')
    b = torch.randn(k, n, dtype=dtype, device='cuda')

    if use_linear:
        # F.linear uses (input, weight.T) -> uses rocBLAS by default
        weight = torch.randn(n, k, dtype=dtype, device='cuda')
        input_tensor = torch.randn(m, k, dtype=dtype, device='cuda')
        fn = lambda: F.linear(input_tensor, weight)
    else:
        # torch.matmul uses hipBLASLt (optimized path)
        fn = lambda: torch.matmul(a, b)

    time_ms = do_bench(fn)
    tflops = compute_tflops(m, n, k, time_ms)

    # Clear tensors
    del a, b
    if use_linear:
        del weight, input_tensor
    torch.cuda.empty_cache()

    return time_ms, tflops

def run_benchmark_suite(warmup: int = 30, rep: int = 200):
    """Run the full GEMM benchmark suite"""

    # GPU info
    gpu_info = get_gpu_info()
    print("=" * 70)
    print("GEMM PERFORMANCE BENCHMARK")
    print("Based on SemiAnalysis MI300X vs H100/H200 methodology")
    print("=" * 70)
    print(f"GPU: {gpu_info['name']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"GPU Count: {gpu_info['num_gpus']}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timed repetitions: {rep}")
    print("=" * 70)

    # Matrix shapes from Llama 70B training (SemiAnalysis)
    shapes = [
        # (M, N, K, Description)
        (16384, 8192, 1280, "Fused QKV Projection"),
        (16384, 1024, 8192, "Attention Output Projection"),
        (16384, 8192, 7168, "FFN GEMM 1"),
        (16384, 3584, 8192, "FFN GEMM 2"),
        (8192, 8192, 8192, "Standard Square (8K)"),
        (4096, 4096, 4096, "Standard Square (4K)"),
        (16384, 16384, 16384, "Large Square (16K)"),
    ]

    # Data types to test
    dtypes = [
        (torch.float16, "FP16"),
        (torch.bfloat16, "BF16"),
    ]

    results = []

    for dtype, dtype_name in dtypes:
        print(f"\n{'='*70}")
        print(f"Data Type: {dtype_name}")
        print(f"{'='*70}")
        print(f"{'Shape':<35} {'Time (ms)':<12} {'TFLOPS':<12} {'Efficiency'}")
        print("-" * 70)

        # Theoretical peak for MI350X (estimated based on MI300X specs scaled)
        # MI300X: 1,307 TFLOPS FP16, MI350X should be higher
        # Using 1,500 TFLOPS as estimated peak for MI350X FP16
        theoretical_peak = 1500 if dtype == torch.float16 else 1500  # BF16 same as FP16 on AMD

        for m, n, k, desc in shapes:
            try:
                time_ms, tflops = benchmark_gemm(m, n, k, dtype)
                efficiency = (tflops / theoretical_peak) * 100
                shape_str = f"({m}, {n}, {k})"
                print(f"{shape_str:<20} {desc:<14} {time_ms:>8.3f} ms  {tflops:>8.1f}    {efficiency:>5.1f}%")
                results.append({
                    'dtype': dtype_name,
                    'shape': (m, n, k),
                    'desc': desc,
                    'time_ms': time_ms,
                    'tflops': tflops,
                    'efficiency': efficiency
                })
            except Exception as e:
                print(f"{desc:<35} ERROR: {e}")

    # Test hipBLASLt vs rocBLAS (torch.matmul vs F.linear)
    print(f"\n{'='*70}")
    print("LIBRARY COMPARISON: torch.matmul (hipBLASLt) vs F.linear (rocBLAS)")
    print(f"{'='*70}")
    print(f"{'Shape':<25} {'matmul TFLOPS':<15} {'linear TFLOPS':<15} {'Difference'}")
    print("-" * 70)

    test_shapes = [
        (8192, 8192, 8192),
        (16384, 8192, 1280),
        (4096, 4096, 4096),
    ]

    for m, n, k in test_shapes:
        try:
            _, tflops_matmul = benchmark_gemm(m, n, k, torch.float16, use_linear=False)
            _, tflops_linear = benchmark_gemm(m, n, k, torch.float16, use_linear=True)
            diff = ((tflops_matmul - tflops_linear) / tflops_linear) * 100
            shape_str = f"({m}, {n}, {k})"
            print(f"{shape_str:<25} {tflops_matmul:>10.1f}     {tflops_linear:>10.1f}     {diff:>+6.1f}%")
        except Exception as e:
            print(f"({m}, {n}, {k})  ERROR: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    fp16_results = [r for r in results if r['dtype'] == 'FP16']
    bf16_results = [r for r in results if r['dtype'] == 'BF16']

    if fp16_results:
        avg_fp16 = sum(r['tflops'] for r in fp16_results) / len(fp16_results)
        max_fp16 = max(r['tflops'] for r in fp16_results)
        print(f"FP16 Average TFLOPS: {avg_fp16:.1f}")
        print(f"FP16 Peak TFLOPS: {max_fp16:.1f}")

    if bf16_results:
        avg_bf16 = sum(r['tflops'] for r in bf16_results) / len(bf16_results)
        max_bf16 = max(r['tflops'] for r in bf16_results)
        print(f"BF16 Average TFLOPS: {avg_bf16:.1f}")
        print(f"BF16 Peak TFLOPS: {max_bf16:.1f}")

    print(f"\nEnvironment variables:")
    import os
    for var in ['TORCH_BLAS_PREFER_HIPBLASLT', 'VLLM_ROCM_USE_AITER', 'HIP_FORCE_DEV_KERNARG']:
        print(f"  {var}={os.environ.get(var, 'not set')}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM Benchmark for AMD MI350X")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Timed repetitions")
    args = parser.parse_args()

    run_benchmark_suite(warmup=args.warmup, rep=args.rep)
