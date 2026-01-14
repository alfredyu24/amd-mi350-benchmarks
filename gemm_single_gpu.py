#!/usr/bin/env python3
"""
Single GPU GEMM Performance Benchmark for AMD MI350X
Based on SemiAnalysis methodology
"""

import torch
import torch.nn.functional as F
import time
import os

# Force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

assert torch.cuda.is_available(), "CUDA/ROCm not available"

def get_gpu_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
    }

def do_bench(fn, warmup=50, rep=300):
    """OpenAI-style benchmark with more iterations for accuracy"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times.sort()
    return times[len(times) // 2]  # median

def compute_tflops(m, n, k, time_ms):
    flops = 2 * m * n * k
    return (flops / (time_ms / 1000)) / 1e12

def benchmark_gemm(m, n, k, dtype):
    a = torch.randn(m, k, dtype=dtype, device='cuda')
    b = torch.randn(k, n, dtype=dtype, device='cuda')

    # Warmup and benchmark
    fn = lambda: torch.matmul(a, b)
    time_ms = do_bench(fn)
    tflops = compute_tflops(m, n, k, time_ms)

    del a, b
    torch.cuda.empty_cache()
    return time_ms, tflops

def main():
    gpu_info = get_gpu_info()

    # AMD MI350X theoretical peaks (single GPU)
    THEORETICAL_FP16 = 2306.88
    THEORETICAL_BF16 = 2306.88

    print("=" * 80)
    print("SINGLE GPU GEMM BENCHMARK - AMD MI350X")
    print("=" * 80)
    print(f"GPU: {gpu_info['name']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"Theoretical Peak FP16/BF16: {THEORETICAL_FP16:.0f} TFLOPS")
    print("=" * 80)

    # Test shapes - from small to very large
    shapes = [
        # Standard squares
        (1024, 1024, 1024, "1K x 1K x 1K"),
        (2048, 2048, 2048, "2K x 2K x 2K"),
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (8192, 8192, 8192, "8K x 8K x 8K"),
        (16384, 16384, 16384, "16K x 16K x 16K"),
        (32768, 32768, 32768, "32K x 32K x 32K"),
        # Llama 70B shapes
        (16384, 8192, 1280, "Llama QKV"),
        (16384, 1024, 8192, "Llama Attn Out"),
        (16384, 8192, 7168, "Llama FFN 1"),
        (16384, 3584, 8192, "Llama FFN 2"),
    ]

    dtypes = [
        (torch.float16, "FP16", THEORETICAL_FP16),
        (torch.bfloat16, "BF16", THEORETICAL_BF16),
    ]

    results = {}

    for dtype, dtype_name, theoretical in dtypes:
        print(f"\n{'='*80}")
        print(f"  {dtype_name} GEMM (Theoretical Peak: {theoretical:.0f} TFLOPS)")
        print(f"{'='*80}")
        print(f"{'Shape':<22} {'Description':<15} {'Time (ms)':<12} {'TFLOPS':<10} {'Efficiency'}")
        print("-" * 80)

        dtype_results = []
        for m, n, k, desc in shapes:
            try:
                time_ms, tflops = benchmark_gemm(m, n, k, dtype)
                efficiency = (tflops / theoretical) * 100
                shape_str = f"({m}, {n}, {k})"
                print(f"{shape_str:<22} {desc:<15} {time_ms:>8.3f} ms   {tflops:>7.1f}    {efficiency:>5.1f}%")
                dtype_results.append({
                    'shape': (m, n, k),
                    'desc': desc,
                    'time_ms': time_ms,
                    'tflops': tflops,
                    'efficiency': efficiency
                })
            except torch.cuda.OutOfMemoryError:
                print(f"{shape_str:<22} {desc:<15} OOM")
            except Exception as e:
                print(f"{shape_str:<22} {desc:<15} ERROR: {e}")

        results[dtype_name] = dtype_results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - SINGLE GPU MI350X")
    print(f"{'='*80}")

    for dtype_name, dtype_results in results.items():
        if dtype_results:
            peak = max(r['tflops'] for r in dtype_results)
            avg = sum(r['tflops'] for r in dtype_results) / len(dtype_results)
            best = max(dtype_results, key=lambda x: x['tflops'])
            theoretical = THEORETICAL_FP16 if dtype_name == "FP16" else THEORETICAL_BF16

            print(f"\n{dtype_name}:")
            print(f"  Peak TFLOPS:    {peak:.1f} ({peak/theoretical*100:.1f}% of theoretical)")
            print(f"  Average TFLOPS: {avg:.1f} ({avg/theoretical*100:.1f}% of theoretical)")
            print(f"  Best Shape:     {best['desc']} - {best['shape']}")

    print(f"\n{'='*80}")
    print("COMPARISON WITH AMD CLAIMS")
    print(f"{'='*80}")
    print(f"AMD Claimed FP16/BF16: {THEORETICAL_FP16:.0f} TFLOPS")

    if 'FP16' in results and results['FP16']:
        fp16_peak = max(r['tflops'] for r in results['FP16'])
        gap = THEORETICAL_FP16 - fp16_peak
        print(f"Actual FP16 Peak:      {fp16_peak:.0f} TFLOPS")
        print(f"Gap:                   {gap:.0f} TFLOPS ({gap/THEORETICAL_FP16*100:.1f}% below theoretical)")

    if 'BF16' in results and results['BF16']:
        bf16_peak = max(r['tflops'] for r in results['BF16'])
        gap = THEORETICAL_BF16 - bf16_peak
        print(f"Actual BF16 Peak:      {bf16_peak:.0f} TFLOPS")
        print(f"Gap:                   {gap:.0f} TFLOPS ({gap/THEORETICAL_BF16*100:.1f}% below theoretical)")

if __name__ == "__main__":
    main()
