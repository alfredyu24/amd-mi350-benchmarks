#!/usr/bin/env python3
"""
DeepSeek R1 Throughput Benchmark Script
Tests vLLM inference throughput with concurrent requests
"""

import requests
import time
import concurrent.futures
import argparse

URL = "http://localhost:8000/v1/chat/completions"

def send_request(req_id, prompt, max_tokens, model):
    try:
        response = requests.post(URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }, timeout=120)
        data = response.json()
        return data['usage']['completion_tokens']
    except Exception as e:
        print(f"Request {req_id} failed: {e}")
        return 0

def run_benchmark(concurrency, prompt, max_tokens, model):
    print(f"\n=== Concurrency: {concurrency} ===")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i, prompt, max_tokens, model)
                   for i in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.time() - start_time
    total_tokens = sum(results)
    successful = len([r for r in results if r > 0])

    print(f"Requests completed: {successful}/{concurrency}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_tokens/total_time:.2f} tokens/sec")

    return {
        "concurrency": concurrency,
        "successful": successful,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "throughput": total_tokens / total_time
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM throughput")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1", help="Model name")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256],
                        help="Concurrency levels to test")
    parser.add_argument("--peak-only", action="store_true", help="Only run peak test (256 concurrent)")
    args = parser.parse_args()

    print("=" * 60)
    print("DeepSeek R1 Throughput Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    if args.peak_only:
        concurrency_levels = [256]
    else:
        concurrency_levels = args.concurrency

    results = []
    for concurrency in concurrency_levels:
        result = run_benchmark(concurrency, args.prompt, args.max_tokens, args.model)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Concurrency':<12} {'Tokens':<12} {'Time (s)':<12} {'Throughput':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['concurrency']:<12} {r['total_tokens']:<12} {r['total_time']:<12.2f} {r['throughput']:<15.2f}")

    peak = max(results, key=lambda x: x['throughput'])
    print("-" * 60)
    print(f"Peak throughput: {peak['throughput']:.2f} tokens/sec at concurrency {peak['concurrency']}")

if __name__ == "__main__":
    main()
