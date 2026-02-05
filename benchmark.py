"""
Benchmark script for Query Gateway
Measures P50, P95, P99 latency and throughput
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Tuple
import json


async def send_request(session: aiohttp.ClientSession, url: str, data: dict) -> Tuple[float, str]:
    """Send a single request and return (latency_ms, label)"""
    start = time.perf_counter()
    try:
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
            result = await response.json()
            headers = response.headers
            # Get router latency from header
            router_latency = float(headers.get('x-router-latency', 0))
            elapsed = (time.perf_counter() - start) * 1000  # Total time including simulation
            return router_latency, result.get('label', 'unknown'), elapsed
    except Exception as e:
        print(f"Request failed: {e}")
        return -1, 'error', -1


async def benchmark(url: str, num_requests: int, concurrency: int):
    """Run benchmark with specified concurrency"""

    # Test queries
    test_queries = [
        {"text": "Which is a species of fish? Tope or Rope"},  # Fast Path
        {"text": "Identify which instrument is string or woodwind: Panduri, Zurna"},  # Fast Path
        {"text": "How do I start running?"},  # Slow Path
        {"text": "Write a short poem about the ocean."},  # Slow Path
        {"text": "What is the capital of France?"},  # Fast Path
        {"text": "Explain quantum computing in simple terms."},  # Slow Path
    ]

    print(f"\n{'='*80}")
    print(f"Benchmark Configuration")
    print(f"{'='*80}")
    print(f"Target URL: {url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Test Queries: {len(test_queries)} unique queries")
    print(f"{'='*80}\n")

    router_latencies = []
    total_latencies = []
    labels = {'0': 0, '1': 0, 'error': 0}

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # Create tasks
        tasks = []
        for i in range(num_requests):
            query = test_queries[i % len(test_queries)]
            task = send_request(session, url, query)
            tasks.append(task)

        # Execute with concurrency limit
        results = []
        for i in range(0, len(tasks), concurrency):
            batch = tasks[i:i+concurrency]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

            # Progress indicator
            if (i + concurrency) % 100 == 0:
                print(f"Progress: {min(i + concurrency, num_requests)}/{num_requests} requests completed")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Collect statistics
    for router_lat, label, total_lat in results:
        if router_lat > 0:  # Successful request
            router_latencies.append(router_lat)
            total_latencies.append(total_lat)
            labels[label] = labels.get(label, 0) + 1

    # Calculate percentiles
    router_latencies.sort()
    total_latencies.sort()

    successful_requests = len(router_latencies)

    def percentile(data: List[float], p: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    # Router latency stats
    router_p50 = percentile(router_latencies, 50)
    router_p95 = percentile(router_latencies, 95)
    router_p99 = percentile(router_latencies, 99)
    router_mean = statistics.mean(router_latencies) if router_latencies else 0

    # Total latency stats (including Fast/Slow Path simulation)
    total_p50 = percentile(total_latencies, 50)
    total_p95 = percentile(total_latencies, 95)
    total_p99 = percentile(total_latencies, 99)

    # Throughput
    throughput = successful_requests / total_time

    # Print results
    print(f"\n{'='*80}")
    print(f"Benchmark Results")
    print(f"{'='*80}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Successful Requests: {successful_requests}/{num_requests}")
    print(f"Failed Requests: {labels.get('error', 0)}")
    print()
    print(f"Router Latency (ms) - Decision Time Only:")
    print(f"  P50: {router_p50:.2f}ms")
    print(f"  P95: {router_p95:.2f}ms")
    print(f"  P99: {router_p99:.2f}ms")
    print(f"  Mean: {router_mean:.2f}ms")
    print()
    print(f"Total Latency (ms) - Including Fast/Slow Path Simulation:")
    print(f"  P50: {total_p50:.2f}ms")
    print(f"  P95: {total_p95:.2f}ms")
    print(f"  P99: {total_p99:.2f}ms")
    print()
    print(f"Throughput: {throughput:.2f} req/s")
    print()
    print(f"Routing Distribution:")
    print(f"  Fast Path (Label 0): {labels.get('0', 0)} requests ({labels.get('0', 0)/successful_requests*100:.1f}%)")
    print(f"  Slow Path (Label 1): {labels.get('1', 0)} requests ({labels.get('1', 0)/successful_requests*100:.1f}%)")
    print(f"{'='*80}\n")

    # Return data for README
    return {
        'router_latency': {
            'p50': router_p50,
            'p95': router_p95,
            'p99': router_p99,
            'mean': router_mean
        },
        'throughput': throughput,
        'total_requests': num_requests,
        'successful_requests': successful_requests,
        'concurrency': concurrency,
        'distribution': labels
    }


async def main():
    """Run benchmarks"""
    url = "http://localhost:8000/v1/query-process"

    # Warmup
    print("Warming up...")
    await benchmark(url, num_requests=50, concurrency=10)

    # Wait a bit
    await asyncio.sleep(2)

    # Actual benchmark
    print("\nStarting actual benchmark...")
    results = await benchmark(url, num_requests=1000, concurrency=50)

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
