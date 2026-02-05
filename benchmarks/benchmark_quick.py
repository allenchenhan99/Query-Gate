"""
Quick benchmark focusing on Router latency only
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List


async def send_request(session: aiohttp.ClientSession, url: str, data: dict):
    """Send request and return router latency"""
    try:
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
            result = await response.json()
            router_latency = float(response.headers.get('x-router-latency', 0))
            return router_latency, result.get('label', 'error')
    except asyncio.TimeoutError:
        return -1, 'timeout'
    except Exception as e:
        return -1, 'error'


async def benchmark_router_latency(url: str, num_requests: int, concurrency: int):
    """Benchmark router latency"""

    test_queries = [
        {"text": "Which is a species of fish? Tope or Rope"},
        {"text": "Identify which instrument is string or woodwind: Panduri, Zurna"},
        {"text": "How do I start running?"},
        {"text": "Write a short poem about the ocean."},
        {"text": "What is the capital of France?"},
        {"text": "Explain quantum computing in simple terms."},
    ]

    print(f"\n{'='*70}")
    print(f"Router Latency Benchmark")
    print(f"{'='*70}")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"{'='*70}\n")

    latencies = []
    labels = {'0': 0, '1': 0, 'timeout': 0, 'error': 0}

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            query = test_queries[i % len(test_queries)]
            tasks.append(send_request(session, url, query))

        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    for lat, label in results:
        if lat > 0:
            latencies.append(lat)
        labels[label] = labels.get(label, 0) + 1

    latencies.sort()
    successful = len(latencies)

    if successful > 0:
        p50 = latencies[int(successful * 0.5)]
        p95 = latencies[int(successful * 0.95)] if successful > 20 else latencies[-1]
        p99 = latencies[int(successful * 0.99)] if successful > 100 else latencies[-1]
        mean = statistics.mean(latencies)

        print(f"Results:")
        print(f"  Successful: {successful}/{num_requests}")
        print(f"  Failed: {labels.get('timeout', 0) + labels.get('error', 0)}")
        print(f"  Total Time: {elapsed:.2f}s")
        print(f"  Throughput: {num_requests/elapsed:.2f} req/s")
        print()
        print(f"Router Latency (x-router-latency header):")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Mean: {mean:.2f}ms")
        print()
        print(f"Routing:")
        print(f"  Fast (0): {labels.get('0', 0)}")
        print(f"  Slow (1): {labels.get('1', 0)}")
        print(f"{'='*70}\n")

        return {
            'p50': p50, 'p95': p95, 'p99': p99, 'mean': mean,
            'throughput': num_requests/elapsed,
            'successful': successful,
            'total': num_requests
        }


async def main():
    url = "http://localhost:8000/v1/query-process"

    # Test 1: Low concurrency (focus on cache miss)
    print("\n[Test 1] Low Concurrency - First-time requests")
    await benchmark_router_latency(url, num_requests=100, concurrency=10)

    await asyncio.sleep(2)

    # Test 2: Medium concurrency (mix of cache hit/miss)
    print("\n[Test 2] Medium Concurrency - Mixed requests")
    await benchmark_router_latency(url, num_requests=500, concurrency=50)

    await asyncio.sleep(2)

    # Test 3: High concurrency (mostly cache hits)
    print("\n[Test 3] High Concurrency - Repeated requests")
    await benchmark_router_latency(url, num_requests=1000, concurrency=100)


if __name__ == "__main__":
    asyncio.run(main())
