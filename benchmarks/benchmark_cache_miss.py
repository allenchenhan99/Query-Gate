"""
Benchmark cache miss scenario with unique queries
"""

import asyncio
import aiohttp
import time
import statistics


async def send_request(session, url, text):
    """Send request"""
    try:
        async with session.post(url, json={"text": text}, timeout=aiohttp.ClientTimeout(total=5)) as response:
            router_latency = float(response.headers.get('x-router-latency', 0))
            return router_latency
    except:
        return -1


async def test_cache_miss():
    """Test with unique queries (cache miss)"""
    url = "http://localhost:8000/v1/query-process"

    # Generate unique queries to avoid cache
    queries = [
        f"What is the answer to question number {i}?" for i in range(200)
    ]

    print(f"\n{'='*70}")
    print("Cache Miss Test - Unique Queries")
    print(f"{'='*70}")
    print(f"Queries: {len(queries)} unique queries")
    print(f"Expected: All cache misses, need inference")
    print(f"{'='*70}\n")

    latencies = []

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.perf_counter()

        # Send requests sequentially in small batches to avoid overwhelming
        for i in range(0, len(queries), 20):
            batch = queries[i:i+20]
            tasks = [send_request(session, url, q) for q in batch]
            results = await asyncio.gather(*tasks)

            for lat in results:
                if lat > 0:
                    latencies.append(lat)

            print(f"Progress: {min(i+20, len(queries))}/{len(queries)}")

        elapsed = time.perf_counter() - start

    latencies.sort()
    successful = len(latencies)

    if successful > 0:
        p50 = latencies[int(successful * 0.5)]
        p95 = latencies[int(successful * 0.95)] if successful > 20 else latencies[-1]
        p99 = latencies[int(successful * 0.99)] if successful > 100 else latencies[-1]
        mean = statistics.mean(latencies)

        print(f"\nResults (Cache Miss Scenario):")
        print(f"  Successful: {successful}/{len(queries)}")
        print(f"  Total Time: {elapsed:.2f}s")
        print()
        print(f"Router Latency (cache miss):")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Mean: {mean:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        print(f"{'='*70}\n")

        return {'p50': p50, 'p95': p95, 'p99': p99, 'mean': mean}


async def test_cache_hit():
    """Test with repeated query (cache hit)"""
    url = "http://localhost:8000/v1/query-process"
    query = "What is machine learning?"

    print(f"\n{'='*70}")
    print("Cache Hit Test - Repeated Query")
    print(f"{'='*70}")
    print(f"Query: '{query}'")
    print(f"Requests: 100 (same query)")
    print(f"Expected: First request cache miss, rest cache hits")
    print(f"{'='*70}\n")

    latencies = []

    async with aiohttp.ClientSession() as session:
        # First request (cache miss)
        first_lat = await send_request(session, url, query)
        print(f"First request (cache miss): {first_lat:.2f}ms")

        # Wait a bit
        await asyncio.sleep(0.5)

        # Repeated requests (cache hits)
        tasks = [send_request(session, url, query) for _ in range(99)]
        results = await asyncio.gather(*tasks)

        for lat in results:
            if lat > 0:
                latencies.append(lat)

    if latencies:
        print(f"\nCache Hit Latency (99 repeated requests):")
        print(f"  P50: {statistics.median(latencies):.2f}ms")
        print(f"  Mean: {statistics.mean(latencies):.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        print(f"{'='*70}\n")


async def main():
    # Test cache miss
    cache_miss_stats = await test_cache_miss()

    await asyncio.sleep(2)

    # Test cache hit
    await test_cache_hit()

    # Save results
    import json
    with open('cache_benchmark.json', 'w') as f:
        json.dump(cache_miss_stats, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
