#!/usr/bin/env python3
"""Benchmark KV cache transfer: compare performance with and without transfer.

Runs multiple workload patterns and measures latency, throughput, and cache
efficiency. Designed to be called twice (once with transfer enabled, once
without) and results combined at the end.

Usage:
    python bench_kv_transfer.py --label with-transfer
    python bench_kv_transfer.py --label no-transfer
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import List

import aiohttp

# ---------------------------------------------------------------------------
# Shared prefix (~280 tokens after tokenization with Qwen3)
# ---------------------------------------------------------------------------
SHARED_PREFIX = (
    "You are a helpful assistant. I want you to analyze the following "
    "detailed technical specification for a distributed inference system. "
    "The system uses a KV-aware router that maintains a global radix tree "
    "indexing which KV cache blocks exist on which workers. When a request "
    "arrives, the router computes block hashes from the token sequence and "
    "looks up overlap with each worker's cached blocks. The cost function "
    "balances prefill cost (tokens that must be computed) against queue "
    "depth (how busy the worker is). With cross-worker KV cache transfer "
    "enabled, the router can also consider routing to a less-loaded worker "
    "and attaching a TransferHint that tells the target worker to pull "
    "cached blocks from a remote source worker via NIXL RDMA. This avoids "
    "redundant prefill computation and improves time-to-first-token. "
    "The transfer protocol has three steps: (1) the target fetches NIXL "
    "metadata from the source, (2) the target queries which physical KV "
    "cache page indices correspond to the needed block hashes, and (3) the "
    "target executes an RDMA read to pull the blocks directly from the "
    "source GPU's memory into its own GPU memory. After transfer, the "
    "blocks are inserted into the target's local radix tree and KV events "
    "are published so the router learns the blocks now exist on both "
    "workers. Please analyze this system and answer the question below.\n\n"
)

SUFFIXES = [
    "What are the main benefits of this architecture?",
    "What failure modes should be handled?",
    "How does the cost function decide when to transfer?",
    "What happens if the source worker evicts blocks during transfer?",
    "How does this interact with tensor parallelism?",
    "What metrics would you monitor in production?",
    "How would you tune the transfer_cost_weight parameter?",
    "What is the expected latency improvement from KV transfer?",
    "How does the radix tree integration work on the receiving worker?",
    "What are the known limitations of this approach?",
    "Compare this to disaggregated prefill/decode architectures.",
    "How would you test this system end-to-end?",
    "What network bandwidth is needed for efficient transfer?",
    "How does page_size affect transfer granularity?",
    "What happens when both workers have partial overlap?",
    "Describe the NIXL RDMA transfer protocol in detail.",
    "What security concerns arise with direct GPU memory access?",
    "How does the eviction policy interact with transferred blocks?",
    "What is the impact on tail latency under high load?",
    "How would you implement graceful degradation on transfer failure?",
    "Describe the interaction between KV transfer and speculative decoding.",
    "What are the memory overhead implications of dual-cached blocks?",
    "How does the system handle network partitions between workers?",
    "What role does the ZMQ event bus play in cache coherence?",
    "How would you design a cache warming strategy for cold starts?",
    "What are the tradeoffs between transfer latency and prefill latency?",
    "How does request priority scheduling interact with KV transfer?",
    "What is the optimal number of workers for this architecture?",
    "How does context length affect transfer efficiency?",
    "Describe the memory layout of KV cache blocks on GPU.",
    "What are the implications for multi-turn conversations?",
    "How would you handle KV transfer across different GPU types?",
    "What is the role of the block hash in cache coherence?",
    "How does batch scheduling affect transfer decisions?",
    "Describe the lifecycle of a KV cache block from allocation to eviction.",
    "What are the failure recovery mechanisms for interrupted transfers?",
    "How does the system scale with increasing number of workers?",
    "What is the impact of network congestion on transfer performance?",
    "How would you implement prefetch for predictable access patterns?",
    "Describe the interaction between KV transfer and continuous batching.",
    "What are the consistency guarantees of the distributed KV cache?",
    "How does the system handle heterogeneous GPU memory sizes?",
    "What monitoring dashboards would you build for this system?",
    "How does the router handle conflicting transfer hints?",
    "Describe the NIXL agent registration and discovery protocol.",
    "What are the implications of KV transfer for model parallelism beyond TP?",
    "How would you benchmark KV transfer under realistic production traffic?",
    "What is the expected behavior under memory pressure on the source worker?",
    "How does the system handle concurrent transfers to the same target?",
    "Describe the error handling chain from RDMA failure to user response.",
    "What are the power and thermal implications of frequent RDMA transfers?",
    "How would you implement KV transfer for mixture-of-experts models?",
    "What is the interaction between KV transfer and request cancellation?",
    "How does the system maintain cache coherence during worker restarts?",
    "Describe the optimal configuration for latency-sensitive workloads.",
    "What are the security implications of shared memory across workers?",
    "How would you extend this system to support cross-node transfers?",
    "What is the role of the cost function weight in routing decisions?",
    "How does the system handle token sequence collisions in the radix tree?",
    "Describe the performance characteristics of NIXL vs other RDMA libraries.",
    "What are the implications for serving multiple models on the same cluster?",
    "How would you implement admission control for transfer-heavy workloads?",
    "What is the expected throughput improvement under sustained load?",
    "How does the system handle partial block transfers?",
    "Describe the interaction between KV transfer and dynamic batching.",
]


@dataclass
class RequestResult:
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_s: float = 0.0  # time to first token (streaming)


@dataclass
class ScenarioResult:
    name: str
    num_requests: int
    concurrency: int
    wall_time_s: float
    avg_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    p99_latency_s: float
    min_latency_s: float
    max_latency_s: float
    avg_ttft_s: float
    total_cached_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    throughput_req_per_s: float
    requests: List[RequestResult] = field(default_factory=list)


async def chat_completion_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 64,
) -> RequestResult:
    """Send a streaming chat completion and measure TTFT + total latency."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    t0 = time.monotonic()
    ttft = 0.0
    completion_tokens = 0

    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") and ttft == 0.0:
                ttft = time.monotonic() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]

    elapsed = time.monotonic() - t0

    # Usage is typically in the last chunk with stream_options
    # Fall back to non-streaming request if needed
    prompt_tokens = usage.get("prompt_tokens", 0) if "usage" in dir() else 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if "usage" in dir() else 0

    return RequestResult(
        latency_s=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=usage.get("completion_tokens", 0) if "usage" in dir() else 0,
        cached_tokens=cached,
        ttft_s=ttft if ttft > 0 else elapsed,
    )


async def chat_completion_non_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 64,
) -> RequestResult:
    """Send a non-streaming chat completion request."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.monotonic()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    elapsed = time.monotonic() - t0

    usage = data.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return RequestResult(
        latency_s=elapsed,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=cached,
        ttft_s=elapsed,  # no streaming, ttft ≈ total latency
    )


async def run_scenario(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    name: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 64,
) -> ScenarioResult:
    """Run a batch of requests and collect metrics."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded(prompt):
        async with sem:
            return await chat_completion_non_streaming(
                session, base_url, model, prompt, max_tokens
            )

    t0 = time.monotonic()
    results = await asyncio.gather(*[bounded(p) for p in prompts])
    wall_time = time.monotonic() - t0

    latencies = [r.latency_s for r in results]
    ttfts = [r.ttft_s for r in results]
    latencies_sorted = sorted(latencies)

    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    return ScenarioResult(
        name=name,
        num_requests=len(results),
        concurrency=concurrency,
        wall_time_s=round(wall_time, 3),
        avg_latency_s=round(statistics.mean(latencies), 3),
        p50_latency_s=round(percentile(latencies_sorted, 50), 3),
        p90_latency_s=round(percentile(latencies_sorted, 90), 3),
        p99_latency_s=round(percentile(latencies_sorted, 99), 3),
        min_latency_s=round(min(latencies), 3),
        max_latency_s=round(max(latencies), 3),
        avg_ttft_s=round(statistics.mean(ttfts), 3),
        total_cached_tokens=sum(r.cached_tokens for r in results),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        throughput_req_per_s=round(len(results) / wall_time, 2),
        requests=[r for r in results],
    )


async def run_benchmark(base_url: str, model: str, label: str):
    """Run the full benchmark suite."""
    all_results = []

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ── Warmup: cache the prefix on one worker ──────────────────────
        print(f"[warmup] Caching shared prefix on one worker...")
        warmup = await chat_completion_non_streaming(
            session, base_url, model,
            SHARED_PREFIX + "Summarize in one sentence.",
            max_tokens=32,
        )
        print(f"  {warmup.prompt_tokens} prompt tokens, {warmup.cached_tokens} cached, {warmup.latency_s:.2f}s")
        # Small delay to let router state settle
        await asyncio.sleep(1)

        # ── Scenario 1: Burst (concurrency=16) ──────────────────────────
        print(f"\n[scenario 1] Burst: 64 shared-prefix requests, concurrency=16")
        prompts_s1 = [SHARED_PREFIX + SUFFIXES[i % len(SUFFIXES)] for i in range(64)]
        s1 = await run_scenario(
            session, base_url, model,
            "burst_c16_64req", prompts_s1, concurrency=16, max_tokens=64,
        )
        all_results.append(s1)
        print(f"  wall={s1.wall_time_s}s avg={s1.avg_latency_s}s p50={s1.p50_latency_s}s "
              f"p90={s1.p90_latency_s}s cached={s1.total_cached_tokens}")

        await asyncio.sleep(2)

        # ── Scenario 2: High-concurrency burst (concurrency=32) ────────
        print(f"\n[scenario 2] High-concurrency burst: 128 requests, concurrency=32")
        prompts_s2 = [SHARED_PREFIX + SUFFIXES[i % len(SUFFIXES)] for i in range(128)]
        s2 = await run_scenario(
            session, base_url, model,
            "burst_c32_128req", prompts_s2, concurrency=32, max_tokens=64,
        )
        all_results.append(s2)
        print(f"  wall={s2.wall_time_s}s avg={s2.avg_latency_s}s p50={s2.p50_latency_s}s "
              f"p90={s2.p90_latency_s}s cached={s2.total_cached_tokens}")

        await asyncio.sleep(2)

        # ── Scenario 3: Sequential requests (control, no load pressure) ─
        print(f"\n[scenario 3] Sequential: 16 requests, concurrency=1")
        prompts_s3 = [SHARED_PREFIX + s for s in SUFFIXES[:16]]
        s3 = await run_scenario(
            session, base_url, model,
            "sequential_16req", prompts_s3, concurrency=1, max_tokens=64,
        )
        all_results.append(s3)
        print(f"  wall={s3.wall_time_s}s avg={s3.avg_latency_s}s p50={s3.p50_latency_s}s "
              f"p90={s3.p90_latency_s}s cached={s3.total_cached_tokens}")

        await asyncio.sleep(2)

        # ── Scenario 4: Sustained load (5 waves of 16, concurrency=16) ─
        print(f"\n[scenario 4] Sustained: 5 waves of 16 requests, concurrency=16")
        all_wave_results = []
        for wave in range(5):
            offset = (wave * 16) % len(SUFFIXES)
            prompts_w = [SHARED_PREFIX + SUFFIXES[(offset + i) % len(SUFFIXES)] for i in range(16)]
            wr = await run_scenario(
                session, base_url, model,
                f"sustained_wave{wave}", prompts_w, concurrency=16, max_tokens=64,
            )
            all_wave_results.append(wr)
            print(f"  wave {wave}: wall={wr.wall_time_s}s avg={wr.avg_latency_s}s cached={wr.total_cached_tokens}")
            await asyncio.sleep(1)

        # Aggregate sustained results
        all_sustained_requests = []
        for wr in all_wave_results:
            all_sustained_requests.extend(wr.requests)
        sustained_latencies = sorted([r.latency_s for r in all_sustained_requests])
        total_wall = sum(wr.wall_time_s for wr in all_wave_results)

        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        s4 = ScenarioResult(
            name="sustained_5waves",
            num_requests=len(all_sustained_requests),
            concurrency=16,
            wall_time_s=round(total_wall, 3),
            avg_latency_s=round(statistics.mean(sustained_latencies), 3),
            p50_latency_s=round(percentile(sustained_latencies, 50), 3),
            p90_latency_s=round(percentile(sustained_latencies, 90), 3),
            p99_latency_s=round(percentile(sustained_latencies, 99), 3),
            min_latency_s=round(min(sustained_latencies), 3),
            max_latency_s=round(max(sustained_latencies), 3),
            avg_ttft_s=round(statistics.mean([r.ttft_s for r in all_sustained_requests]), 3),
            total_cached_tokens=sum(r.cached_tokens for r in all_sustained_requests),
            total_prompt_tokens=sum(r.prompt_tokens for r in all_sustained_requests),
            total_completion_tokens=sum(r.completion_tokens for r in all_sustained_requests),
            throughput_req_per_s=round(len(all_sustained_requests) / total_wall, 2),
        )
        all_results.append(s4)
        print(f"  total: wall={s4.wall_time_s}s avg={s4.avg_latency_s}s p90={s4.p90_latency_s}s "
              f"cached={s4.total_cached_tokens}")

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "label": label,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": [],
    }
    for s in all_results:
        d = asdict(s)
        del d["requests"]  # Don't save per-request details to keep JSON small
        output["scenarios"].append(d)

    outfile = f"/home/ubuntu/stack/bench_results_{label}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")

    return output


def main():
    parser = argparse.ArgumentParser(description="KV transfer benchmark")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--label", required=True, help="Label for this run (e.g. 'with-transfer', 'no-transfer')")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.model, args.label))


if __name__ == "__main__":
    main()
