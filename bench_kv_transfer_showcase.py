#!/usr/bin/env python3
"""Benchmark showcasing cross-worker KV transfer with mixed output lengths.

The scenario: requests share a large prefix (high KV cache match) but have
varying output sequence lengths. Without KV transfer, they all pile on the
worker that cached the prefix — short-output requests wait behind long ones.
With KV transfer + load-aware routing, long-output requests get moved to
other workers (paying a small KV transfer cost), unblocking short requests.

The key metric: latency of short-OSL requests.

Usage:
    python bench_kv_transfer_showcase.py --label baseline
    python bench_kv_transfer_showcase.py --label kv-enabled
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import aiohttp

# ---------------------------------------------------------------------------
# Large shared prefix (~2000 tokens after tokenization).
# This will get cached on whichever worker first processes a request.
# ---------------------------------------------------------------------------

SHARED_PREFIX = """You are a highly knowledgeable technical assistant. Below is a detailed \
technical document that you must use to answer the question at the end. \
Read the entire document carefully before answering.

DOCUMENT: Comprehensive Guide to Distributed Consensus Protocols

Chapter 1: Foundations of Distributed Systems
A distributed system is a collection of independent computers that appears to its users as a single coherent system. The fundamental challenges include partial failures, where some components fail while others continue operating, and the absence of a global clock, making it impossible to determine the exact ordering of events across nodes.

The CAP theorem, formulated by Eric Brewer in 2000 and later proved by Gilbert and Lynch, states that a distributed system can provide at most two of three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a non-error response), and Partition Tolerance (the system operates despite network partitions). In practice, network partitions are inevitable in distributed systems, so the real choice is between consistency and availability during partitions.

The FLP impossibility result, published by Fischer, Lynch, and Paterson in 1985, proves that in an asynchronous distributed system where even a single process can fail, no deterministic protocol can guarantee consensus. This fundamental result shapes all practical consensus algorithms, which must either use timeouts (breaking pure asynchrony), randomization, or failure detectors to circumvent FLP.

Chapter 2: Paxos and Its Variants
Paxos, invented by Leslie Lamport, is the foundational consensus algorithm. The basic protocol operates in two phases. In Phase 1 (Prepare), a proposer selects a proposal number n and sends a prepare request to a majority of acceptors. Each acceptor responds with a promise not to accept proposals numbered less than n, along with any previously accepted proposal. In Phase 2 (Accept), if the proposer receives promises from a majority, it sends an accept request with the proposal number and value. The value is either from the highest-numbered previously accepted proposal or a new value if no prior proposals exist.

Multi-Paxos optimizes for repeated consensus by establishing a stable leader. Once a leader is elected, it can skip Phase 1 for subsequent proposals, reducing the protocol to a single round trip. This optimization is critical for practical systems that need to agree on a sequence of values (like a replicated log).

Fast Paxos, proposed by Lamport in 2006, reduces latency by allowing clients to send proposals directly to acceptors, bypassing the leader. This achieves two message delays in the fast path but requires a larger quorum (three-fourths of acceptors instead of a majority) and falls back to classic Paxos when conflicts occur.

Flexible Paxos, discovered by Howard et al. in 2016, relaxes the quorum intersection requirement. The key insight is that Phase 1 and Phase 2 quorums only need to intersect, rather than each being a majority. This allows configurations like using a large Phase 1 quorum (for leader election, which is rare) and a small Phase 2 quorum (for replication, which is frequent), improving throughput at the cost of leader election latency.

Chapter 3: Raft Consensus Algorithm
Raft was designed by Diego Ongaro and John Ousterhout in 2014 with the explicit goal of understandability. Raft decomposes consensus into three sub-problems: leader election, log replication, and safety.

Leader Election: Raft uses randomized election timeouts to minimize split votes. When a follower's election timeout fires without receiving a heartbeat from the leader, it increments its term, transitions to candidate state, votes for itself, and sends RequestVote RPCs to all other servers. A candidate wins if it receives votes from a majority. The randomized timeout (typically 150-300ms) ensures that in most cases only a single server times out and wins the election.

Log Replication: The leader accepts client requests, appends entries to its log, and replicates them to followers via AppendEntries RPCs. An entry is committed once replicated to a majority. The leader tracks the highest committed index and communicates it in subsequent AppendEntries RPCs, allowing followers to apply committed entries to their state machines.

Safety: Raft ensures safety through the Election Safety property (at most one leader per term), Leader Append-Only (leaders never overwrite or delete entries), Log Matching (if two logs contain an entry with the same index and term, all preceding entries are identical), and Leader Completeness (if an entry is committed in a given term, it will be present in the logs of leaders for all higher terms).

Chapter 4: Byzantine Fault Tolerance
Byzantine fault tolerance (BFT) addresses the scenario where nodes may behave arbitrarily, including sending contradictory messages to different peers. The classic result by Lamport, Shostak, and Pease shows that consensus requires at least 3f+1 nodes to tolerate f Byzantine faults.

PBFT (Practical Byzantine Fault Tolerance), proposed by Castro and Liskov in 1999, was the first practical BFT algorithm for asynchronous systems. PBFT operates in three phases: pre-prepare (leader assigns sequence number), prepare (nodes verify and broadcast), and commit (nodes agree on ordering). PBFT achieves O(n^2) message complexity, which limits scalability beyond approximately 20 nodes.

HotStuff, developed by Yin et al. in 2019, improves on PBFT by achieving linear message complexity through a three-phase protocol with threshold signatures. HotStuff's key innovation is the use of a pacemaker for view synchronization and the ability to pipeline phases across different consensus instances. HotStuff forms the basis of several blockchain consensus protocols, including Facebook's LibraBFT.

Chapter 5: Consensus in Modern Systems
Modern distributed databases and coordination services use consensus in various ways. Google's Spanner uses a Multi-Paxos variant called TrueTime to achieve external consistency across globally distributed data centers. The TrueTime API provides bounded clock uncertainty, allowing Spanner to assign globally meaningful timestamps to transactions.

Apache ZooKeeper uses a protocol called Zab (ZooKeeper Atomic Broadcast) that is similar to Paxos but optimized for the primary-backup model. Zab guarantees that all changes are delivered in the order they were applied by the primary, making it suitable for configuration management and distributed coordination.

etcd, the distributed key-value store used by Kubernetes, implements Raft for consensus. etcd's Raft implementation includes optimizations like pre-vote (to prevent disruption from partitioned nodes), learner nodes (for safe cluster membership changes), and leadership transfer (for graceful leader handoff during maintenance).

CockroachDB uses a Multi-Raft approach where different ranges of the keyspace have independent Raft groups. This allows CockroachDB to scale horizontally by distributing Raft leadership across nodes, avoiding the single-leader bottleneck of a monolithic Raft group.

Chapter 6: Performance Optimization Techniques
Batching is a fundamental optimization: instead of running consensus for each individual request, batch multiple requests into a single consensus round. This amortizes the per-round overhead across many requests. Systems like EPaxos and Mencius allow concurrent proposers to improve throughput by reducing leader bottleneck.

Pipelining allows a leader to initiate new consensus rounds before previous rounds complete. This overlaps the latency of multiple rounds, improving throughput when round-trip times are significant. The key constraint is that the number of in-flight rounds must be bounded to prevent unbounded memory growth.

Speculative execution allows followers to speculatively execute commands before they are committed, reducing latency for read-heavy workloads. If speculation is correct (the command commits as expected), the response is already available. If speculation fails (rare), the follower must roll back and re-execute.

Read optimization techniques include leader leases (allowing the leader to serve reads without consensus), follower reads (serving reads from followers with bounded staleness), and witness replicas (lightweight replicas that participate in consensus but don't store data, reducing replication overhead for write-heavy workloads).

Now answer the following question based on the document above.

"""

# ---------------------------------------------------------------------------
# Short-OSL questions: expect brief answers (~32 tokens)
# ---------------------------------------------------------------------------
SHORT_QUESTIONS = [
    "In one sentence, what is the CAP theorem?",
    "Name the three sub-problems Raft decomposes consensus into.",
    "What is the message complexity of PBFT?",
    "How many nodes are needed to tolerate f Byzantine faults?",
    "What does FLP stand for?",
    "What quorum size does Fast Paxos require?",
    "What database uses Multi-Raft?",
    "What is the purpose of the TrueTime API in Spanner?",
    "What protocol does ZooKeeper use for consensus?",
    "What is the main advantage of Flexible Paxos?",
    "What is the randomized timeout range in Raft leader election?",
    "Name one read optimization technique mentioned in Chapter 6.",
    "What year was the FLP result published?",
    "What is the key innovation of HotStuff?",
    "How does Multi-Paxos reduce latency compared to basic Paxos?",
    "What is the purpose of learner nodes in etcd's Raft?",
]

# ---------------------------------------------------------------------------
# Long-OSL questions: expect detailed answers (~512 tokens)
# ---------------------------------------------------------------------------
LONG_QUESTIONS = [
    "Provide a comprehensive comparison of Paxos, Multi-Paxos, Fast Paxos, and Flexible Paxos. For each variant, explain the protocol phases, quorum requirements, latency characteristics, and practical use cases. Include specific examples of systems that use each variant and analyze the tradeoffs between them in terms of throughput, latency, fault tolerance, and implementation complexity.",
    "Write a detailed analysis of Byzantine fault tolerance covering PBFT and HotStuff. Explain each protocol's phases, message complexity, scalability limits, and real-world deployments. Compare their approaches to view synchronization and discuss why HotStuff's linear complexity matters for blockchain consensus. Include the historical evolution from theoretical BFT to practical implementations.",
    "Explain the complete Raft consensus algorithm in detail, covering leader election with randomized timeouts, log replication with AppendEntries RPCs, and all four safety properties. For each property, explain what specific failure scenario it prevents. Then compare Raft's approach to safety with how Paxos achieves the same guarantees, discussing the tradeoffs in understandability versus flexibility.",
    "Analyze all the performance optimization techniques from Chapter 6 (batching, pipelining, speculative execution, leader leases, follower reads, witness replicas) in detail. For each technique, explain the mechanism, when it helps, what workloads benefit most, and what the tradeoffs are. Then describe how these techniques interact with each other and propose an optimal combination for different workload patterns.",
    "Write a comprehensive essay on how consensus protocols are used in modern distributed databases. Cover Google Spanner (TrueTime, external consistency), ZooKeeper (Zab), etcd (Raft with optimizations), and CockroachDB (Multi-Raft). For each system, explain the specific consensus variant used, why it was chosen, and how it addresses the system's scalability and consistency requirements.",
    "Provide a thorough analysis of the CAP theorem and the FLP impossibility result, explaining how they fundamentally constrain distributed system design. Then show how each consensus protocol discussed in the document (Paxos, Raft, PBFT, HotStuff) navigates these constraints. Include specific examples of how each protocol handles network partitions, process failures, and asynchrony.",
    "Discuss the evolution of consensus protocols from Paxos (1989) through PBFT (1999), Raft (2014), Flexible Paxos (2016), to HotStuff (2019). For each milestone, explain what problem it solved, what new capability it enabled, and how it influenced subsequent work. Analyze the trend toward simpler, more practical algorithms and predict future directions.",
    "Compare and contrast the crash fault tolerance model (Paxos, Raft) with the Byzantine fault tolerance model (PBFT, HotStuff). Explain the fundamental differences in assumptions, the impact on quorum sizes and message complexity, and the practical implications for deployment. Discuss hybrid approaches and when each model is appropriate.",
]


@dataclass
class RequestResult:
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_s: float
    osl_class: str  # "short" or "long"


@dataclass
class ScenarioResult:
    name: str
    num_requests: int
    concurrency: int
    wall_time_s: float
    # Overall
    avg_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    p99_latency_s: float
    avg_ttft_s: float
    p50_ttft_s: float
    p90_ttft_s: float
    throughput_req_per_s: float
    total_cached_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    # Per OSL class
    short_osl_count: int
    short_osl_avg_latency_s: float
    short_osl_p50_latency_s: float
    short_osl_p90_latency_s: float
    short_osl_avg_ttft_s: float
    short_osl_p90_ttft_s: float
    long_osl_count: int
    long_osl_avg_latency_s: float
    long_osl_p50_latency_s: float
    long_osl_p90_latency_s: float
    long_osl_avg_ttft_s: float
    long_osl_p90_ttft_s: float
    requests: List[RequestResult] = field(default_factory=list)


def pct(data, p):
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


async def chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int,
    osl_class: str,
) -> RequestResult:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = 0.0
    usage = {}

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
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    return RequestResult(
        latency_s=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=cached,
        ttft_s=ttft if ttft > 0 else elapsed,
        osl_class=osl_class,
    )


def build_result(name, results, concurrency, wall_time):
    lats = [r.latency_s for r in results]
    ttfts = [r.ttft_s for r in results]

    short = [r for r in results if r.osl_class == "short"]
    long = [r for r in results if r.osl_class == "long"]

    s_lats = [r.latency_s for r in short] or [0]
    s_ttfts = [r.ttft_s for r in short] or [0]
    l_lats = [r.latency_s for r in long] or [0]
    l_ttfts = [r.ttft_s for r in long] or [0]

    return ScenarioResult(
        name=name,
        num_requests=len(results),
        concurrency=concurrency,
        wall_time_s=round(wall_time, 3),
        avg_latency_s=round(statistics.mean(lats), 3),
        p50_latency_s=round(pct(lats, 50), 3),
        p90_latency_s=round(pct(lats, 90), 3),
        p99_latency_s=round(pct(lats, 99), 3),
        avg_ttft_s=round(statistics.mean(ttfts), 3),
        p50_ttft_s=round(pct(ttfts, 50), 3),
        p90_ttft_s=round(pct(ttfts, 90), 3),
        throughput_req_per_s=round(len(results) / wall_time, 2),
        total_cached_tokens=sum(r.cached_tokens for r in results),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        short_osl_count=len(short),
        short_osl_avg_latency_s=round(statistics.mean(s_lats), 3),
        short_osl_p50_latency_s=round(pct(s_lats, 50), 3),
        short_osl_p90_latency_s=round(pct(s_lats, 90), 3),
        short_osl_avg_ttft_s=round(statistics.mean(s_ttfts), 3),
        short_osl_p90_ttft_s=round(pct(s_ttfts, 90), 3),
        long_osl_count=len(long),
        long_osl_avg_latency_s=round(statistics.mean(l_lats), 3),
        long_osl_p50_latency_s=round(pct(l_lats, 50), 3),
        long_osl_p90_latency_s=round(pct(l_lats, 90), 3),
        long_osl_avg_ttft_s=round(statistics.mean(l_ttfts), 3),
        long_osl_p90_ttft_s=round(pct(l_ttfts, 90), 3),
        requests=results,
    )


def print_scenario(s: ScenarioResult):
    print(f"  wall={s.wall_time_s}s  throughput={s.throughput_req_per_s} req/s  "
          f"cached={s.total_cached_tokens}")
    print(f"  ALL       : avg_lat={s.avg_latency_s}s  avg_ttft={s.avg_ttft_s}s  "
          f"p90_ttft={s.p90_ttft_s}s")
    print(f"  SHORT ({s.short_osl_count:>2}): avg_lat={s.short_osl_avg_latency_s}s  "
          f"p50_lat={s.short_osl_p50_latency_s}s  p90_lat={s.short_osl_p90_latency_s}s  "
          f"avg_ttft={s.short_osl_avg_ttft_s}s  p90_ttft={s.short_osl_p90_ttft_s}s")
    print(f"  LONG  ({s.long_osl_count:>2}): avg_lat={s.long_osl_avg_latency_s}s  "
          f"p50_lat={s.long_osl_p50_latency_s}s  p90_lat={s.long_osl_p90_latency_s}s  "
          f"avg_ttft={s.long_osl_avg_ttft_s}s  p90_ttft={s.long_osl_p90_ttft_s}s")


async def run_benchmark(
    base_url: str, model: str, label: str,
    short_tokens: int, long_tokens: int,
    num_short: int, num_long: int,
):
    all_results = []

    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ── Warmup: cache the prefix on one worker ───────────────────
        print("[warmup] Caching shared prefix...")
        r = await chat_completion(
            session, base_url, model,
            SHARED_PREFIX + "Say OK.", max_tokens=8, min_tokens=1,
            osl_class="warmup",
        )
        print(f"  {r.prompt_tokens} prompt tokens, {r.cached_tokens} cached, "
              f"{r.latency_s:.2f}s")
        await asyncio.sleep(2)

        # ── Scenario 1: Mixed OSL burst ──────────────────────────────
        # Fire short + long requests together, all sharing the prefix.
        # Without KV transfer: all go to cached worker, shorts wait
        # behind longs. With KV transfer: longs move to other workers.
        total = num_short + num_long
        conc = total
        print(f"\n[scenario 1] Mixed OSL burst: {num_short} short "
              f"({short_tokens} tok) + {num_long} long ({long_tokens} tok) "
              f"= {total} total, concurrency={conc}")

        prompts = []
        for i in range(num_short):
            prompts.append((
                SHARED_PREFIX + SHORT_QUESTIONS[i % len(SHORT_QUESTIONS)],
                short_tokens, short_tokens, "short",
            ))
        for i in range(num_long):
            prompts.append((
                SHARED_PREFIX + LONG_QUESTIONS[i % len(LONG_QUESTIONS)],
                long_tokens, long_tokens, "long",
            ))

        sem = asyncio.Semaphore(conc)

        async def fire(p):
            prompt, maxt, mint, osl = p
            async with sem:
                return await chat_completion(
                    session, base_url, model, prompt, maxt, mint, osl,
                )

        t0 = time.monotonic()
        results = await asyncio.gather(*[fire(p) for p in prompts])
        wall = time.monotonic() - t0

        s1 = build_result(f"mixed_burst_{total}req_c{conc}", list(results), conc, wall)
        all_results.append(s1)
        print_scenario(s1)

        await asyncio.sleep(5)

        # ── Scenario 2: Staggered — longs first, then shorts ────────
        # Fire long requests first to load up the cached worker, then
        # fire shorts 2s later. This is the worst case for no-transfer:
        # the cached worker is fully occupied decoding longs.
        print(f"\n[scenario 2] Staggered: {num_long} long first, "
              f"then {num_short} short after 2s delay")

        long_prompts = []
        for i in range(num_long):
            long_prompts.append((
                SHARED_PREFIX + LONG_QUESTIONS[i % len(LONG_QUESTIONS)],
                long_tokens, long_tokens, "long",
            ))
        short_prompts = []
        for i in range(num_short):
            short_prompts.append((
                SHARED_PREFIX + SHORT_QUESTIONS[i % len(SHORT_QUESTIONS)],
                short_tokens, short_tokens, "short",
            ))

        sem2 = asyncio.Semaphore(total)
        all_s2 = []

        async def fire2(p):
            prompt, maxt, mint, osl = p
            async with sem2:
                return await chat_completion(
                    session, base_url, model, prompt, maxt, mint, osl,
                )

        t0 = time.monotonic()
        # Fire longs
        long_tasks = [asyncio.create_task(fire2(p)) for p in long_prompts]
        # Wait for longs to saturate the cached worker's decode batch
        await asyncio.sleep(2)
        # Now fire shorts — cached worker is busy decoding longs
        short_tasks = [asyncio.create_task(fire2(p)) for p in short_prompts]
        # Wait for everything
        long_results = await asyncio.gather(*long_tasks)
        short_results = await asyncio.gather(*short_tasks)
        wall2 = time.monotonic() - t0

        all_s2 = list(long_results) + list(short_results)
        s2 = build_result(
            f"staggered_{total}req", all_s2, total, wall2,
        )
        all_results.append(s2)
        print_scenario(s2)

        await asyncio.sleep(5)

        # ── Scenario 3: Repeat burst (cache should be warm everywhere)
        # After scenarios 1 & 2, KV blocks may have been transferred to
        # multiple workers. This tests whether the system benefits from
        # accumulated cross-worker cache.
        print(f"\n[scenario 3] Repeat mixed burst (post-warmup): {total} req")

        t0 = time.monotonic()
        results3 = await asyncio.gather(*[fire(p) for p in prompts])
        wall3 = time.monotonic() - t0

        s3 = build_result(f"repeat_burst_{total}req", list(results3), conc, wall3)
        all_results.append(s3)
        print_scenario(s3)

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "label": label,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "short_tokens": short_tokens,
            "long_tokens": long_tokens,
            "num_short": num_short,
            "num_long": num_long,
        },
        "scenarios": [],
    }
    for s in all_results:
        d = asdict(s)
        del d["requests"]
        output["scenarios"].append(d)

    outfile = f"/home/ubuntu/stack/bench_results_{label}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(description="KV transfer showcase benchmark")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--label", required=True)
    parser.add_argument("--short-tokens", type=int, default=32,
                        help="Output length for short requests (default: 32)")
    parser.add_argument("--long-tokens", type=int, default=512,
                        help="Output length for long requests (default: 512)")
    parser.add_argument("--num-short", type=int, default=16,
                        help="Number of short-OSL requests (default: 16)")
    parser.add_argument("--num-long", type=int, default=8,
                        help="Number of long-OSL requests (default: 8)")
    args = parser.parse_args()
    asyncio.run(run_benchmark(
        args.base_url, args.model, args.label,
        args.short_tokens, args.long_tokens,
        args.num_short, args.num_long,
    ))


if __name__ == "__main__":
    main()
