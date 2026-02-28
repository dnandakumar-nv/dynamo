#!/usr/bin/env bash
# Nemotron-3-Nano-30B on 4 GPUs with Dynamo frontend, HiCache, and PIN.
# Two TP=2 workers enable cache-aware routing via the KV router.
# Edit the config below, then: ./dynamo-stack.sh
# Ctrl+C to stop. Logs in /tmp/dynamo-stack/
#
# Prerequisites (run once, stays up):
#   cd dynamo/deploy
#   docker compose -f docker-compose.yml up -d --remove-orphans
#   docker compose -f docker-observability.yml up -d --remove-orphans
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PAGE_SIZE=16
HICACHE_RATIO=1.0
HICACHE_POLICY=write_through
CONTEXT_LENGTH=262144
MEM_FRACTION=0.9

LOG_DIR="/tmp/dynamo-stack"

# ── Cleanup ──────────────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "Done. Logs in $LOG_DIR/"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"

# ── Preflight ────────────────────────────────────────────────────────────────
curl -sf http://localhost:2379/health >/dev/null 2>&1 || { echo "etcd not running. See header comment."; exit 1; }
curl -sf http://localhost:8222/healthz >/dev/null 2>&1 || { echo "NATS not running. See header comment."; exit 1; }

LOGFILE="$LOG_DIR/all.log"
> "$LOGFILE"

# ── Frontend ─────────────────────────────────────────────────────────────────
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --http-port 8001 \
    --router-mode kv \
    --router-reset-states \
    --enable-cache-control \
    2>&1 | tee -a "$LOGFILE" &
PIDS+=($!)

# ── Workers (2 copies, TP=2 each) ────────────────────────────────────────────
for WORKER in 0 1; do
    PORT=$((8081 + WORKER))
    ZMQ_PORT=$((20080 + WORKER))
    GPU_START=$((WORKER * 2))
    GPUS="$GPU_START,$((GPU_START + 1))"

    CUDA_VISIBLE_DEVICES=$GPUS \
    OTEL_SERVICE_NAME=dynamo-worker-$WORKER \
    DYN_SYSTEM_PORT=$PORT \
    python3 -m dynamo.sglang \
        --model-path "$MODEL" \
        --served-model-name "$MODEL" \
        --page-size $PAGE_SIZE \
        --tp 2 \
        --mem-fraction-static $MEM_FRACTION \
        --context-length $CONTEXT_LENGTH \
        --trust-remote-code \
        --dyn-reasoning-parser deepseek_r1 \
        --dyn-tool-call-parser hermes \
        --enable-metrics \
        --schedule-low-priority-values-first \
        --enable-priority-scheduling \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$ZMQ_PORT"'"}' \
        2>&1 | tee -a "$LOGFILE" &
    PIDS+=($!)
done

echo "Ctrl+C to stop. Log: $LOGFILE"
wait
