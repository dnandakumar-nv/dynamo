#!/usr/bin/env bash
# Qwen3-30B-A3B on 4 GPUs with KV cache transfer enabled.
# Two TP=2 workers with cross-worker KV block transfer via NIXL RDMA.
# Edit the config below, then: ./dynamo_stack.sh
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
MEM_FRACTION=0.7

# KV Transfer tuning                              # <<< NEW
TRANSFER_COST_WEIGHT=0.1                           # <<< NEW
MIN_TRANSFER_QUEUE_ADVANTAGE=4                     # <<< NEW
MAX_TRANSFER_BLOCKS=256                            # <<< NEW
TRANSFER_TIMEOUT_MS=5000                           # <<< NEW

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
    --enable-kv-transfer \
    --transfer-cost-weight $TRANSFER_COST_WEIGHT \
    --min-transfer-queue-advantage $MIN_TRANSFER_QUEUE_ADVANTAGE \
    --max-transfer-blocks $MAX_TRANSFER_BLOCKS \
    2>&1 | tee -a "$LOGFILE" &
PIDS+=($!)

# ── Workers (2 copies, TP=2 each) ────────────────────────────────────────────
for WORKER in 0 1 2 3; do
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
        --enable-kv-transfer \
        --transfer-timeout-ms $TRANSFER_TIMEOUT_MS \
        2>&1 | tee -a "$LOGFILE" &
    PIDS+=($!)
done

echo "Ctrl+C to stop. Log: $LOGFILE"
wait