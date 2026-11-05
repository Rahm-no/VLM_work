#!/usr/bin/env bash
# Two cells (one GPU each) and the link between them.
#   MODEL=Qwen/Qwen2-VL-7B-Instruct GPU_SRC=0 GPU_TGT=1 bash scripts/launch.sh
# Run from the repository root. Logs in logs/.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2-VL-7B-Instruct}
GPU_SRC=${GPU_SRC:-0}
GPU_TGT=${GPU_TGT:-1}
MBPS=${MBPS:-300}
MAX_LEN=${MAX_LEN:-32768}          # 10 images of ~1,800 tokens + text
MAX_IMAGES=${MAX_IMAGES:-12}
MEM=${MEM:-0.85}
MM_KWARGS=${MM_KWARGS:-}            # e.g. '{"max_pixels": 3211264}' for Qwen2-VL (4096 image tokens)
EXTRA=(--max-images "$MAX_IMAGES")
[ -n "$MM_KWARGS" ] && EXTRA+=(--mm-processor-kwargs "$MM_KWARGS")

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

# source cell (8100 control, 9101 data), target cell (8200 control, 9201 data)
# each cell gets its own compile caches: two cells compiling at once into one cache dir race
# (FileNotFoundError on a .cubin the other cell is still writing)
TRITON_BASE=${TRITON_CACHE_DIR:-$HOME/.triton/cache}
VLLM_BASE=${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}
TRITON_CACHE_DIR=$TRITON_BASE/source VLLM_CACHE_ROOT=$VLLM_BASE/source \
CUDA_VISIBLE_DEVICES=$GPU_SRC nohup python3 -m pallas.cell --name source --model "$MODEL" \
  --http-port 8100 --data-port 9101 --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM" "${EXTRA[@]}" \
  > logs/source.log 2>&1 &
PID_SRC=$!
TRITON_CACHE_DIR=$TRITON_BASE/target VLLM_CACHE_ROOT=$VLLM_BASE/target \
CUDA_VISIBLE_DEVICES=$GPU_TGT nohup python3 -m pallas.cell --name target --model "$MODEL" \
  --http-port 8200 --data-port 9201 --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM" "${EXTRA[@]}" \
  > logs/target.log 2>&1 &
PID_TGT=$!
# link: source sends to 9100, relay forwards to the target's data port at $MBPS
nohup python3 -m pallas.link --listen 127.0.0.1:9100 --to 127.0.0.1:9201 --mbps "$MBPS" --stats-port 9199 \
  > logs/link.log 2>&1 &

# link to a cell outside the experiment: where a wrong-target prediction sends the state
nohup python3 -m pallas.link --listen 127.0.0.1:9300 --to sink --mbps "$MBPS" --stats-port 9399 \
  > logs/link_wrong_target.log 2>&1 &

echo "waiting for the cells..."
for pp in "8100 $PID_SRC source" "8200 $PID_TGT target"; do
  set -- $pp
  until curl -sf "http://127.0.0.1:$1/info" > /dev/null; do
    if ! kill -0 "$2" 2>/dev/null; then echo "$3 cell exited; last lines of logs/$3.log:"; tail -30 "logs/$3.log"; exit 1; fi
    sleep 5
  done
done
echo "ready: source :8100, target :8200, link :9100 ($MBPS Mbps, control :9199), wrong-target link :9300"