#!/bin/bash
# The study: when do images cost a lot in a handover, and when not? Two workloads, both on the
# nuScenes handover traces, each with its text-only twin (0 images, same text):
#
#   front     front camera only, 10 turns; 0, 1, 3, 6, 10 of them carry an image
#   multicam  3 turns, each with 0, 1, 3 or 6 surround cameras (DriveLM-style questions)
#
# and three factors on top: cell size, other users on the cell, image resolution.
# Submit one job per sweep; they can run at the same time on different nodes:
#
#   sbatch --export=ALL,SWEEP=front    scripts/study.slurm
#   sbatch --export=ALL,SWEEP=multicam scripts/study.slurm
#   sbatch --export=ALL,SWEEP=cells    scripts/study.slurm     # 50 and 150 m cells (or cells50 / cells150 separately)
#   sbatch --export=ALL,SWEEP=load     scripts/study.slurm     # 4 other users on each cell
#   sbatch --export=ALL,SWEEP=resolution scripts/study.slurm   # front images at 800 px
#
# Models: default Qwen2-VL-7B (32k context: multicam runs with 1024 px images so 18 images fit).
# For full-resolution multicam use Qwen3-VL-8B (long context):
#   sbatch --export=ALL,SWEEP=multicam,MODEL=Qwen/Qwen3-VL-8B-Instruct,TAG=qwen3vl,MM_KWARGS=,MC_SIZE=0,MAX_LEN=65536 scripts/study.slurm
# Then:  python3 scripts/summarize.py "results/qwen2vl_*.csv" --out results/summary_qwen2vl
#SBATCH -J pallas_study
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o study.%j.out

module load tacc-apptainer
SIF=${SIF:-$SCRATCH/vllm-0.19.1.sif}
export MODEL=${MODEL:-Qwen/Qwen2-VL-7B-Instruct}
export TAG=${TAG:-qwen2vl}
export SWEEP=${SWEEP:-front}
export MM_KWARGS=${MM_KWARGS-'{"max_pixels": 3211264}'}
export MC_SIZE=${MC_SIZE:-1024}            # multicam image size (px, longest side); 0 = original
export MAX_LEN=${MAX_LEN:-32768}
export IMAGES=${IMAGES:-./imgs6}
export EVENTS=${EVENTS:-20} REPEATS=${REPEATS:-3}
export POLICIES=${POLICIES:-pallas recompute fullcopy} BANDWIDTH=${BANDWIDTH:-100 300 500}
cd "$WORK/pallas_baseline"
echo "sweep $SWEEP, model $MODEL, policies $POLICIES, bandwidths $BANDWIDTH, $(date)"

for v in MODEL TAG SWEEP EVENTS REPEATS MC_SIZE MAX_LEN IMAGES MM_KWARGS POLICIES BANDWIDTH; do export APPTAINERENV_$v="${!v}"; done
export APPTAINERENV_WORKDIR=$PWD APPTAINERENV_MAX_IMAGES=24 APPTAINERENV_HF_HOME=$SCRATCH/hf APPTAINERENV_HF_HUB_OFFLINE=1 \
       APPTAINERENV_VLLM_CACHE_ROOT=$SCRATCH/vllm_cache APPTAINERENV_TRITON_CACHE_DIR=$SCRATCH/triton
apptainer exec --nv --cleanenv --bind /work,/scratch \
  "$SIF" bash -c '
set -u
cd "$WORKDIR"
GPU_SRC=0 GPU_TGT=1 bash scripts/launch.sh || exit 1
run () {   # $1 = file tag, rest = arguments
  local t=$1; shift
  echo; echo "######## $t: $*"
  python3 run_pallas.py --images $IMAGES --traces traces/cvh_mini.jsonl --events $EVENTS --repeats $REPEATS \
      --policies $POLICIES --bandwidth $BANDWIDTH \
      --out results/${TAG}_${SWEEP}_${t}.csv "$@" || echo "FAILED: $t"
}
front () { run "$1" --turns 10 --image-turns "$2" --cameras 1 "${@:3}"; }
multi () { local c=$2; [ "$c" = 0 ] && run "$1" --turns 3 --image-turns 0 "${@:3}" \
                                     || run "$1" --turns 3 --image-turns 3 --cameras "$c" --image-size $MC_SIZE "${@:3}"; }
case $SWEEP in
  front)      for k in 0 1 3 6 10; do front k${k} $k --cell 100; done ;;
  multicam)   for c in 0 1 3 6; do multi cam${c} $c --cell 100; done ;;
  cells)      for m in 50 150; do
                for k in 0 3 10; do front c${m}_k${k} $k --cell $m; done
                for c in 0 6;    do multi c${m}_cam${c} $c --cell $m; done
              done ;;
  cells50|cells150) m=${SWEEP#cells}
              for k in 0 3 10; do front c${m}_k${k} $k --cell $m; done
              for c in 0 6;    do multi c${m}_cam${c} $c --cell $m; done ;;
  load)       for k in 0 3 10; do front load4_k${k} $k --cell 100 --cell-load 4; done
              for c in 0 6;    do multi load4_cam${c} $c --cell 100 --cell-load 4; done ;;
  resolution) for k in 3 10; do front px800_k${k} $k --cell 100 --image-size 800; done ;;
  *)          echo "unknown SWEEP=$SWEEP"; exit 1 ;;
esac
'
echo "done $(date)"
