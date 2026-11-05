#!/bin/bash
#SBATCH -J pallas_study
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o study.%j.out

module load tacc-apptainer
SIF=$SCRATCH/vllm-0.19.1.sif
MODEL=${MODEL:-Qwen/Qwen2-VL-7B-Instruct}
TAG=${TAG:-qwen2vl}
apptainer exec --nv --cleanenv --bind /work,/scratch \
  --env HF_HOME=$SCRATCH/hf,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$SCRATCH/vllm_cache,TRITON_CACHE_DIR=$SCRATCH/triton,MODEL=$MODEL,TAG=$TAG \
  "$SIF" bash -c '
cd '"$WORK"'/pallas_baseline
MM_KWARGS="{\"max_pixels\": 3211264}" GPU_SRC=0 GPU_TGT=1 bash scripts/launch.sh || exit 1
for c in 50 100 150; do
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh_mini.jsonl --cell $c --events 30 \
      --out results/${TAG}_img_${c}m.csv
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh_mini.jsonl --cell $c --events 30 \
      --no-images --out results/${TAG}_txt_${c}m.csv
done'
