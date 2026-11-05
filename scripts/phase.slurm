#!/bin/bash
# How long does the vision encoder take? Per image: preprocessing, vision encoder, LLM prefill,
# decode per token; and text-only prefill speed at several lengths, for comparison.
# One GPU is enough (gpu-a100-small queue):
#
#   sbatch scripts/phase.slurm                                            # Qwen2-VL, front images
#   sbatch --export=ALL,COUNTS="1 3 6 10",IMAGES=./imgs6,MAX_LEN=32768,TAG=qwen2vl_multi scripts/phase.slurm  # many images per prompt
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-VL-8B-Instruct,TAG=qwen3vl,MAX_PIXELS=0 scripts/phase.slurm
#SBATCH -J phase_bench
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o phase.%j.out

module load tacc-apptainer
SIF=${SIF:-$SCRATCH/vllm-0.19.1.sif}
MODEL=${MODEL:-Qwen/Qwen2-VL-7B-Instruct}
TAG=${TAG:-qwen2vl}
IMAGES=${IMAGES:-./imgs}
MAX_PIXELS=${MAX_PIXELS:-3211264}
COUNTS=${COUNTS:-}
MAX_LEN=${MAX_LEN:-16384}
cd "$WORK/pallas_baseline"
mkdir -p results
export APPTAINERENV_HF_HOME=$SCRATCH/hf APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       APPTAINERENV_VLLM_CACHE_ROOT=$SCRATCH/vllm_cache APPTAINERENV_TRITON_CACHE_DIR=$SCRATCH/triton
apptainer exec --nv --cleanenv --bind /work,/scratch "$SIF" \
  python3 scripts/phase_bench.py --model "$MODEL" --images "$IMAGES" --max-pixels "$MAX_PIXELS" --counts "$COUNTS" --max-len "$MAX_LEN" \
      --text-lengths 512,1024,2048,4096,8192 --out "results/phases_${TAG}.csv"
