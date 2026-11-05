#!/bin/bash
#SBATCH -J pallas_check
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o check.%j.out

module load tacc-apptainer
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
SIF=$SCRATCH/vllm-0.19.1.sif
if [ ! -f "$SIF" ]; then
  echo "== pulling vLLM 0.19.1 image (first time only, ~10-15 min)"
  apptainer pull "$SIF" docker://vllm/vllm-openai:v0.19.1 || exit 1
fi

run () {   # run a command inside the container, with the GPUs, /work and /scratch
  apptainer exec --nv --cleanenv --bind /work,/scratch \
    --env HF_HOME=$SCRATCH/hf,HF_HUB_OFFLINE=1,VLLM_CACHE_ROOT=$SCRATCH/vllm_cache,TRITON_CACHE_DIR=$SCRATCH/triton \
    "$SIF" bash -c "cd $WORK/pallas_baseline && $1"
}

echo "== GPUs"; nvidia-smi | head -12
run 'python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.version.cuda)"'

echo "== start cells"
run 'MODEL=Qwen/Qwen2-VL-7B-Instruct GPU_SRC=0 GPU_TGT=1 MM_KWARGS='"'"'{"max_pixels": 3211264}'"'"' bash scripts/launch.sh \
     && grep -E "pallas: .* KV layers|Asynchronous scheduling|hybrid kv cache" logs/source.log \
     && echo "== Full-Copy check" \
     && python3 run_pallas.py --images ./imgs --turns 2 --policies fullcopy --bandwidth 1000 --repeats 1 --out results/check.csv \
     && echo "== short Pallas run with CVH" \
     && python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh_mini.jsonl --cell 100 --events 3 --bandwidth 300 --repeats 1 --out results/try.csv'
