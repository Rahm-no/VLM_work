# Pallas baseline on vision-language models

A reimplementation of **Pallas** (Ding, Liu, Xu, *A Proactive KV Cache Migration Framework for LLM
Inference in AI-RAN*, arXiv 2608.16477) run unchanged on VLMs (Qwen2-VL, Qwen3-VL, Gemma 3).
The goal is to measure how the method behaves when sessions contain images.

Each cell is a vLLM engine on its own GPU. KV blocks are copied out of the source's paged KV cache,
sent over a throttled TCP link, and injected into the target's cache. Handovers are replayed from
nuScenes paths with CVH predictions. Baselines: Recomputation and Full-Copy.

## Layout

| Path | What |
|---|---|
| `pallas/model.py` | Cost model and window choice (Eqs. 1–9, Algorithm 1) |
| `pallas/controller.py` | Control loop, policies, measurements |
| `pallas/connector.py` | vLLM KV connector (stream from source, inject at target) |
| `pallas/cell.py`, `rx.py` | A cell: vLLM engine + control API + link receiver |
| `pallas/link.py`, `wire.py` | Throttled link and KV block packing |
| `pallas/mobility.py`, `bookkeeping.py` | Mobility/CVH traces, block planning |
| `run_pallas.py` | Experiment CLI |
| `scripts/` | Launch, SLURM jobs, trace and image prep, summary |
| `tests/` | Unit and end-to-end tests (no GPU needed) |
| `results/` | Study outputs (CSV) and summary |

## Run (2 GPUs)

```bash
pip install vllm fastapi uvicorn requests pillow numpy

# start two cells and the link
MODEL=Qwen/Qwen2-VL-7B-Instruct GPU_SRC=0 GPU_TGT=1 bash scripts/launch.sh

# sanity check: same_next_token should be True
python3 run_pallas.py --images ./imgs --turns 2 --policies fullcopy --bandwidth 1000 --repeats 1 --out results/check.csv

# full study, one SLURM job per sweep
for s in front multicam cells load resolution; do sbatch --export=ALL,SWEEP=$s scripts/study.slurm; done
python3 scripts/summarize.py "results/qwen2vl_*.csv" --out results/summary_qwen2vl
```

Images are not included. Pick them from nuScenes with `scripts/pick_multiview.py`.

## Tests

```bash
python3 -m pytest -q tests
python3 tests/run_fake_system.py
```

## Differences from the paper

- A recent vLLM version instead of 0.8.5, and A100 GPUs instead of RTX A6000.
- Both cells run on one node, and the link between them is emulated.
- Migrated blocks are injected when the target creates the request, not into a running request.
- With only two GPU cells, a wrong-target trigger sends its data to a sink.
- Parameters the paper does not give (T_max, ΔT, EWMA factor, etc.) are command-line flags.
