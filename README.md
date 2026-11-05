# Pallas on vision-language models (baseline)

A reimplementation of **Pallas** (Ding, Liu, Xu, *A Proactive KV Cache Migration Framework for LLM
Inference in AI-RAN*, arXiv 2608.16477; no public code), run unchanged on VLMs (Qwen2-VL, Gemma 3) to
measure how the state of the art behaves when sessions contain images. No VLM-specific optimisation.

Real inference, real KV, real bytes on a link:
- each cell is a vLLM engine on its own GPU;
- the source really decodes during migration;
- the suffix KV blocks are really copied out of the source's paged KV cache, streamed over a
  throttled TCP link, and injected into the target's paged KV cache;
- the target really prefills the prefix (encoding the images);
- SIT is measured on the clock.

## What Pallas does, and where it is

| Pallas (paper) | Here |
|---|---|
| KV size per token, Eq. 1 | `pallas/model.py: kv_bytes_per_token` (layers, KV heads, head dim read from the model config) |
| Cost model, Eqs. 3 to 9; window choice, Algorithm 1 (α = 0.8, W = {0, ΔT, …, T_limit}) | `pallas/model.py: choose_window, should_trigger` |
| Mobility: nuScenes paths on 50/100/150 m grids, CVH predicts the next cell and the handover time | `pallas/mobility.py`, `scripts/cvh_traces.py` (one prediction trace per true handover) |
| Control loop: every control cycle, read the prediction, run Algorithm 1 with the *predicted* T_remain; trigger toward the *predicted* target when T_remain ≤ T_w* | `pallas/controller.py: _control_loop` replays a trace in real time; the real handover happens at the true time |
| Prediction changes: cancel the attempt (epoch), start a new one for the new target | `/cancel`, new trigger with a new epoch (`bookkeeping.apply_trigger`); a wrong target receives the state over its own link (`pallas.link --to sink`) |
| EWMA-smoothed r_p, r_d, B | r_p probed on the target, then updated from each observed prefix prefill; r_d from the source's token stream; B probed end to end over the link (`controller.py`) |
| Trigger: snap to the last complete block (16 tokens); prefix [0, k) sent as exact token ids; target prefills | connector writes the ack with k (`bookkeeping.plan_trigger`); cell sends the prefix over the link (`cell.py /trigger`); target prefills (`cell._prefill_prefix`) |
| Source keeps decoding; suffix KV blocks streamed as they complete | `connector.py`: scheduler side plans the blocks completed by each step, worker side copies them from the GPU KV cache after the forward and streams them (`wire.pack_blocks`, `_Sender`) |
| Handover: source finishes its step, records n, sends residual blocks + final token snapshot | `bookkeeping.plan_handover`, `cell.py /handover` |
| Suffix blocks resident at the target before the handover | worker thread moves received blocks to GPU memory as they arrive (`connector._Stager`) |
| Target waits for prefix + all of [k, n), then activates without re-prefilling | `bookkeeping.plan_load` (returns "retry later" until complete); activation copies the staged blocks GPU-to-GPU into the request's blocks (`wire.write_blocks`) |
| Epochs: a cancelled attempt's blocks are rejected | `cell._accept_epoch` |
| Not triggered before the handover, or triggered toward the wrong cell: post-handover recovery | falls back to recomputation |
| Baselines Recomputation, Full-Copy | `controller.py` modes `recompute`, `fullcopy` (same link, same cells) |
| Fixed goodputs 100/300/500 Mbps, time-varying traces | `pallas/link.py --mbps`, `--trace t,mbps` |

**For a VLM, unchanged:** the prefix Pallas sends as "exact inputs" must include the raw images, since image tokens can only be rebuilt from them. The target encodes them during its prefix prefill. Everything else is as in the paper.

## Layout

```
pallas/model.py        Eqs. 1-9, Algorithm 1 (pure Python)
pallas/mobility.py     nuScenes paths, cell grid, CVH prediction traces
scripts/pick_multiview.py  keyframes x 6 cameras from nuScenes (standard library only)
scripts/study.slurm    the study (one job per sweep), in the vLLM container
scripts/summarize.py   results -> when do images cost a lot
pallas/bookkeeping.py  block planning for trigger / stream / handover / load (pure Python)
pallas/connector.py    vLLM KV connector: suffix streaming (source) and injection (target)
pallas/wire.py         link frames; KV block packing / injection for any paged-KV layout
pallas/link.py         the inter-gNB link: TCP relay with a shared token bucket
pallas/rx.py           target side: store received blocks and the snapshot
pallas/cell.py         a cell: vLLM AsyncLLM + control API + link receiver
pallas/controller.py   prediction, Algorithm 1 control loop, policies, measurements
run_pallas.py          experiment CLI
scripts/launch.sh      two cells, the link, and the link to a wrong target
scripts/cvh_traces.py  CVH prediction traces from nuScenes (once)
tests/                 unit tests, end-to-end connector test, whole-system test (no GPU needed)
```

## Run (2 GPUs, e.g. 2× A100 on one node)

```bash
pip install vllm fastapi uvicorn requests pillow numpy
cd pallas_baseline
# 0. once: CVH prediction traces from nuScenes (metadata only)
python3 scripts/cvh_traces.py --dataroot /path/to/nuscenes --version v1.0-trainval --out traces/cvh.jsonl

MODEL=Qwen/Qwen2-VL-7B-Instruct GPU_SRC=0 GPU_TGT=1 MM_KWARGS='{"max_pixels": 3211264}' bash scripts/launch.sh

# 1. first: Full-Copy only, one bandwidth -- exercises the KV path; "next token matches source" must be True
python3 run_pallas.py --images ./imgs --turns 2 --policies fullcopy --bandwidth 1000 --repeats 1 --out results/check.csv

# 2. the baseline study (paper's goodputs and cell sizes), with images and without
for c in 50 100 150; do
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh.jsonl --cell $c --events 30 --out results/qwen_img_${c}m.csv
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh.jsonl --cell $c --events 30 --no-images --out results/qwen_txt_${c}m.csv
done
```

Each Pallas run replays one real handover: CVH's predictions for that handover drive Algorithm 1, and the handover happens at its true time. The same handovers (`--seed`) are used for every link setting and for images vs text. `--handover-after` sets how far into the answer the handover happens. `--prediction oracle --tavail 1 2 4` gives Pallas the exact target and time instead (an upper bound).

## The study: when do images cost a lot?

Two workloads on the same nuScenes handovers, each with a text-only twin (same text, 0 images):

| Workload | Conversation | Images |
|---|---|---|
| `front` | 10 turns, front camera only | 0, 1, 3, 6 or 10 of the turns carry an image |
| `multicam` | 3 turns, in-car questions (DriveLM-style) | each turn 0, 1, 3 or 6 surround cameras |

plus cell size (50/150 m), 4 other users on each cell, and lower resolution (800 px).

```bash
# once, on the Mac: 12 keyframes x 6 cameras from nuScenes mini, then copy to LS6
python3 scripts/pick_multiview.py --dataroot ~/Downloads/nuscenes_mini --version v1.0-mini --n 12 --out imgs6
# on LS6: one job per sweep (see scripts/study.slurm for the model options)
for s in front multicam cells load resolution; do sbatch --export=ALL,SWEEP=$s scripts/study.slurm; done
python3 scripts/summarize.py "results/qwen2vl_*.csv" --out results/summary_qwen2vl
```

`summarize.py` gives, per setting, median SIT for each policy, how much of it is due to images
(difference to the text-only twin), how often Pallas was ready in time, how late its prefix prefill
finished relative to its window, and the KV size Full-Copy moved.

Output per run (CSV):
- `sit_s`, `t_w_star_s` (chosen window), `t_remain_pred_s`, `window_s` (actual, to the true handover), `prefix_prefill_s`, `t_early_s`;
- prediction: `scenario`, `cell_m`, `dwell_s`, `n_triggers`, `n_cancels`, `wrong_target_at_handover`, `wrong_prefix_MB`, `triggered`, `mode` (`pallas` or `recompute` when it fell back);
- `activation_load_s` (injecting the suffix at the target; staged blocks are a GPU copy), `activation_files_staged`;
- `k`, `n_kv`, `kv_MB_sent`, `prompt_tokens`, `image_bytes`;
- `r_p`, `r_d`, `B_mbps`;
- `same_next_token` (the target's first token equals the source's next token).

## Tests without a GPU

```bash
pip install torch pytest fastapi uvicorn requests pillow
python3 -m pytest -q tests            # model, wire format (2 KV layouts), bookkeeping, link rate,
                                      # connector end to end over the real link (byte-exact blocks)
python3 tests/run_fake_system.py      # cells + links + controller with a deterministic fake engine,
                                      # oracle and CVH replay (right, late, switching and wrong target)
```

`run_fake_system.py` runs the real cells, link, connector and controller. Only the model is fake: it is deterministic, so wrong or misplaced KV changes the next token. This was used to find and fix:
- the Algorithm 1 candidate set;
- the link probe;
- the order in which KV and prefix frames arrive;
- that a CVH prediction later than the true handover makes Pallas miss it (correct behaviour, now tested).

## Check on the first GPU run

1. `logs/*.log` shows `pallas: N KV layers, block dim D, shape ...`. If the connector fails to load, check that your vLLM version has `kv_connector_module_path`.
2. Step 1 above gives `same_next_token = True`. If it is False, KV injection is misplaced; send me the log.
3. Pallas rows: most have `triggered = True`; `activation_files_staged` equals `activation_files` and `activation_load_s` is around a millisecond.
4. The log shows "Asynchronous scheduling is disabled" and "Turning off hybrid kv cache manager".

## Differences from the paper (state them)

- **Insertion into the target.** Pallas inserts migrated blocks into a running request inside vLLM 0.8.5. Here the suffix blocks are moved to the target's GPU as they arrive, and when the target creates the request, vLLM's connector API copies them GPU-to-GPU into its blocks before the first step (measured: `activation_load_s`). vLLM then computes the last partial block (< 16 tokens) together with the next token in that first step. That one small step is the only extra work at activation.
- **Control.** Every message between the cells (prefix, KV blocks, final snapshot, full sequence) goes over the measured link, and the target activates when the snapshot arrives. The controller talks to each cell locally over HTTP, as Pallas' node daemon talks to its gNB over gRPC.
- **Parameters the paper does not give** are flags: T_max, ΔT, EWMA factor, control period, probe sizes.
- **Wrong target.** Only two GPU cells exist; a trigger toward a wrong cell sends the same bytes at the same rate over a separate link to a sink (the source pays the full cost; the true target gets nothing).
- **Same host.** Both cells on one node, the link emulated by the relay with real bytes.
- **Scheduling.** vLLM's async scheduling (default since 0.11) is turned off in the cells, as in vLLM 0.8.5; with it the source's final snapshot would lag one token behind its KV. vLLM also turns off its hybrid KV-cache manager when a connector is set, so Gemma 3's sliding-window layers use one KV group, which the connector moves entirely.
- **vLLM version.** A recent vLLM (needed for current VLMs, e.g. Qwen3-VL) instead of 0.8.5; GPUs are A100s instead of RTX A6000.
