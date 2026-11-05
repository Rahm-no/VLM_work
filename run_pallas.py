#!/usr/bin/env python3
"""Run Pallas and the paper's baselines (Recomputation, Full-Copy) on a VLM, with or without images.

Start the cells and the links first (scripts/launch.sh), make CVH traces once
(scripts/cvh_traces.py), then for example:
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh.jsonl --cell 100 --out results/qwen_images.csv
  python3 run_pallas.py --images ./imgs --turns 3 --traces traces/cvh.jsonl --cell 100 --no-images --out results/qwen_text.csv

Pallas sees CVH's prediction (target cell, handover time) every control cycle, replayed from one
nuScenes handover per run; the real handover happens at the true time. --prediction oracle instead
gives the exact target and time T_avail before the handover (--tavail).

Defaults follow the paper where it gives values (alpha = 0.8, block 16, goodputs 100/300/500 Mbps);
parameters the paper does not give are flags (--t-max, --delta-t, --ewma, --control-dt).
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics as st

import requests

from pallas.controller import Experiment, post
from pallas.mobility import oracle_trace
from pallas.model import PallasParams

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
HISTORY_QUESTIONS = [                         # an in-car assistant; the same questions with and without images
    "What are the important objects around the ego vehicle?",
    "Which of these objects could affect what the ego vehicle does next?",
    "What is the state of the traffic ahead?",
    "Are there pedestrians or cyclists nearby, and where?",
    "What should the ego vehicle do next, and why?",
    "Describe the road layout here.",
    "Is anything unusual in this scene?",
    "What changed compared to the previous view?",
    "Roughly how far is the closest vehicle?",
    "Is it safe to change to the left lane now?",
]
LONG_QUESTION = "Summarise everything you have seen so far and give a detailed, step by step driving plan."
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def keyframes(folder):
    """Images per keyframe, in turn order. Either sub-folders (scripts/pick_multiview.py: one per
    keyframe, files CAM_*.jpg) or a flat folder of single images (one keyframe each)."""
    subs = sorted(d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d) and not d.endswith(".img_cache"))
    if subs:
        out = []
        for d in subs:
            files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(d, "*"))
                     if f.lower().endswith(IMG_EXT)}
            out.append([files[c] for c in CAMS if c in files] + sorted(v for k, v in files.items() if k not in CAMS))
        return out
    return [[f] for f in sorted(glob.glob(os.path.join(folder, "*"))) if f.lower().endswith(IMG_EXT)]


def resize(files, size, out_dir):
    """Copies with the longest side resized to `size` px (fewer image tokens); size 0 = originals."""
    if not size:
        return files
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for f in files:
        dst = os.path.join(out_dir, f"{size}_{os.path.basename(os.path.dirname(f))}_{os.path.basename(f)}")
        if not os.path.exists(dst):
            im = Image.open(f).convert("RGB")
            im.thumbnail((size, size))
            im.save(dst, quality=95)
        out.append(dst)
    return out


def build_history(source, frames, turns, image_turns, cameras, answer_tokens):
    """Serve a multi-turn history at the source. User turn i < image_turns carries the first `cameras`
    camera images of keyframe i; the text is the same whatever image_turns and cameras are. The last
    turn asks for a long answer (decoded during the handover)."""
    hist = []
    for i in range(turns):
        imgs = [os.path.abspath(f) for f in frames[i][:cameras]] if i < image_turns else []
        hist.append({"role": "user", "text": HISTORY_QUESTIONS[i % len(HISTORY_QUESTIONS)], "images": imgs})
        # every answer exactly answer_tokens long (no early stop), so the text is the same whatever the images
        r = post(source, "/generate", {"session": f"hist{i}", "turns": hist, "max_tokens": answer_tokens,
                                       "ignore_eos": True, "wait": True})
        hist.append({"role": "assistant", "text": r["text"]})
    post(source, "/reset", {})
    return hist + [{"role": "user", "text": LONG_QUESTION}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="http://127.0.0.1:8100", help="source cell control url")
    ap.add_argument("--target", default="http://127.0.0.1:8200", help="target cell control url")
    ap.add_argument("--link", default="127.0.0.1:9100", help="link relay address the source sends to")
    ap.add_argument("--link-control", default="http://127.0.0.1:9199", help="link relay control url")
    ap.add_argument("--wrong-link", default="127.0.0.1:9300", help="link to a cell outside the experiment")
    ap.add_argument("--wrong-link-control", default="http://127.0.0.1:9399")
    ap.add_argument("--prediction", choices=["cvh", "oracle"], default="cvh")
    ap.add_argument("--traces", help="CVH traces from scripts/cvh_traces.py (--prediction cvh)")
    ap.add_argument("--cell", type=float, default=100.0, help="cell size of the traces to use (m)")
    ap.add_argument("--events", type=int, default=20, help="handovers (traces) per link setting for Pallas")
    ap.add_argument("--min-dwell", type=float, default=0.0,
                    help="skip handovers whose vehicle spent less than this in the source cell (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the discarded warm-up runs")
    ap.add_argument("--images", required=True)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--no-images", action="store_true", help="same as --image-turns 0")
    ap.add_argument("--image-turns", "--num-images", dest="image_turns", type=int,
                    help="history turns that carry images (default: all --turns)")
    ap.add_argument("--cameras", type=int, default=1,
                    help="images per such turn: 1 = front camera only, 6 = all surround cameras")
    ap.add_argument("--image-size", type=int, default=0, help="resize images to this longest side (px); 0 = original")
    ap.add_argument("--cell-load", type=int, default=0, help="other users kept busy on each cell (text requests)")
    ap.add_argument("--load-prompt-tokens", type=int, default=1000)
    ap.add_argument("--load-max-tokens", type=int, default=256)
    ap.add_argument("--history-answer-tokens", type=int, default=64)
    ap.add_argument("--bandwidth", type=float, nargs="+", default=[100, 300, 500], help="link goodput, Mbps")
    ap.add_argument("--tavail", type=float, nargs="+", default=[0.5, 1, 2, 4],
                    help="--prediction oracle: s from the exact prediction to the handover")
    ap.add_argument("--handover-after", type=float, default=6.0, help="s into the answer when the handover happens")
    ap.add_argument("--max-tokens", type=int, default=4096, help="length budget of the answer being decoded")
    ap.add_argument("--policies", nargs="+", default=["pallas", "recompute", "fullcopy"])
    ap.add_argument("--repeats", type=int, default=3, help="runs per link setting for the baselines")
    ap.add_argument("--alpha", type=float, default=0.8, help="Eq. 8 weight (paper: 0.8)")
    ap.add_argument("--t-max", type=float, default=5.0, help="largest window considered (not in the paper)")
    ap.add_argument("--delta-t", type=float, default=0.05, help="window grid step (not in the paper)")
    ap.add_argument("--ewma", type=float, default=0.3, help="EWMA factor (not in the paper)")
    ap.add_argument("--control-dt", type=float, default=0.1, help="control-cycle period (not in the paper)")
    ap.add_argument("--probe-prefill-tokens", type=int, default=3000, help="r_p probe size")
    ap.add_argument("--probe-link-mb", type=float, default=4.0, help="B probe size")
    ap.add_argument("--out", default="results/pallas.csv")
    a = ap.parse_args()

    params = PallasParams(alpha=a.alpha, t_max=a.t_max, delta_t=a.delta_t, ewma=a.ewma)
    exp = Experiment(a.source, a.target, a.link, wrong_link=a.wrong_link, params=params, control_dt=a.control_dt)
    if "pallas" not in a.policies:
        scenarios = []
    elif a.prediction == "cvh":
        if not a.traces:
            ap.error("--prediction cvh needs --traces (scripts/cvh_traces.py)")
        pool = [t for t in map(json.loads, open(a.traces))
                if t["cell_m"] == a.cell and t["dwell_s"] >= a.min_dwell]
        if not pool:
            ap.error(f"no traces for {a.cell:g} m cells in {a.traces}")
        traces = random.Random(a.seed).sample(pool, min(a.events, len(pool)))   # same handovers for every link
        scenarios = [(f"cvh{a.cell:g}m#{i}", t) for i, t in enumerate(traces)]
    else:
        scenarios = [(f"oracle{ta:g}s#{r}", oracle_trace(ta)) for ta in a.tavail for r in range(a.repeats)]
    image_turns = 0 if a.no_images else (a.turns if a.image_turns is None else a.image_turns)
    if image_turns > a.turns:
        ap.error("--image-turns cannot exceed --turns")
    frames = keyframes(a.images)
    if len(frames) < image_turns:
        ap.error(f"{image_turns} keyframes needed (each turn a different one), {len(frames)} in {a.images}")
    if image_turns and any(len(fr) < a.cameras for fr in frames[:image_turns]):
        ap.error(f"--cameras {a.cameras}: some keyframes in {a.images} have fewer images "
                 "(use scripts/pick_multiview.py for 6 cameras)")
    cache = os.path.join(os.path.dirname(a.out) or ".", ".img_cache")
    frames = [resize(fr[:a.cameras], a.image_size, cache) for fr in frames]
    n_images = image_turns * a.cameras
    for cell in (a.source, a.target):
        post(cell, "/load", {"n": 0})
    history = build_history(a.source, frames, a.turns, image_turns, a.cameras, a.history_answer_tokens)
    for cell in (a.source, a.target):                    # other users of both cells, during all runs
        post(cell, "/load", {"n": a.cell_load, "prompt_tokens": a.load_prompt_tokens, "max_tokens": a.load_max_tokens})
    rows = []
    if a.warmup:                     # first requests of each kind on fresh engines are slow: run, discard
        requests.get(f"{a.link_control}/set?mbps={a.bandwidth[0]}", timeout=10)
        exp.setup(a.probe_prefill_tokens, a.probe_link_mb)
        for pol in a.policies:
            exp.run(pol, history, a.handover_after, oracle_trace(2.0), a.max_tokens)
        print("warm-up runs done (not recorded)")
    for bw in a.bandwidth:
        for ctl in (a.link_control, a.wrong_link_control):
            requests.get(f"{ctl}/set?mbps={bw}", timeout=10)
        info = exp.setup(a.probe_prefill_tokens, a.probe_link_mb)          # fresh EWMAs per link setting
        print(f"\n=== link {bw:g} Mbps (probed B {exp.rates.b.value / 1e6:.0f} Mbps), "
              f"r_p {exp.rates.r_p.value:.0f} tok/s, s_KV {exp.s_kv / 1024:.0f} KiB/token, "
              f"{n_images} images ({image_turns} turns x {a.cameras} cameras, {a.image_size or 'original'} px), {a.turns} turns, {a.cell_load} other users per cell, "
              f"prediction {a.prediction} ===")
        runs = [(p, None, None) for p in a.policies if p != "pallas" for _ in range(a.repeats)]
        if "pallas" in a.policies:
            runs += [("pallas", name, tr) for name, tr in scenarios]
        for i, (pol, name, tr) in enumerate(runs):
            row = exp.run(pol, history, a.handover_after, tr, a.max_tokens)
            row.update(link_mbps=bw, scenario=name, turns=a.turns, image_turns=image_turns, cameras=a.cameras,
                       image_size=a.image_size, cell_load=a.cell_load,
                       cell_arg_m=a.cell if a.prediction == "cvh" else None, model=info["model"],
                       prediction=a.prediction if pol == "pallas" else None)
            rows.append(row)
            extra = ""
            if pol == "pallas":
                extra = (f" [{name}] triggers={row['n_triggers']} cancels={row['n_cancels']}"
                         + (f" T_w*={row['t_w_star_s']:.2f}s window={row['window_s']:.2f}s "
                            f"prefix prefill={row.get('prefix_prefill_s') or 0:.2f}s" if row["triggered"]
                            else " -> recomputation" + (" (wrong target)" if row["wrong_target_at_handover"] else "")))
            act = f", activation load {row['activation_load_s'] * 1000:.1f} ms" if row.get("activation_load_s") else ""
            print(f"  {pol:>9}: SIT {row['sit_s'] * 1000:7.0f} ms, prompt {row['prompt_tokens']} tok "
                  f"({row['image_tokens']} image), "
                  f"KV sent {row['kv_MB_sent']:.1f} MB{act}, next token matches source: {row['same_next_token']}{extra}")
            if (i + 1) % 5 == 0:
                exp.probe(0, a.probe_link_mb)                               # keep B's EWMA current

    for cell in (a.source, a.target):
        post(cell, "/load", {"n": 0})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nmedian SIT (ms)")
    for bw in a.bandwidth:
        for pol in a.policies:
            v = [r["sit_s"] for r in rows if r["policy"] == pol and r["link_mbps"] == bw]
            if v:
                print(f"  {bw:>5g} Mbps {pol:>9}: {st.median(v) * 1000:.0f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
