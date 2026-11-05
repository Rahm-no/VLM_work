#!/usr/bin/env python3
"""Summarise run_pallas.py results: when do images cost a lot, and when not?

  python3 scripts/summarize.py results/study_*.csv --out results/summary

One row per setting (model, images, image size, other users, link, cell size), with per policy:
median / p90 SIT, and for Pallas how often it was ready in time, how often it fell back, and how late
its prefix prefill finished relative to its window (the part images make worse).
Also, for every setting with images, the same setting without images, so the cost of images is a
difference you can read off directly.

Writes <out>.csv and <out>.md; with matplotlib, also <out>_sit_vs_images.png.
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

KEY = ("model", "turns", "cameras", "n_images", "image_size", "cell_load", "link_mbps", "cell_arg_m")
POLICIES = ("pallas", "recompute", "fullcopy")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            r["n_images"] = str(int(f(r.get("n_images")) or 0))
            r["image_size"] = r.get("image_size") or "0"
            r["cell_load"] = r.get("cell_load") or "0"
            r["cell_arg_m"] = r.get("cell_arg_m") or r.get("cell_m") or ""
            r["cameras"] = r.get("cameras") or "1"
            r["turns"] = r.get("turns") or ""
            rows.append(r)
    return rows


def summarise(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[tuple(r.get(k, "") for k in KEY)][r["policy"]].append(r)
    out = []
    for key, by_pol in sorted(groups.items(), key=lambda kv: (kv[0][0],) + tuple(f(x) if f(x) is not None else 0 for x in kv[0][1:])):
        row = dict(zip(KEY, key))
        anyr = next(iter(by_pol.values()))
        row["prompt_tokens"] = int(st.median(f(r["prompt_tokens"]) for r in anyr))
        row["image_tokens"] = int(st.median(f(r.get("image_tokens")) or 0 for r in anyr))
        for pol in POLICIES:
            v = [f(r["sit_s"]) * 1000 for r in by_pol.get(pol, []) if f(r["sit_s"]) is not None]
            row[f"{pol}_n"] = len(v)
            row[f"{pol}_sit_p50_ms"] = round(st.median(v)) if v else None
            row[f"{pol}_sit_p90_ms"] = round(pct(v, 0.9)) if v else None
            ok = [r["same_next_token"] == "True" for r in by_pol.get(pol, [])]
            row[f"{pol}_correct_%"] = round(100 * sum(ok) / len(ok)) if ok else None
        p = by_pol.get("pallas", [])
        if p:
            trig = [r for r in p if r.get("triggered") == "True"]
            row["pallas_ready_%"] = round(100 * len(trig) / len(p))
            row["pallas_wrong_target_%"] = round(100 * sum(r.get("wrong_target_at_handover") == "True" for r in p) / len(p))
            late = [f(r["prefix_prefill_s"]) - f(r["window_s"]) for r in trig
                    if f(r.get("prefix_prefill_s")) is not None and f(r.get("window_s")) is not None]
            row["pallas_prefix_late_p50_ms"] = round(1000 * st.median(late)) if late else None
            w = [f(r["t_w_star_s"]) for r in trig if f(r.get("t_w_star_s")) is not None]
            row["pallas_window_p50_s"] = round(st.median(w), 2) if w else None
            pp = [f(r["prefix_prefill_s"]) for r in trig if f(r.get("prefix_prefill_s")) is not None]
            row["pallas_prefix_prefill_p50_s"] = round(st.median(pp), 2) if pp else None
        fc = by_pol.get("fullcopy", [])
        if fc:
            row["fullcopy_MB"] = round(st.median(f(r["kv_MB_sent"]) for r in fc), 1)
        out.append(row)
    # cost of images: same setting without images
    twin = lambda r: tuple(r[k] for k in KEY if k not in ("n_images", "cameras", "image_size"))
    base = {twin(r): r for r in out if r["n_images"] == "0"}
    for r in out:
        b = base.get(twin(r))
        for pol in POLICIES:
            x, y = r.get(f"{pol}_sit_p50_ms"), b.get(f"{pol}_sit_p50_ms") if b else None
            r[f"{pol}_extra_ms_vs_text"] = (x - y) if x is not None and y is not None else None
    return out


def write(out, prefix):
    cols = list(dict.fromkeys(k for r in out for k in r))
    with open(prefix + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    show = ["turns", "cameras", "n_images", "image_tokens", "image_size", "cell_load", "link_mbps", "cell_arg_m",
            "pallas_sit_p50_ms", "pallas_extra_ms_vs_text", "pallas_ready_%", "pallas_prefix_late_p50_ms",
            "recompute_sit_p50_ms", "recompute_extra_ms_vs_text", "fullcopy_sit_p50_ms", "fullcopy_MB"]
    names = ["turns", "cams/turn", "images", "image tok", "px", "users", "Mbps", "cell m", "Pallas SIT", "+ vs text", "Pallas ready %",
             "prefix late", "Recomp SIT", "+ vs text", "Full-Copy SIT", "KV MB"]
    with open(prefix + ".md", "w") as fh:
        for model in sorted({r["model"] for r in out}):
            fh.write(f"## {model}\n\nSIT in ms (median). '+ vs text': extra over the same setting without images.\n\n")
            fh.write("| " + " | ".join(names) + " |\n|" + "---|" * len(names) + "\n")
            for r in out:
                if r["model"] == model:
                    fh.write("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in show) + " |\n")
            fh.write("\n")
    bad = [(r["n_images"], r["link_mbps"], pol) for r in out for pol in POLICIES
           if r.get(f"{pol}_correct_%") not in (None, 100)]
    if bad:
        print("WARNING: next token differed from the source in some runs:", bad)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pick = [r for r in out if r["cell_load"] == "0" and f(r["link_mbps"]) == 300]
    workloads = sorted({(r["model"], r["turns"]) for r in pick})
    if not workloads:
        return
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.5 * len(workloads), 3.8), sharey=True, squeeze=False)
    for ax, (model, turns) in zip(axes[0], workloads):
        rs = [r for r in pick if r["model"] == model and r["turns"] == turns]
        size = max({r["image_size"] for r in rs if r["n_images"] != "0"} or {"0"}, key=lambda z: sum(r["image_size"] == z for r in rs))
        rs = sorted([r for r in rs if r["n_images"] == "0" or r["image_size"] == size],
                    key=lambda r: int(r["n_images"]))
        x = [int(r["n_images"]) for r in rs]
        for key, col, ls, lab in (("recompute_sit_p50_ms", "#eb6834", "-", "Recomputation, median"),
                                  ("pallas_sit_p90_ms", "#2a78d6", "--", "Pallas, worst 10%"),
                                  ("pallas_sit_p50_ms", "#2a78d6", "-", "Pallas, median")):
            pts = [(xi, r[key] / 1000) for xi, r in zip(x, rs) if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), color=col, ls=ls, marker="o", lw=2, label=lab)
        px = "native" if size in ("0", "") else f"{size} px"
        ax.set_title(f"{model.split('/')[-1]}, {turns} questions, images at {px}", fontsize=10, loc="left")
        ax.set_xlabel("images in the conversation")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("pause at handover (s), 300 Mbps")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(prefix + "_sit_vs_images.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()
    paths = [p for g in a.csvs for p in glob.glob(g)]
    out = summarise(load(paths))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write(out, a.out)
    print(open(a.out + ".md").read())
    print(f"wrote {a.out}.csv, {a.out}.md" + (f", {a.out}_sit_vs_images.png" if os.path.exists(a.out + "_sit_vs_images.png") else ""))


if __name__ == "__main__":
    main()
