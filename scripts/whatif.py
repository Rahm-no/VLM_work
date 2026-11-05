#!/usr/bin/env python3
"""What-if: how often would Pallas freeze the user, over ALL handovers in the traces?

Replays Pallas's trigger rule on every handover's prediction ticks, then uses the relation measured
on the GPU runs (Qwen2-VL-7B, 300 Mbps):
    pause ~= max(0, R - W) + BASE      (W = time from Pallas's trigger to the true handover)
    pause ~= R + BASE                  (never triggered, or prepared the wrong cell: fallback)
Pallas triggers toward the predicted cell once the PREDICTED time to handover <= R (its planned
window), and cancels if the predicted cell changes, as in the controller. So a handover that comes
earlier than predicted shortens W, as on the GPU. R = re-processing time on the new cell (measured
for the first rows). A faster car divides every time in the trace by the speed factor.

  python3 scripts/whatif.py traces/cvh_mini.jsonl
"""
import json
import sys
from collections import defaultdict

BASE = 0.2
# re-processing time per handover (s). measured: Qwen2-VL-7B on A100. others: estimates, labelled.
R = [("text (874 tok), measured", 0.15),
     ("18 cams 1024px (14k tok), measured", 1.45),
     ("10 front full (19k tok), measured", 1.97),
     ("18 cams full, Qwen3-VL (25k tok), measured", 5.16),
     ("10 front full, 32B model (~4x), est.", 7.9)]
SPEEDS = [1, 2, 3]          # 1 = nuScenes city speed; 3 ~ highway


def warning(tr):
    """Seconds before the handover from which every tick names the right cell (0 if never)."""
    w = 0.0
    for t, tgt, correct, _ in sorted(tr["ticks"], key=lambda x: -x[0]):   # from the handover backwards
        if tgt is None or not correct:
            break
        w = -t
    return w


def pallas_window(tr, r, speed):
    """Pallas's trigger rule on this trace: window W (s) if the attempt in place at the handover is
    toward the right cell, else None (fallback to recomputation)."""
    att = None                                         # (target, correct, t_trigger)
    for t, tgt, correct, t_pred in tr["ticks"]:
        t, t_pred = t / speed, (t_pred / speed if t_pred is not None else None)
        if tgt is None:
            continue                                   # no prediction: keep what is there
        if att and tgt != att[0]:
            att = None                                 # predicted cell changed: cancel
        if att is None and t_pred is not None and max(0.0, t_pred - t) <= r:
            att = (tgt, bool(correct), t)
    if att is None or not att[1]:
        return None
    return -att[2]


def main(path):
    by_cell = defaultdict(list)
    for line in open(path):
        tr = json.loads(line)
        by_cell[tr["cell_m"]].append(tr)
    for c in sorted(by_cell):
        trs = by_cell[c]
        ws = [warning(tr) for tr in trs]
        print(f"\n### {c:g} m cells: {len(trs)} handovers; right cell known < 2 s before in "
              f"{100 * sum(w < 2 for w in ws) / len(ws):.0f}%")
        print(f"| conversation | car speed | pause > 0.5 s | pause > 1 s | mean pause | fallback |")
        print("|---|---|---|---|---|---|")
        for name, r in R:
            for s in SPEEDS:
                win = [pallas_window(tr, r, s) for tr in trs]
                p = [r + BASE if w is None else max(0.0, r - w) + BASE for w in win]
                print(f"| {name} | x{s} | {100 * sum(x > 0.5 for x in p) / len(p):.0f}% | "
                      f"{100 * sum(x > 1 for x in p) / len(p):.0f}% | {sum(p) / len(p):.2f} s | "
                      f"{100 * sum(w is None for w in win) / len(win):.0f}% |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "traces/cvh_mini.jsonl")
