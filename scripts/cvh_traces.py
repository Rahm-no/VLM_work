#!/usr/bin/env python3
"""Write CVH prediction traces (one per true handover) for run_pallas.py --prediction cvh.

  python3 scripts/cvh_traces.py --dataroot /path/to/nuscenes --version v1.0-trainval --out traces/cvh.jsonl
  python3 scripts/cvh_traces.py --synthetic --out traces/syn.jsonl        # no nuScenes, to test

Each line: {"cell_m", "path", "placement", "dwell_s", "ticks": [[t, target, correct, t_ho_pred], ...]}
with times relative to the true handover. Needs only the nuScenes metadata (v1.0-trainval_meta).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pallas import mobility as m  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cells", default="50,100,150", help="cell sizes in m (paper: 50, 100, 150)")
    ap.add_argument("--placements", type=int, default=5, help="random grid placements per path")
    ap.add_argument("--vel-win", type=float, default=0.5, help="velocity estimation window (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="traces/cvh.jsonl")
    a = ap.parse_args()

    paths = m.synthetic_paths(20, a.seed) if a.synthetic else m.load_nuscenes(a.dataroot, a.version)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for c in (float(x) for x in a.cells.split(",")):
            traces = m.prediction_traces(paths, c, a.placements, a.seed, a.vel_win)
            for tr in traces:
                f.write(json.dumps(tr) + "\n")
            right = [tr["ticks"][-1][2] for tr in traces]
            print(f"{c:g} m cells: {len(traces)} handovers, CVH right at the handover in "
                  f"{100 * np.mean(right):.0f}%, median time in source cell {np.median([t['dwell_s'] for t in traces]):.1f} s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
