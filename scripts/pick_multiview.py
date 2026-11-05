#!/usr/bin/env python3
"""Pick nuScenes keyframes and copy their 6 surround-camera images, one folder per keyframe:

  out/00_scene-0061/CAM_FRONT.jpg, CAM_FRONT_LEFT.jpg, CAM_FRONT_RIGHT.jpg,
                    CAM_BACK.jpg, CAM_BACK_LEFT.jpg, CAM_BACK_RIGHT.jpg
  out/01_scene-0103/...

Keyframes are spread over scenes (round-robin) and over time within each scene, so consecutive
conversation turns see different places. Standard library only (runs on a Mac as is).

  python3 pick_multiview.py --dataroot ~/Downloads/nuscenes_mini --version v1.0-mini --n 12 --out imgs6
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", required=True, help="folder that holds samples/ and v1.0-mini/")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--n", type=int, default=12, help="keyframes (one per conversation turn)")
    ap.add_argument("--out", default="imgs6")
    a = ap.parse_args()

    def table(name):
        with open(os.path.join(a.dataroot, a.version, name + ".json")) as f:
            return json.load(f)

    sensor = {s["token"]: s["channel"] for s in table("sensor")}
    channel = {c["token"]: sensor[c["sensor_token"]] for c in table("calibrated_sensor")}
    scene_name = {s["token"]: s["name"] for s in table("scene")}
    samples = {s["token"]: s for s in table("sample")}
    cams = defaultdict(dict)                                  # sample token -> {camera: file}
    for sd in table("sample_data"):
        ch = channel[sd["calibrated_sensor_token"]]
        if sd["is_key_frame"] and ch in CAMS:
            cams[sd["sample_token"]][ch] = sd["filename"]
    by_scene = defaultdict(list)
    for tok, s in samples.items():
        if len(cams[tok]) == len(CAMS):
            by_scene[scene_name[s["scene_token"]]].append((s["timestamp"], tok))
    for v in by_scene.values():
        v.sort()
    scenes = sorted(by_scene)
    per_scene = -(-a.n // len(scenes))
    picks = []
    for j in range(per_scene):                               # round-robin over scenes, spread in time
        for sc in scenes:
            v = by_scene[sc]
            idx = int((j + 0.5) * len(v) / per_scene)
            if len(picks) < a.n and idx < len(v):
                picks.append((sc, v[idx][1]))
    os.makedirs(a.out, exist_ok=True)
    for i, (sc, tok) in enumerate(picks):
        d = os.path.join(a.out, f"{i:02d}_{sc}")
        os.makedirs(d, exist_ok=True)
        for cam in CAMS:
            shutil.copy(os.path.join(a.dataroot, cams[tok][cam]), os.path.join(d, cam + ".jpg"))
    print(f"{len(picks)} keyframes x {len(CAMS)} cameras from {len(set(p[0] for p in picks))} scenes -> {a.out}/")


if __name__ == "__main__":
    main()
