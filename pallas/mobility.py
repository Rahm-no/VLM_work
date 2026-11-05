"""Mobility and CVH handover prediction, as in Pallas: vehicle paths (nuScenes ego poses) laid over a
square grid of cells (50 / 100 / 150 m), and constant-velocity-and-heading (CVH) prediction of the
next cell and of the time the vehicle crosses into it, every control cycle (0.1 s).

For each true handover, `prediction_trace` records what CVH said at every tick while the vehicle was
in the source cell: the predicted target cell, whether it is the true target, and the predicted
handover time. The controller replays these traces in real time (pallas/controller.py).
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

TICK = 0.1           # control cycle (s)
MIN_SPEED = 0.5      # m/s; slower: no prediction


def load_nuscenes(dataroot, version):
    """Ego paths (LIDAR_TOP poses) per scene, resampled to TICK. Needs only the metadata JSONs."""
    d = os.path.join(dataroot, version)

    def table(name):
        with open(os.path.join(d, name + ".json")) as f:
            return json.load(f)

    sensor = {s["token"]: s["channel"] for s in table("sensor")}
    channel = {c["token"]: sensor[c["sensor_token"]] for c in table("calibrated_sensor")}
    scene_of = {s["token"]: s["scene_token"] for s in table("sample")}
    pose = {p["token"]: p for p in table("ego_pose")}
    by_scene = defaultdict(list)
    for sd in table("sample_data"):
        if channel[sd["calibrated_sensor_token"]] != "LIDAR_TOP":
            continue
        p = pose[sd["ego_pose_token"]]
        by_scene[scene_of[sd["sample_token"]]].append((p["timestamp"] / 1e6, p["translation"][0], p["translation"][1]))
    out = []
    for rows in by_scene.values():
        a = np.array(sorted(rows))
        if a[-1, 0] - a[0, 0] > 3:
            tt = np.arange(a[0, 0], a[-1, 0], TICK)
            out.append(np.stack([np.interp(tt, a[:, 0], a[:, 1]), np.interp(tt, a[:, 0], a[:, 2])], 1))
    return out


def synthetic_paths(n, seed, duration=90.0):
    """Random urban-like paths, only to test the pipeline without nuScenes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v, th, x, y, om, acc, pts = rng.uniform(8, 14), rng.uniform(0, 2 * math.pi), 0.0, 0.0, 0.0, 0.0, []
        for _ in range(int(duration / TICK)):
            if rng.random() < TICK / 4:
                om = 0.0 if rng.random() < 0.6 else rng.normal(0, 0.3)
                acc = rng.normal(0, 0.8)
            v = float(np.clip(v + acc * TICK, 1, 18))
            th += om * TICK
            x += v * math.cos(th) * TICK
            y += v * math.sin(th) * TICK
            pts.append((x, y))
        out.append(np.array(pts))
    return out


def place(xy, c, rng):
    """Random translation and rotation of a path onto the grid (no scaling)."""
    phi = rng.uniform(0, 2 * math.pi)
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    return (xy - xy[0]) @ R.T + rng.uniform(0, c, size=2)


def handovers(P, c):
    """True handovers [(tick before crossing, time, source cell, target cell)]. At a corner, the edge
    crossed first counts."""
    cells = np.floor(P / c).astype(int)
    evs = []
    for i in np.nonzero((cells[1:] != cells[:-1]).any(axis=1))[0]:
        d = P[i + 1] - P[i]
        frac, ax_first = 2.0, 0
        for ax in (0, 1):
            if cells[i + 1, ax] != cells[i, ax] and d[ax] != 0:
                f = (max(cells[i, ax], cells[i + 1, ax]) * c - P[i, ax]) / d[ax]
                if f < frac:
                    frac, ax_first = f, ax
        dst = cells[i].copy()
        dst[ax_first] = cells[i + 1, ax_first]
        evs.append((int(i), (i + frac) * TICK, tuple(int(v) for v in cells[i]), tuple(int(v) for v in dst)))
    return cells, evs


def cvh(P, V, cells, c):
    """CVH at every tick: predicted next cell and time until crossing into it."""
    n = len(P)
    t = np.full((n, 2), np.inf)
    for ax in (0, 1):
        pos, neg = V[:, ax] > 0, V[:, ax] < 0
        t[pos, ax] = ((cells[pos, ax] + 1) * c - P[pos, ax]) / V[pos, ax]
        t[neg, ax] = (cells[neg, ax] * c - P[neg, ax]) / V[neg, ax]
    t = np.maximum(t, 0.0)
    ax = np.argmin(t, axis=1)
    tc = t[np.arange(n), ax]
    tgt = cells.copy()
    tgt[np.arange(n), ax] += np.sign(V[np.arange(n), ax]).astype(int)
    valid = (np.linalg.norm(V, axis=1) >= MIN_SPEED) & np.isfinite(tc)
    return tgt, tc, valid


def prediction_traces(paths, c, placements, seed, vel_win=0.5, lookback=15.0):
    """One trace per true handover:
      {"cell_m", "path", "placement", "dwell_s",
       "ticks": [[t, target or None, correct, predicted handover time], ...]}
    Times are relative to the true handover (t <= 0). Ticks cover the time in the source cell
    (at most `lookback` s)."""
    rng = np.random.default_rng(seed)
    win = max(1, int(round(vel_win / TICK)))
    out = []
    for k, xy in enumerate(paths):
        for p in range(placements):
            P = place(xy, c, rng)
            V = np.zeros_like(P)
            V[win:] = (P[win:] - P[:-win]) / (win * TICK)
            cells, evs = handovers(P, c)
            tgt, tc, valid = cvh(P, V, cells, c)
            valid[:win] = False
            for (i, t_ho, src, dst) in evs:
                j0 = i
                while j0 > 0 and tuple(cells[j0 - 1]) == src and (i - j0 + 1) * TICK < lookback:
                    j0 -= 1
                ticks = []
                for j in range(j0, i + 1):
                    t_rel = round(j * TICK - t_ho, 3)
                    if not valid[j]:
                        ticks.append([t_rel, None, 0, None])
                        continue
                    g = (int(tgt[j, 0]), int(tgt[j, 1]))
                    ticks.append([t_rel, f"{g[0]},{g[1]}", int(g == dst), round(t_rel + float(tc[j]), 3)])
                out.append({"cell_m": c, "path": k, "placement": p, "dwell_s": round(t_ho - j0 * TICK, 2),
                            "ticks": ticks})
    return out


def oracle_trace(t_avail, dt=TICK):
    """A perfect predictor that knows the right target and time from t_avail before the handover."""
    n = int(round(t_avail / dt))
    return {"cell_m": None, "path": None, "placement": None, "dwell_s": t_avail,
            "ticks": [[round(-t_avail + i * dt, 3), "T", 1, 0.0] for i in range(n + 1)]}


def prediction_at(trace, t_rel):
    """The latest tick at or before t_rel: (target, correct, predicted handover time) or None."""
    best = None
    for tk in trace["ticks"]:
        if tk[0] <= t_rel + 1e-9:
            best = tk
        else:
            break
    if best is None or best[1] is None:
        return None
    return best[1], bool(best[2]), best[3]
