#!/usr/bin/env python3
"""Whole system without a GPU: link relay + two cell processes (fake engine, real connector) +
the real controller (Algorithm 1). Checks every policy completes, the target's next token equals the
source's, and the target never loaded wrong KV.   python3 tests/run_fake_system.py"""
import csv, json, os, subprocess, sys, tempfile, time

import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp()
imgs = os.path.join(tmp, "imgs"); os.makedirs(imgs)
for i in range(2):
    Image.effect_noise((256, 256), 64).convert("RGB").save(os.path.join(imgs, f"im{i}.jpg"), quality=95)
imgs6 = os.path.join(tmp, "imgs6")                  # pick_multiview.py layout: one folder per keyframe
for k in range(3):
    os.makedirs(os.path.join(imgs6, f"{k:02d}_scene"))
    for cam in ("CAM_FRONT", "CAM_FRONT_LEFT"):
        Image.effect_noise((256, 256), 64).convert("RGB").save(os.path.join(imgs6, f"{k:02d}_scene", cam + ".jpg"))
env = dict(os.environ, PALLAS_FAKE_ENGINE="1", PYTHONPATH=ROOT)
procs = [subprocess.Popen([sys.executable, "-m", "pallas.cell", "--name", n, "--model", "fake", "--http-port", str(hp),
                           "--data-port", str(dp), "--state-root", os.path.join(tmp, "state"), "--wait-timeout", "20"],
                          cwd=ROOT, env=env, stdout=open(os.path.join(tmp, f"{n}.log"), "w"), stderr=subprocess.STDOUT)
         for n, hp, dp in (("source", 18100, 19101), ("target", 18200, 19201))]
procs.append(subprocess.Popen([sys.executable, "-m", "pallas.link", "--listen", "127.0.0.1:19100", "--to",
                               "127.0.0.1:19201", "--mbps", "100", "--stats-port", "19199"], cwd=ROOT, env=env))
procs.append(subprocess.Popen([sys.executable, "-m", "pallas.link", "--listen", "127.0.0.1:19300", "--to", "sink",
                               "--mbps", "100", "--stats-port", "19399"], cwd=ROOT, env=env))

# CVH-like traces (times relative to the true handover):
#   right   right target from -5 s, predicted handover 0.3 s early
#   late    right target, predicted 0.3 s late: Pallas waits for T_remain <= T_w* and misses the handover
#   switch  wrong target until -0.1 s (Pallas triggers toward it), then the right one: cancel + re-trigger
#   wrong   wrong target until the handover: post-handover recovery
def ticks(fn):
    return [[round(-5 + 0.1 * i, 3), *fn(round(-5 + 0.1 * i, 3))] for i in range(51)]
traces = [
    {"name": "right", "ticks": ticks(lambda t: ["1,0", 1, -0.3])},
    {"name": "late", "ticks": ticks(lambda t: ["1,0", 1, 0.3])},
    {"name": "switch", "ticks": ticks(lambda t: ["9,9", 0, 0.0] if t < -0.1 else ["1,0", 1, 0.0])},
    {"name": "wrong", "ticks": ticks(lambda t: ["9,9", 0, 0.0])},
]
trace_file = os.path.join(tmp, "traces.jsonl")
with open(trace_file, "w") as f:
    for i, tr in enumerate(traces):
        f.write(json.dumps(dict(tr, cell_m=100.0, path=i, placement=0, dwell_s=5.0)) + "\n")
try:
    for p in (18100, 18200):
        for _ in range(100):
            try:
                requests.get(f"http://127.0.0.1:{p}/info", timeout=1); break
            except requests.RequestException:
                time.sleep(0.2)
    common = [sys.executable, "run_pallas.py", "--source", "http://127.0.0.1:18100", "--target",
              "http://127.0.0.1:18200", "--link", "127.0.0.1:19100", "--link-control", "http://127.0.0.1:19199",
              "--wrong-link", "127.0.0.1:19300", "--wrong-link-control", "http://127.0.0.1:19399",
              "--images", imgs, "--turns", "2", "--handover-after", "4", "--probe-prefill-tokens", "2000",
              "--probe-link-mb", "1", "--history-answer-tokens", "8"]
    out1, out2 = os.path.join(tmp, "oracle.csv"), os.path.join(tmp, "cvh.csv")
    r = subprocess.run(common + ["--prediction", "oracle", "--bandwidth", "100", "300", "--tavail", "0.3", "3",
                                 "--repeats", "1", "--out", out1], cwd=ROOT, env=env)
    assert r.returncode == 0, "controller failed (oracle)"
    r = subprocess.run(common + ["--prediction", "cvh", "--traces", trace_file, "--cell", "100", "--events", "4",
                                 "--policies", "pallas", "--bandwidth", "300", "--out", out2], cwd=ROOT, env=env)
    assert r.returncode == 0, "controller failed (cvh)"
    out3, out4 = os.path.join(tmp, "k0_load.csv"), os.path.join(tmp, "k2_load.csv")
    for k, out in ((0, out3), (2, out4)):           # text-only twin + images, with 2 other users per cell
        args = [a if a != imgs else imgs6 for a in common]   # 2 cameras per turn, 2 turns with images
        r = subprocess.run(args + ["--prediction", "cvh", "--traces", trace_file, "--cell", "100", "--events", "1",
                                   "--repeats", "1", "--bandwidth", "300", "--turns", "3", "--image-turns", str(k),
                                   "--cameras", "2", "--cell-load", "2", "--out", out], cwd=ROOT, env=env)
        assert r.returncode == 0, f"controller failed (load, {k} images)"
    loadrows = list(csv.DictReader(open(out3))) + list(csv.DictReader(open(out4)))
    assert {x["n_images"] for x in loadrows} == {"0", "4"} and all(x["cell_load"] == "2" for x in loadrows)
    assert all(int(x["image_tokens"]) == (2400 if x["n_images"] == "4" else 0) for x in loadrows), \
        [(x["n_images"], x["image_tokens"]) for x in loadrows]
    bad = [x for x in loadrows if x["same_next_token"] != "True"]
    assert not bad, f"with other users on the cells, next token differs: {[(x['policy'], x['n_images']) for x in bad]}"
    r = subprocess.run([sys.executable, "scripts/summarize.py", out1, out3, out4, "--out", os.path.join(tmp, "summary")],
                       cwd=ROOT, env=env)
    assert r.returncode == 0, "summary failed"
    rows = list(csv.DictReader(open(out1)))
    cvh = {traces[int(x["path"])]["name"]: x for x in csv.DictReader(open(out2))}
    mism = requests.get("http://127.0.0.1:18200/info").json()["kv_mismatch"]
    print(f"\n{len(rows)} oracle runs + {len(cvh)} CVH runs + {len(loadrows)} runs with other users; "
          f"target KV mismatches: {mism}")
    allr = rows + list(cvh.values()) + loadrows
    # every policy now records the source's next token; missing (None) counts as a failure
    bad = [x for x in allr if x["same_next_token"] != "True"]
    assert not bad, f"next token differs in {len(bad)} runs: {[(x['policy'], x['scenario']) for x in bad]}"
    assert mism == 0
    trig = [x for x in rows if x["policy"] == "pallas" and x["scenario"].startswith("oracle3s")]
    assert all(x["triggered"] == "True" for x in trig), "Pallas did not trigger with T_avail = 3 s"
    b = [float(x["B_mbps"]) for x in rows if x["link_mbps"] == "100.0"]
    assert all(80 <= v <= 105 for v in b), f"link probe off: {b}"
    # prediction replay
    assert cvh["right"]["triggered"] == "True" and cvh["right"]["n_cancels"] == "0"
    assert cvh["right"]["window_s"] and float(cvh["right"]["window_s"]) > 0.3
    assert cvh["late"]["triggered"] == "False" and cvh["late"]["n_triggers"] == "0"
    assert cvh["switch"]["n_triggers"] == "2" and cvh["switch"]["n_cancels"] == "1", cvh["switch"]
    assert cvh["switch"]["triggered"] == "True" and float(cvh["switch"]["wrong_prefix_MB"]) > 0
    assert cvh["wrong"]["wrong_target_at_handover"] == "True" and cvh["wrong"]["mode"] == "recompute"
    # suffix blocks staged before activation
    staged = [x for x in allr if x["mode"] == "pallas" and x.get("activation_files")]
    # a short window may hold no complete 16-token block; when blocks were moved, all were staged
    assert all(int(x["activation_files_staged"]) == int(x["activation_files"]) > 0 for x in staged), \
        [(x["activation_files_staged"], x["activation_files"]) for x in staged]
    print("activation load (ms):", [round(float(x["activation_load_s"]) * 1000, 2) for x in staged])
    print("OK: all policies ran, next tokens match the source, injected KV is exact, "
          "CVH replay (right / switch / wrong target) behaves as Pallas, suffix staged before activation")
finally:
    for p in procs:
        p.terminate()
    print("logs and results in", tmp)
