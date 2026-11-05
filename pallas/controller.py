"""Experiment controller: mobility prediction, Algorithm 1, and the three handover policies.

Policies (paper Sec. 7 baselines + Pallas):
  pallas     every control cycle: CVH's predicted target and handover time (replayed from nuScenes
             traces) -> Algorithm 1 -> trigger toward the predicted target; prefix prefilled there,
             suffix KV streamed; a changed prediction cancels the attempt and starts a new one;
             at the handover residual + snapshot; target resumes. Not triggered, or triggered
             toward the wrong cell -> post-handover recovery by recomputation
  recompute  at the handover the source sends the full sequence (ids + images); target re-prefills
  fullcopy   at the handover the source sends the full KV over the link; target loads it

Rates used by Pallas' cost model are measured and EWMA-smoothed (paper Sec. 4):
  r_p  target prefill throughput: probed at start, then updated from every observed prefix prefill
  r_d  source decode throughput of this request: from the source's token stream
  B    link goodput: probed at start and between runs
"""
import json
import time
import uuid
from dataclasses import dataclass, field

import requests

from .mobility import prediction_at
from .model import Ewma, PallasParams, kv_bytes_per_token, should_trigger


def post(url, path, body, timeout=600):
    r = requests.post(f"{url}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(url, path, timeout=60):
    r = requests.get(f"{url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


@dataclass
class Rates:
    params: PallasParams
    r_p: Ewma = None
    b: Ewma = None

    def __post_init__(self):
        self.r_p, self.b = Ewma(self.params.ewma), Ewma(self.params.ewma)


@dataclass
class Experiment:
    source: str                     # source cell HTTP url
    target: str                     # target cell HTTP url
    link: str                       # address the source sends to (the link to the target)
    wrong_link: str = ""            # link to a cell outside the experiment (wrong predicted target)
    params: PallasParams = None
    control_dt: float = 0.1         # control-cycle period (s)
    s_kv: float = 0.0
    rates: Rates = None
    epoch: int = field(default=0)

    def setup(self, probe_prefill_tokens, probe_link_mb):
        info = get(self.target, "/info")
        self.s_kv = kv_bytes_per_token(info["layers"], info["kv_heads"], info["head_dim"])   # Eq. 1
        self.rates = Rates(self.params)
        self.probe(probe_prefill_tokens, probe_link_mb, warmup=True)
        return info

    def probe(self, prefill_tokens, link_mb, warmup=False):
        if warmup:                                   # first requests on a fresh engine are slow: discard
            if prefill_tokens:
                post(self.target, "/probe_prefill", {"n_tokens": prefill_tokens})
            if link_mb:
                post(self.source, "/probe_link", {"link": self.link, "nbytes": int(link_mb * 1e6)})
        if prefill_tokens:
            p = post(self.target, "/probe_prefill", {"n_tokens": prefill_tokens})
            self.rates.r_p.update(p["tokens"] / p["seconds"])
        if link_mb:
            p = post(self.source, "/probe_link", {"link": self.link, "nbytes": int(link_mb * 1e6)})
            self.rates.b.update(p["bytes"] * 8 / p["seconds"])

    # --------------------------------------------------------------- one handover
    def run(self, policy, history, handover_after_s, trace, max_tokens):
        """One handover `handover_after_s` into a long answer. `trace` is the prediction replayed
        during the answer (mobility.prediction_traces for CVH on nuScenes, or mobility.oracle_trace)."""
        sid = f"s{uuid.uuid4().hex[:8]}"
        self.epoch += 1
        post(self.source, "/reset", {})
        post(self.target, "/reset", {})
        g = post(self.source, "/generate", {"session": sid, "turns": history, "max_tokens": max_tokens,
                                            "ignore_eos": True})
        t_ho = time.time() + handover_after_s                             # true handover (controller clock)
        row = {"policy": policy, "session": sid, "prompt_tokens": g["prompt_len"], "image_bytes": g["image_bytes"],
               "n_images": g["n_images"], "image_tokens": g["image_tokens"],
               "r_p": self.rates.r_p.value, "B_mbps": self.rates.b.value / 1e6, "s_kv": self.s_kv}
        mode = policy
        if policy == "pallas":
            row.update(cell_m=trace["cell_m"], path=trace["path"], placement=trace["placement"],
                       dwell_s=trace["dwell_s"])
            attempt = self._control_loop(sid, t_ho, trace, row)
            if attempt is None or not attempt["correct"]:
                mode = "recompute"                                         # post-handover recovery
        _sleep_until(t_ho)
        res = self._handover(sid, mode, t_ho)
        row.update(res, triggered=(mode == "pallas") if policy == "pallas" else None)
        n_kv = res["n_kv"] or 0
        row["kv_MB_sent"] = {"pallas": max(0, n_kv - row.get("k", n_kv)), "fullcopy": n_kv}.get(mode, 0) * self.s_kv / 1e6
        info = get(self.target, f"/session/{sid}")
        if info.get("load"):
            row.update(activation_load_s=info["load"]["load_s"], activation_blocks=info["load"]["n_blocks"],
                       activation_files_staged=info["load"]["files_staged_on_gpu"],
                       activation_files=info["load"]["files"])
        if policy == "pallas" and mode == "pallas":
            row["prefix_prefill_s"] = info["prefix_prefill_s"]
            if info["prefix_done_t"] and info["activate_t"]:                  # both on the target's clock
                row["t_early_s"] = max(0.0, info["activate_t"] - info["prefix_done_t"])
            if info["prefix_prefill_s"] and row.get("k"):
                self.rates.r_p.update(row["k"] / info["prefix_prefill_s"])  # observed effective r_p
        return row

    def _control_loop(self, sid, t_ho, trace, row):
        """Every control cycle until the true handover: read the current prediction (target, handover
        time), run Algorithm 1 with T_remain = predicted handover time - now, trigger toward the
        predicted target when T_remain <= T_w*. If the predicted target changes, the attempt is
        cancelled and Algorithm 1 runs again for the new target (new epoch).
        Returns the attempt in place at the true handover, or None."""
        r_d = Ewma(self.params.ewma)
        attempt, n_trig, n_cancel, wrong_sent = None, 0, 0, 0
        t_first = t_ho + trace["ticks"][0][0] if trace["ticks"] else t_ho
        _sleep_until(t_first)
        while time.time() < t_ho - 1e-3:
            now = time.time()
            pred = prediction_at(trace, now - t_ho)
            st = get(self.source, f"/status/{sid}")
            if st["decode_rate"]:
                r_d.update(st["decode_rate"])
            if attempt and pred is not None and pred[0] != attempt["target"]:   # no prediction: keep
                post(self.source, "/cancel", {"session": sid, "epoch": attempt["epoch"]})
                attempt, n_cancel = None, n_cancel + 1
            if attempt is None and pred is not None and r_d.value:
                target, correct, t_pred = pred
                t_remain = max(0.0, t_ho + t_pred - now)                  # predicted, not true
                k_curr = st["prompt_len"] + st["n_out"]
                fire, t_w = should_trigger(t_remain, k_curr, self.rates.r_p.value, r_d.value, self.s_kv,
                                           self.rates.b.value, self.params)
                if fire:
                    self.epoch += 1
                    link = self.link if correct else self.wrong_link
                    t_trig = time.time()
                    ack = post(self.source, "/trigger", {"session": sid, "link": link, "epoch": self.epoch})
                    attempt = {"target": target, "correct": correct, "epoch": self.epoch}
                    n_trig += 1
                    wrong_sent += 0 if correct else ack["prefix_bytes"]
                    row.update(t_w_star_s=t_w, t_remain_pred_s=t_remain, window_s=t_ho - t_trig, k=ack["k"],
                               k_curr=k_curr, r_d=r_d.value, prefix_bytes=ack["prefix_bytes"])
            time.sleep(self.control_dt)
        row.update(r_d=r_d.value, n_triggers=n_trig, n_cancels=n_cancel,
                   wrong_target_at_handover=bool(attempt and not attempt["correct"]),
                   wrong_prefix_MB=wrong_sent / 1e6)
        return attempt

    def _handover(self, sid, mode, t_ho):
        """At t_ho: the source performs the handover and the target is asked to resume, concurrently.
        SIT = handover -> first token emitted by the target (controller clock)."""
        import threading
        out = {}

        def do_handover():
            out["ho"] = post(self.source, "/handover", {"session": sid, "mode": mode, "link": self.link,
                                                        "epoch": self.epoch})
        th = threading.Thread(target=do_handover)
        t0 = time.time()
        th.start()
        first_t, first_tok = None, None
        with requests.post(f"{self.target}/activate", json={"session": sid, "max_tokens": 8},
                           stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=None):
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("tokens") and first_t is None:
                    first_t, first_tok = time.time(), ev["tokens"][0]
        th.join()
        ho = out["ho"]
        return {"mode": mode, "sit_s": first_t - t_ho, "handover_late_s": t0 - t_ho, "n_kv": ho["n_kv"],
                "n_out": ho["n_out"], "target_first_token": first_tok,
                "source_next_token": ho["reference_next_token"],
                "same_next_token": (first_tok == ho["reference_next_token"])
                if ho["reference_next_token"] is not None else None}


def _sleep_until(t):
    time.sleep(max(0.0, t - time.time()))
