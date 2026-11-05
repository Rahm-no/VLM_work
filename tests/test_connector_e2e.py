"""End to end through PallasConnector without a GPU: vLLM's base class is stubbed, everything else is
the real code (scheduler + worker sides, wire format, throttled link relay, receiver)."""
import asyncio
import os
import random
import socket
import sys
import threading
import time
import types
from types import SimpleNamespace

import torch

# ---- stub only vLLM's connector base class -------------------------------------------------
base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")


class KVConnectorMetadata:
    pass


class KVConnectorBase_V1:
    def __init__(self, vllm_config, role, kv_cache_config=None):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._connector_metadata = None

    def bind_connector_metadata(self, m):
        self._connector_metadata = m

    def clear_connector_metadata(self):
        self._connector_metadata = None

    def _get_connector_metadata(self):
        return self._connector_metadata

    def has_connector_metadata(self):
        return self._connector_metadata is not None


base.KVConnectorBase_V1, base.KVConnectorMetadata, base.KVConnectorRole = KVConnectorBase_V1, KVConnectorMetadata, object
for name in ["vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
             "vllm.distributed.kv_transfer.kv_connector.v1"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules[base.__name__] = base

from pallas import link, rx, wire  # noqa: E402
from pallas.bookkeeping import read_json  # noqa: E402
from pallas.connector import PallasConnector  # noqa: E402

BS, NB, LAYERS = 16, 64, 3


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def cfg(root):
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=BS), kv_transfer_config=SimpleNamespace(
        is_kv_producer=True,
        kv_connector_extra_config={"ctl_dir": f"{root}/ctl", "rx_dir": f"{root}/rx", "wait_timeout_s": 5}))


def caches(seed):
    g = torch.Generator().manual_seed(seed)
    return {f"model.layers.{i}.attn": torch.randn((2, NB, BS, 2, 8), generator=g).to(torch.bfloat16)
            for i in range(LAYERS)}


def receiver(port, rx_root):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen()
    while True:
        c, _ = srv.accept()

        def handle(c=c):
            try:
                while True:
                    h, p = wire.recv_frame(c)
                    d = os.path.join(rx_root, h["session"])
                    (rx.store_kv(d, h, p) if h["type"] == "kv" else rx.store_final(d, h))
            except ConnectionError:
                pass
        threading.Thread(target=handle, daemon=True).start()


def sched_out(new=(), cached=(), nsched=None):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=r, block_ids=(b,)) for r, b in new],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[r for r, _ in cached], new_block_ids=[n for _, n in cached],
                                              resumed_req_ids=set()),
        num_scheduled_tokens=nsched or {})


def test_suffix_kv_streams_over_link_and_lands_in_target_blocks(tmp_path):
    src_root, tgt_root = str(tmp_path / "src"), str(tmp_path / "tgt")
    data_port, link_port, ctl_port = _port(), _port(), _port()
    threading.Thread(target=receiver, args=(data_port, f"{tgt_root}/rx"), daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(link.serve(f"127.0.0.1:{link_port}", f"127.0.0.1:{data_port}",
                                                           lambda t: 200e6, 0.0, ctl_port)), daemon=True).start()
    time.sleep(0.3)

    kvc = SimpleNamespace(num_blocks=NB)
    s_sched, s_work = PallasConnector(cfg(src_root), "s", kvc), PallasConnector(cfg(src_root), "w", kvc)
    src = caches(1)
    s_work.register_kv_caches(src)

    P = 100
    req = SimpleNamespace(request_id="r1", kv_transfer_params={"pallas": {"role": "source", "session": "S"}},
                          num_prompt_tokens=P, prompt_token_ids=list(range(P)), all_token_ids=list(range(P)),
                          num_computed_tokens=0)
    pool = random.Random(0).sample(range(NB), NB)
    blocks = [pool.pop() for _ in range(-(-P // BS))]
    assert s_sched.get_num_new_matched_tokens(req, 0) == (0, False)

    def step(new_req, n_tokens, new_blocks):
        so = sched_out(new=[("r1", list(blocks))] if new_req else (),
                       cached=[] if new_req else [("r1", (new_blocks,) if new_blocks else None)],
                       nsched={"r1": n_tokens})
        meta = s_sched.build_connector_meta(so)
        s_work.bind_connector_metadata(meta)
        s_work.wait_for_save()                                   # after the forward: copy + stream
        s_work.clear_connector_metadata()
        req.num_computed_tokens += n_tokens
        req.all_token_ids.append(9000 + len(req.all_token_ids))  # sampled token

    step(True, P, None)                                          # prefill
    ctl = f"{src_root}/ctl/S"
    for i in range(80):                                          # decode
        nb = []
        if len(req.all_token_ids) > len(blocks) * BS:
            nb = [pool.pop()]
            blocks.extend(nb)
        if i == 20:
            from pallas.bookkeeping import atomic_write_json
            atomic_write_json(ctl + ".trigger", {"epoch": 1, "link": f"127.0.0.1:{link_port}"})
        if i == 70:
            atomic_write_json(ctl + ".handover", {"epoch": 1})
        step(False, 1, nb)
    ack, snap = read_json(ctl + ".ack"), read_json(ctl + ".snapshot")
    k, n_kv = ack["k"], snap["n_kv"]
    assert k % BS == 0 and k == ((P + 20) // BS) * BS
    assert snap["output_ids"] == req.all_token_ids[P:n_kv]

    for _ in range(100):                                         # wait for the link to deliver
        if os.path.exists(f"{tgt_root}/rx/S/final.json"):
            break
        time.sleep(0.05)

    # ---- target: prefix [0, k) prefilled locally (simulated), suffix loaded by the connector
    t_sched, t_work = PallasConnector(cfg(tgt_root), "s", kvc), PallasConnector(cfg(tgt_root), "w", kvc)
    tgt = {n: torch.zeros_like(t) for n, t in src.items()}
    t_work.register_kv_caches(tgt)
    tpool = random.Random(7).sample(range(NB), NB)
    tblocks = [tpool.pop() for _ in range(-(-n_kv // BS))]
    treq = SimpleNamespace(request_id="t1", kv_transfer_params={"pallas": {"role": "target", "session": "S", "epoch": 1}},
                           prompt_token_ids=req.all_token_ids[:n_kv])
    n_ext, async_ = t_sched.get_num_new_matched_tokens(treq, k)
    to_block = (n_kv - 1) // BS
    assert to_block - k // BS >= 3                               # the test really moves blocks
    assert (n_ext, async_) == (to_block * BS - k, False)
    t_sched.update_state_after_alloc(treq, None, n_ext)
    meta = t_sched.build_connector_meta(sched_out(new=[("t1", tblocks)]))
    t_work.bind_connector_metadata(meta)
    t_work.start_load_kv(None)
    for name in src:
        for b in range(k // BS, to_block):
            assert torch.equal(tgt[name][:, tblocks[b]], src[name][:, blocks[b]]), (name, b)
        assert not tgt[name][:, tblocks[0]].any()                # prefix blocks untouched by the load
