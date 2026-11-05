"""PallasConnector: vLLM KV connector implementing Pallas' suffix streaming and injection.

Load it with
  --kv-transfer-config '{"kv_connector": "PallasConnector", "kv_connector_module_path": "pallas.connector",
                         "kv_role": "kv_both", "kv_connector_extra_config": {"ctl_dir": ..., "rx_dir": ...}}'
(pallas/cell.py does this). A request takes part through
  SamplingParams(extra_args={"kv_transfer_params": {"pallas": {"role": "source"|"target", "session": s, "epoch": e}}})

Source (scheduler side, every step): reads the trigger / handover control files written by the cell;
after the trigger, plans the blocks completed by this step; at the handover, plans the residual blocks
and the final snapshot. Source (worker side, after the forward): copies those blocks of every layer
from the GPU KV cache and streams them over the link.

Target (scheduler side): when the activation request arrives, reports the received suffix blocks
beyond the locally prefilled prefix as loadable, or asks the scheduler to retry until the final
snapshot is in. Target (worker side): a thread moves received blocks to GPU memory as they arrive;
before the forward of the activation step they are copied GPU-to-GPU into the blocks vLLM allocated.
"""
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata,
                                                                KVConnectorRole)

from . import wire
from .bookkeeping import (SourceState, apply_trigger, atomic_write_json, plan_handover, plan_load, plan_stream,
                          read_json, rx_index)

log = logging.getLogger("pallas.connector")
FRAME_BLOCKS = 32                                    # KV blocks per link frame


@dataclass
class SendItem:
    session: str
    epoch: int
    link: str
    start_block: int
    block_ids: list


@dataclass
class FinalItem:
    session: str
    epoch: int
    link: str
    snapshot: dict
    sent_flag: str                      # control file the worker writes once the residual is on its way


@dataclass
class LoadItem:
    rx_dir: str
    epoch: int
    from_block: int
    dst_block_ids: list


@dataclass
class PallasMetadata(KVConnectorMetadata):
    sends: list = field(default_factory=list)
    finals: list = field(default_factory=list)
    loads: list = field(default_factory=list)


class _Sender(threading.Thread):
    """One ordered stream of frames per link address (blocks, then the final snapshot)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q, self.socks = queue.Queue(), {}

    def run(self):
        while True:
            link, header, payload, flag = self.q.get()
            try:
                if link not in self.socks:
                    self.socks[link] = wire.connect(link)
                wire.send_frame(self.socks[link], header, payload)
                if flag:
                    atomic_write_json(flag, {"epoch": header["epoch"], "t": time.time()})
            except OSError as e:
                log.error("pallas: send to %s failed: %s", link, e)
                self.socks.pop(link, None)


class _Stager(threading.Thread):
    """Target side: moves received KV blocks to GPU memory as they arrive (during the window), so
    activation only copies them device-to-device into the request's blocks, as in Pallas where the
    migrated blocks are already resident when the target resumes. Bounded by max_bytes; anything
    beyond is loaded from host memory at activation."""

    def __init__(self, rx_root, device, max_bytes, poll_s=0.002):
        super().__init__(daemon=True)
        self.rx_root, self.device, self.max_bytes, self.poll_s = rx_root, device, max_bytes, poll_s
        self.staged, self.used, self.lock = {}, 0, threading.Lock()     # (bin path, epoch) -> (tensors, bytes)

    def get(self, path, epoch):
        with self.lock:
            e = self.staged.get((path, epoch))
        return e[0] if e else None

    def drop(self, paths):
        with self.lock:
            for key in [k for k in self.staged if k[0] in paths]:
                self.used -= self.staged.pop(key)[1]

    def run(self):
        import torch
        stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        while True:
            try:
                self._scan(stream)
            except Exception as e:                       # never take the engine down
                log.warning("pallas: staging error: %s", e)
            time.sleep(self.poll_s)

    def _scan(self, stream):
        import torch
        present = set()
        if not os.path.isdir(self.rx_root):              # briefly absent while the cell resets
            return
        for sess in os.scandir(self.rx_root):
            if not sess.is_dir():
                continue
            for f in os.scandir(sess.path):
                if not (f.name.startswith("kv_") and f.name.endswith(".json")):
                    continue
                meta = read_json(f.path)
                if not meta:
                    continue
                key = (f.path[:-5] + ".bin", meta["epoch"])
                present.add(key)
                if key in self.staged:
                    continue
                size = os.path.getsize(key[0]) if os.path.exists(key[0]) else 0
                if not size or self.used + size > self.max_bytes:
                    continue
                with open(key[0], "rb") as fh:
                    cpu = wire.unpack_blocks(fh.read(), meta)
                if stream is not None:
                    with torch.cuda.stream(stream):
                        gpu = {n: t.pin_memory().to(self.device, non_blocking=True) for n, t in cpu.items()}
                    stream.synchronize()
                else:
                    gpu = cpu
                with self.lock:
                    self.staged[key] = (gpu, size)
                    self.used += size
        with self.lock:                                  # files removed (new epoch, reset): free them
            for key in [k for k in self.staged if k not in present]:
                self.used -= self.staged.pop(key)[1]


class PallasConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config, role, kv_cache_config=None, *args, **kwargs):
        try:
            super().__init__(vllm_config, role, kv_cache_config, *args, **kwargs)
        except TypeError:                                    # older vLLM: (vllm_config, role)
            super().__init__(vllm_config, role)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.ctl_dir, self.rx_root = extra["ctl_dir"], extra["rx_dir"]
        self.wait_timeout_s = float(extra.get("wait_timeout_s", 60.0))
        os.makedirs(self.ctl_dir, exist_ok=True)
        os.makedirs(self.rx_root, exist_ok=True)
        self.block_size = vllm_config.cache_config.block_size
        self.num_blocks = getattr(kv_cache_config, "num_blocks", None)
        # scheduler side
        self.sources: dict[str, SourceState] = {}
        self.pending: dict[str, tuple] = {}
        self.loads_ready: dict[str, tuple] = {}
        self.wait_since: dict[str, float] = {}
        # worker side
        self.kv_caches, self.layers, self.bdim = None, None, None
        self.sender = self.stager = None
        self.stage_bytes = int(float(extra.get("stage_gpu_mb", 2048)) * 1e6)   # 0 disables staging

    # ------------------------------------------------------------------ scheduler side
    @staticmethod
    def _params(request):
        p = getattr(request, "kv_transfer_params", None) or {}
        return p.get("pallas")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        p = self._params(request)
        if not p:
            return 0, False
        if p["role"] == "source":
            self.sources.setdefault(request.request_id, SourceState(p["session"], request))
            return 0, False
        rx_dir = os.path.join(self.rx_root, p["session"])
        plan = plan_load(rx_dir, int(p["epoch"]), num_computed_tokens, len(request.prompt_token_ids),
                         self.block_size)
        if plan is None:                                     # snapshot not complete yet: retry later
            t0 = self.wait_since.setdefault(request.request_id, time.time())
            if time.time() - t0 < self.wait_timeout_s:
                return None, False
            log.error("pallas: %s gave up waiting for the suffix, recomputing", p["session"])
            return 0, False
        self.wait_since.pop(request.request_id, None)
        n_ext, fb, tb = plan
        if n_ext > 0:
            self.pending[request.request_id] = (rx_dir, int(p["epoch"]), fb, tb)
        return n_ext, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0 and request.request_id in self.pending:
            self.loads_ready[request.request_id] = self.pending.pop(request.request_id)

    def _track_blocks(self, scheduler_output):
        for nr in scheduler_output.scheduled_new_reqs:
            if nr.req_id in self.sources:
                self.sources[nr.req_id].block_ids = list(nr.block_ids[0])
        cached = scheduler_output.scheduled_cached_reqs
        if hasattr(cached, "req_ids"):                       # vLLM >= 0.10 layout
            resumed = getattr(cached, "resumed_req_ids", set()) or set()
            if not resumed and hasattr(cached, "resumed_from_preemption"):
                resumed = {r for r, f in zip(cached.req_ids, cached.resumed_from_preemption) if f}
            items = [(r, cached.new_block_ids[i], r in resumed) for i, r in enumerate(cached.req_ids)]
        else:                                                # older: list of per-request objects
            items = [(c.req_id, c.new_block_ids, c.resumed_from_preemption) for c in cached]
        for rid, nb, resumed in items:
            if rid in self.sources and nb is not None:
                if resumed:
                    self.sources[rid].block_ids = list(nb[0])
                else:
                    self.sources[rid].block_ids.extend(nb[0])

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = PallasMetadata()
        self._track_blocks(scheduler_output)
        for nr in scheduler_output.scheduled_new_reqs:
            if nr.req_id in self.loads_ready:
                rx_dir, epoch, fb, tb = self.loads_ready.pop(nr.req_id)
                meta.loads.append(LoadItem(rx_dir, epoch, fb, list(nr.block_ids[0][fb:tb])))
        nsched = scheduler_output.num_scheduled_tokens
        for rid, st in list(self.sources.items()):
            ctl = os.path.join(self.ctl_dir, st.session)
            ack = apply_trigger(st, read_json(ctl + ".trigger"), self.block_size)
            if ack is not None:
                atomic_write_json(ctl + ".ack", ack)
            sent = plan_stream(st, nsched.get(rid, 0), self.block_size)
            if sent:
                meta.sends.append(SendItem(st.session, st.epoch, st.link, sent[0], sent[1]))
            if st.triggered and not st.handed_over and os.path.exists(ctl + ".handover"):
                residual, snapshot = plan_handover(st, nsched.get(rid, 0), self.block_size)
                if residual:
                    meta.sends.append(SendItem(st.session, st.epoch, st.link, residual[0], residual[1]))
                atomic_write_json(ctl + ".snapshot", snapshot)
                meta.finals.append(FinalItem(st.session, st.epoch, st.link, snapshot, ctl + ".sent"))
        return meta

    def request_finished(self, request, block_ids):
        self.sources.pop(request.request_id, None)
        self.pending.pop(request.request_id, None)
        self.loads_ready.pop(request.request_id, None)
        self.wait_since.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        self.layers = sorted(kv_caches)
        first = kv_caches[self.layers[0]]
        n = self.num_blocks or max(first.shape)
        self.bdim = wire.block_dim(first, n)
        self.sender = _Sender()
        self.sender.start()
        if self.stage_bytes > 0:
            self.stager = _Stager(self.rx_root, first.device, self.stage_bytes)
            self.stager.start()
        log.info("pallas: %d KV layers, block dim %d, shape %s", len(self.layers), self.bdim, tuple(first.shape))

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        for item in meta.loads:
            self._load(item)

    def _load(self, item):
        """Copy the received blocks into the blocks vLLM allocated. Blocks already staged on the GPU
        (the usual case: they arrived during the window) are a device-to-device copy."""
        import torch
        t0 = time.time()
        have = rx_index(item.rx_dir, item.epoch)
        files, by_layer, staged = {}, {name: [] for name in self.layers}, 0
        for i in range(len(item.dst_block_ids)):
            path, pos = have[item.from_block + i]
            if path not in files:
                files[path] = self.stager.get(path, item.epoch) if self.stager else None
                if files[path] is None:
                    with open(path, "rb") as f:
                        files[path] = wire.unpack_blocks(f.read(), read_json(path[:-4] + ".json"))
                else:
                    staged += 1
            for name in self.layers:
                by_layer[name].append(files[path][name][pos])
        wire.write_blocks(self.kv_caches, self.bdim, {n: torch.stack(v) for n, v in by_layer.items()},
                          item.dst_block_ids)
        if torch.cuda.is_available() and self.kv_caches[self.layers[0]].is_cuda:
            torch.cuda.synchronize()
        atomic_write_json(os.path.join(item.rx_dir, "loaded.json"),
                          {"epoch": item.epoch, "n_blocks": len(item.dst_block_ids), "files": len(files),
                           "files_staged_on_gpu": staged, "load_s": time.time() - t0})
        if self.stager:
            self.stager.drop(files)

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        """After the forward of this step: copy the planned blocks to host memory, then stream them."""
        if not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        for s in meta.sends:                         # frames of <= FRAME_BLOCKS blocks, so the link,
            for i in range(0, len(s.block_ids), FRAME_BLOCKS):   # the target's store and staging overlap
                ids = s.block_ids[i:i + FRAME_BLOCKS]
                payload, m = wire.pack_blocks(self.kv_caches, self.layers, self.bdim, ids)
                header = dict(m, type="kv", session=s.session, epoch=s.epoch, start_block=s.start_block + i)
                self.sender.q.put((s.link, header, payload, None))
        for f in meta.finals:
            header = dict(f.snapshot, type="final", session=f.session)
            self.sender.q.put((f.link, header, b"", f.sent_flag))
            atomic_write_json(f.sent_flag[:-len(".sent")] + ".copied", {"epoch": f.epoch, "t": time.time()})
