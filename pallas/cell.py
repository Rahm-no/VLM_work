#!/usr/bin/env python3
"""A cell (gNB + colocated GPU): one vLLM AsyncLLM engine with the PallasConnector, an HTTP control
API for the controller, and a data port that receives the link's messages.

  python3 -m pallas.cell --name source --model Qwen/Qwen2-VL-7B-Instruct --http-port 8100 --data-port 9101
  python3 -m pallas.cell --name target --model Qwen/Qwen2-VL-7B-Instruct --http-port 8200 --data-port 9201

Control API (JSON over HTTP; control traffic is not on the measured link):
  source  POST /generate   start an answer (or serve a history turn)            -> prompt length
          GET  /status/{s} tokens so far and the source's decode rate
          POST /trigger    Pallas trigger: snap to block k, send the prefix over the link, start streaming
          POST /handover   pallas | fullcopy: final step, residual + snapshot over the link, then stop
                           recompute: stop, send the whole sequence + images over the link
          POST /probe_link send N bytes over the link and time them (goodput B)
  target  POST /activate   wait until the state is complete, resume decoding; streams tokens (NDJSON)
          GET  /session/{s} prefix prefill timing at this cell
          POST /probe_prefill prefill throughput r_p on a prompt of N text tokens
  both    POST /reset, GET /info
"""
import argparse
import asyncio
import io
import json
import os
import shutil
import socket
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from . import rx, wire
from .bookkeeping import atomic_write_json, read_json

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--http-port", type=int, required=True)
ap.add_argument("--data-port", type=int, required=True)
ap.add_argument("--state-root", default="/dev/shm/pallas", help="control and received-KV files")
ap.add_argument("--max-model-len", type=int, default=16384)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
ap.add_argument("--max-images", type=int, default=8)
ap.add_argument("--mm-processor-kwargs", default=None, help='JSON, e.g. {"max_pixels": 3211264}')
ap.add_argument("--enforce-eager", action="store_true")
ap.add_argument("--wait-timeout", type=float, default=60.0, help="max wait for the suffix at activation (s)")
ap.add_argument("--stage-gpu-mb", type=float, default=2048,
                help="GPU memory for received KV blocks staged before activation (0: load from host at activation)")
ARGS = ap.parse_args()

STATE = os.path.join(ARGS.state_root, ARGS.name)
CTL, RX = os.path.join(STATE, "ctl"), os.path.join(STATE, "rx")
app = FastAPI()
ENGINE = PROCESSOR = None
SESS: dict[str, dict] = {}


# ------------------------------------------------------------------ engine and prompts
def build_engine():
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    kv = KVTransferConfig(kv_connector="PallasConnector", kv_connector_module_path="pallas.connector",
                          kv_role="kv_both",
                          kv_connector_extra_config={"ctl_dir": CTL, "rx_dir": RX, "wait_timeout_s": ARGS.wait_timeout,
                                                     "stage_gpu_mb": ARGS.stage_gpu_mb})
    eargs = AsyncEngineArgs(model=ARGS.model, max_model_len=ARGS.max_model_len,
                            gpu_memory_utilization=ARGS.gpu_memory_utilization, enable_prefix_caching=True,
                            limit_mm_per_prompt={"image": ARGS.max_images}, enforce_eager=ARGS.enforce_eager,
                            mm_processor_kwargs=json.loads(ARGS.mm_processor_kwargs) if ARGS.mm_processor_kwargs else None,
                            kv_transfer_config=kv,
                            # vLLM >= 0.11 schedules step N+1 before step N's token is known; the
                            # final token snapshot needs every token whose KV exists (Pallas ran on
                            # vLLM 0.8.5, which has no async scheduling)
                            async_scheduling=False)
    return AsyncLLM.from_engine_args(eargs)


def _turn_images(t):
    """Image files of a user turn: "images" (list, e.g. 6 cameras) or "image" (one)."""
    return list(t.get("images") or ([t["image"]] if t.get("image") else []))


def base_prompt_ids(turns):
    """Chat template -> token ids with ONE placeholder per image (vLLM expands them, identically at
    both cells, from the images)."""
    msgs = []
    for t in turns:
        if t["role"] == "user":
            content = [{"type": "image"}] * len(_turn_images(t)) + [{"type": "text", "text": t["text"]}]
        else:
            content = [{"type": "text", "text": t["text"]}]
        msgs.append({"role": t["role"], "content": content})
    text = PROCESSOR.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return PROCESSOR.tokenizer.encode(text, add_special_tokens=False)


def pil(images_bytes):
    from PIL import Image
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]


def prompt(ids, images_bytes):
    p = {"prompt_token_ids": list(ids)}
    if images_bytes:
        p["multi_modal_data"] = {"image": pil(images_bytes)}
    return p


def sampling(max_tokens, ignore_eos=False, kv_params=None):
    if os.environ.get("PALLAS_FAKE_ENGINE"):
        from types import SimpleNamespace
        return SimpleNamespace(max_tokens=max_tokens, extra_args={"kv_transfer_params": {"pallas": kv_params}}
                               if kv_params else None)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    extra = {"kv_transfer_params": {"pallas": kv_params}} if kv_params else None
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=ignore_eos,
                          output_kind=RequestOutputKind.DELTA, extra_args=extra)


def link_send(link, header, payload=b"", wait_ack=False):
    s = wire.connect(link)
    try:
        wire.send_frame(s, header, payload)
        if wait_ack:                                   # target acknowledges once it has every byte
            if s.recv(1) != b"\x01":
                raise ConnectionError("no ack from target")
    finally:
        s.close()


async def wait_file(path, timeout=10.0, epoch=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (d := read_json(path)) is not None and (epoch is None or d.get("epoch") == epoch):
            return d
        await asyncio.sleep(0.002)
    raise TimeoutError(path)


# ------------------------------------------------------------------ source role
async def _run_answer(sid, ids, images, max_tokens, ignore_eos, as_source):
    s = SESS[sid]
    params = {"role": "source", "session": sid} if as_source else None
    async for out in ENGINE.generate(prompt(ids, images), sampling(max_tokens, ignore_eos, params), s["rid"]):
        if s["prompt_len"] is None and out.prompt_token_ids is not None:
            s["prompt_len"] = len(out.prompt_token_ids)
        now = time.time()
        for tok in out.outputs[0].token_ids:
            s["out"].append(tok)
            s["t"].append(now)
        if out.finished:
            break
    s["done"].set()


@app.post("/generate")
async def generate(req: dict):
    """{session, turns:[{role,text,image?}], max_tokens, ignore_eos, wait}"""
    sid = req["session"]
    images = [open(f, "rb").read() for t in req["turns"] if t["role"] == "user" for f in _turn_images(t)]
    ids = base_prompt_ids(req["turns"])
    SESS[sid] = {"rid": f"{sid}-{uuid.uuid4().hex[:6]}", "base_ids": ids, "images": images, "out": [], "t": [],
                 "prompt_len": None, "done": asyncio.Event(), "epoch": 0}
    task = asyncio.create_task(_run_answer(sid, ids, images, req["max_tokens"], req.get("ignore_eos", False),
                                           as_source=not req.get("wait", False)))
    SESS[sid]["task"] = task
    if req.get("wait"):
        await SESS[sid]["done"].wait()
        return {"text": PROCESSOR.tokenizer.decode(SESS[sid]["out"], skip_special_tokens=True),
                "prompt_len": SESS[sid]["prompt_len"]}
    while not SESS[sid]["out"]:
        await asyncio.sleep(0.002)
    p_len = SESS[sid]["prompt_len"]
    return {"prompt_len": p_len, "base_len": len(ids), "image_bytes": sum(map(len, images)), "n_images": len(images),
            "image_tokens": p_len - len(ids) + len(images),    # each image placeholder expands to its tokens
            "t_first": SESS[sid]["t"][0]}


@app.get("/status/{sid}")
async def status(sid: str, window: float = 1.0):
    s = SESS[sid]
    now = time.time()
    span = min(window, now - s["t"][0]) if s["t"] else 0.0
    recent = sum(1 for t in s["t"] if t >= now - window)
    return {"n_out": len(s["out"]), "prompt_len": s["prompt_len"],
            "decode_rate": recent / span if span >= 0.2 else None, "now": now}


@app.post("/trigger")
async def trigger(req: dict):
    """{session, link, epoch}: Pallas trigger toward the predicted target (reached through `link`).
    The connector snaps to block k on its next step. A later trigger with a new epoch replaces this
    attempt (the predicted target changed)."""
    sid, s = req["session"], SESS[req["session"]]
    s["epoch"], s["link"] = req["epoch"], req["link"]
    for suffix in (".snapshot", ".sent", ".copied", ".handover"):
        _rm(os.path.join(CTL, sid + suffix))
    atomic_write_json(os.path.join(CTL, sid + ".trigger"), {"epoch": req["epoch"], "link": req["link"]})
    ack = await wait_file(os.path.join(CTL, sid + ".ack"), epoch=req["epoch"])
    payload = b"".join(s["images"])
    header = {"type": "prefix", "session": sid, "epoch": req["epoch"], "prefill": True,
              "base_ids": s["base_ids"], "out_ids": ack["prefix_output_ids"],
              "image_lens": [len(b) for b in s["images"]]}
    t0 = time.time()
    await asyncio.to_thread(link_send, req["link"], header, payload)       # exact inputs of [0, k)
    return dict(ack, prefix_bytes=len(payload) + 4 * (len(s["base_ids"]) + len(ack["prefix_output_ids"])),
                prefix_sent_s=time.time() - t0)


@app.post("/cancel")
async def cancel(req: dict):
    """{session, epoch}: the prediction no longer names this attempt's target; stop streaming."""
    sid = req["session"]
    atomic_write_json(os.path.join(CTL, sid + ".trigger"), {"epoch": req["epoch"], "cancel": True})
    await wait_file(os.path.join(CTL, sid + ".ack"), epoch=req["epoch"])
    return {"ok": True}


@app.post("/handover")
async def handover(req: dict):
    """{session, mode: pallas|fullcopy|recompute, link, epoch}"""
    sid, s, mode = req["session"], SESS[req["session"]], req["mode"]
    t_ho = time.time()
    if mode in ("pallas", "fullcopy"):
        if mode == "fullcopy":                                              # all KV after the handover
            s["epoch"], s["link"] = req["epoch"], req["link"]
            for suffix in (".ack", ".snapshot", ".sent", ".copied"):
                _rm(os.path.join(CTL, sid + suffix))
            meta = {"type": "prefix", "session": sid, "epoch": req["epoch"], "prefill": False,
                    "base_ids": s["base_ids"], "out_ids": [], "image_lens": [len(b) for b in s["images"]]}
            await asyncio.to_thread(link_send, req["link"], meta, b"".join(s["images"]))
            atomic_write_json(os.path.join(CTL, sid + ".trigger"),
                              {"epoch": req["epoch"], "link": req["link"], "from_block": 0})
        atomic_write_json(os.path.join(CTL, sid + ".handover"), {"epoch": s["epoch"]})
        snap = await wait_file(os.path.join(CTL, sid + ".snapshot"))
        await wait_file(os.path.join(CTL, sid + ".copied"))                 # residual is off the GPU
        n_out = snap["n_kv"] - snap["num_prompt_tokens"]
        # Reference for the correctness check only: the token the source's handover step produces
        # (what the target must produce from the migrated state). The target's activation does not
        # wait for this; it is triggered by the snapshot on the link.
        t_ref = time.time()
        while len(s["out"]) <= n_out and not s["done"].is_set() and time.time() - t_ref < 5.0:
            await asyncio.sleep(0.001)
        await ENGINE.abort(s["rid"])                                        # source stops serving
    else:                                                                   # recompute
        out_ids = list(s["out"])                                            # tokens at the handover
        header = {"type": "full", "session": sid, "epoch": req["epoch"], "base_ids": s["base_ids"],
                  "out_ids": out_ids, "image_lens": [len(b) for b in s["images"]]}
        send = asyncio.create_task(asyncio.to_thread(link_send, req["link"], header, b"".join(s["images"])))
        n_out, t_ref = len(out_ids), time.time()                            # reference: the next token
        while len(s["out"]) <= n_out and not s["done"].is_set() and time.time() - t_ref < 5.0:
            await asyncio.sleep(0.001)
        await ENGINE.abort(s["rid"])
        await send
        snap = {"n_kv": None, "output_ids": out_ids}
    reference = s["out"][n_out] if len(s["out"]) > n_out else None         # source's next token
    return {"t_handover_cell": t_ho, "n_kv": snap["n_kv"], "n_out": n_out, "reference_next_token": reference}


@app.post("/probe_link")
async def probe_link(req: dict):
    """{link, nbytes}: goodput measurement for Pallas' B."""
    payload = os.urandom(int(req["nbytes"]))
    t0 = time.time()
    await asyncio.to_thread(link_send, req["link"], {"type": "probe", "session": "_probe", "epoch": 0}, payload, True)
    return {"bytes": len(payload), "seconds": time.time() - t0}


# ------------------------------------------------------------------ target role
def _rx(sid):
    return SESS.setdefault(sid, {"ready": asyncio.Event(), "prefix_done": asyncio.Event()})


async def _prefill_prefix(sid, ids, images):
    s = SESS[sid]
    t0 = time.time()
    async for _ in ENGINE.generate(prompt(ids, images), sampling(1), f"{sid}-prefix-{uuid.uuid4().hex[:6]}"):
        pass
    s.update(prefix_prefill_s=time.time() - t0, prefix_done_t=time.time(), prefix_tokens=None)
    s["prefix_done"].set()


def _accept_epoch(s, d, epoch):
    """Pallas epochs: a newer attempt replaces older state; frames of an older attempt are dropped.
    Frames of one attempt may arrive in any order (prefix and KV use separate connections)."""
    cur = s.get("epoch")
    if cur is not None and epoch < cur:
        return False
    if cur is None or epoch > cur:
        s["epoch"] = epoch
        s["ready"].clear()
        s["prefix_done"].clear()
        for k in ("final", "base_ids", "out_ids", "images", "kind", "prefix_prefill_s", "prefix_done_t"):
            s.pop(k, None)
        for f in os.listdir(d):
            _rm(os.path.join(d, f))
    return True


async def on_frame(header, payload):
    sid, typ = header.get("session"), header["type"]
    if typ == "probe":
        return
    d = os.path.join(RX, sid)
    os.makedirs(d, exist_ok=True)
    s = _rx(sid)
    if not _accept_epoch(s, d, header["epoch"]):
        return
    if typ in ("prefix", "full"):
        imgs, off = [], 0
        for n in header["image_lens"]:
            imgs.append(payload[off:off + n])
            off += n
        s.update(base_ids=header["base_ids"], out_ids=header["out_ids"], images=imgs, kind=typ,
                 recv_t=time.time())
        if typ == "prefix" and header.get("prefill"):
            asyncio.create_task(_prefill_prefix(sid, header["base_ids"] + header["out_ids"], imgs))
        else:
            s["prefix_done"].set()
        if typ == "full":
            s["ready"].set()
    elif typ == "kv":
        rx.store_kv(d, header, payload)
    elif typ == "final":
        rx.store_final(d, header)
        s.update(final=header, final_t=time.time())
        s["ready"].set()


async def data_server():
    async def handle(reader, writer):
        try:
            while True:
                header, payload = await wire.read_frame(reader)
                await on_frame(header, payload)
                if header["type"] == "probe":
                    writer.write(b"\x01")
                    await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
    srv = await asyncio.start_server(handle, "0.0.0.0", ARGS.data_port)
    async with srv:
        await srv.serve_forever()


@app.post("/activate")
async def activate(req: dict):
    """{session, max_tokens}: wait for the prefix and the final snapshot (or the full sequence),
    then resume decoding. Streams NDJSON lines {"t": wall time, "tokens": [...]}."""
    sid = req["session"]
    s = _rx(sid)
    s["activate_t"] = time.time()

    async def body():
        await s["ready"].wait()
        await s["prefix_done"].wait()
        out_ids = s["final"]["output_ids"] if s.get("kind") == "prefix" else s["out_ids"]
        ids = s["base_ids"] + out_ids
        params = {"role": "target", "session": sid, "epoch": s["epoch"]} if s.get("kind") == "prefix" else None
        t_submit = time.time()
        yield json.dumps({"event": "submit", "t": t_submit, "tokens_in": len(ids)}) + "\n"
        async for out in ENGINE.generate(prompt(ids, s["images"]), sampling(req.get("max_tokens", 8), False, params),
                                         f"{sid}-act-{uuid.uuid4().hex[:6]}"):
            yield json.dumps({"t": time.time(), "tokens": list(out.outputs[0].token_ids)}) + "\n"
    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.get("/session/{sid}")
async def session_info(sid: str):
    s = SESS.get(sid, {})
    out = {k: s.get(k) for k in ("prefix_prefill_s", "prefix_done_t", "recv_t", "final_t", "activate_t", "epoch", "kind")}
    out["load"] = read_json(os.path.join(RX, sid, "loaded.json"))   # suffix injection at activation
    return out


@app.post("/probe_prefill")
async def probe_prefill(req: dict):
    """{n_tokens}: cold prefill of n text tokens -> tokens/s (Pallas' r_p), under the cell's current load."""
    await _stop_load()
    await _reset_prefix_cache()
    _start_load()
    vocab = PROCESSOR.tokenizer.vocab_size
    ids = [(i * 7919) % (vocab - 1000) + 500 for i in range(int(req["n_tokens"]))]
    t0 = time.time()
    async for _ in ENGINE.generate({"prompt_token_ids": ids}, sampling(1), f"probe-{uuid.uuid4().hex[:6]}"):
        pass
    dt = time.time() - t0
    return {"tokens": len(ids), "seconds": dt}


# ------------------------------------------------------------------ both
def _rm(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


LOAD = {"n": 0, "prompt_tokens": 1000, "max_tokens": 256, "tasks": [], "rids": set()}


async def _load_user(i):
    """One other user of this cell: back-to-back text requests (prompt, then a long answer)."""
    vocab = PROCESSOR.tokenizer.vocab_size
    k = 0
    while True:
        rid = f"load-{i}-{k}-{uuid.uuid4().hex[:6]}"
        k += 1
        ids = [(j * 7919 + i * 104729 + k * 1299709) % (vocab - 1000) + 500 for j in range(LOAD["prompt_tokens"])]
        LOAD["rids"].add(rid)
        try:
            async for _ in ENGINE.generate({"prompt_token_ids": ids}, sampling(LOAD["max_tokens"], True), rid):
                pass
        finally:
            LOAD["rids"].discard(rid)


async def _stop_load():
    for t in LOAD["tasks"]:
        t.cancel()
    for t in LOAD["tasks"]:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    for rid in list(LOAD["rids"]):
        await ENGINE.abort(rid)
    LOAD["tasks"], LOAD["rids"] = [], set()


def _start_load():
    LOAD["tasks"] = [asyncio.create_task(_load_user(i)) for i in range(LOAD["n"])]


@app.post("/load")
async def load(req: dict):
    """{n, prompt_tokens, max_tokens}: n other users keep this cell busy (n = 0 stops them)."""
    await _stop_load()
    LOAD.update(n=int(req.get("n", 0)), prompt_tokens=int(req.get("prompt_tokens", 1000)),
                max_tokens=int(req.get("max_tokens", 256)))
    _start_load()
    return {"n": LOAD["n"]}


async def _reset_prefix_cache():
    """vLLM only resets the prefix cache when no request is running: retry until it does, so no run
    reuses the previous run's prefix."""
    for _ in range(200):
        ok = await ENGINE.reset_prefix_cache()
        if ok is None or ok:
            return True
        await asyncio.sleep(0.01)
    return False


@app.post("/reset")
async def reset():
    await _stop_load()                                   # running requests block the cache reset
    for s in SESS.values():
        if s.get("rid") and not s.get("done", asyncio.Event()).is_set():
            await ENGINE.abort(s["rid"])
    SESS.clear()
    ok = await _reset_prefix_cache()
    for d in (CTL, RX):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    _start_load()
    return {"ok": True, "prefix_cache_reset": ok}


@app.get("/info")
async def info():
    if os.environ.get("PALLAS_FAKE_ENGINE"):
        return {"name": ARGS.name, "model": "fake", "layers": 2, "kv_heads": 2, "head_dim": 4,
                "data_port": ARGS.data_port, "host": socket.gethostname(), "kv_mismatch": ENGINE.kv_mismatch}
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(ARGS.model)
    c = getattr(c, "text_config", None) or c
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return {"name": ARGS.name, "model": ARGS.model, "layers": c.num_hidden_layers,
            "kv_heads": getattr(c, "num_key_value_heads", None) or c.num_attention_heads,
            "head_dim": head_dim, "data_port": ARGS.data_port, "host": socket.gethostname()}


@asynccontextmanager
async def lifespan(_app):
    global ENGINE, PROCESSOR
    os.makedirs(CTL, exist_ok=True)
    os.makedirs(RX, exist_ok=True)
    if os.environ.get("PALLAS_FAKE_ENGINE"):                                # whole-system test, no GPU
        from tests.fake_engine import FakeEngine, FakeProcessor
        PROCESSOR, ENGINE = FakeProcessor(), FakeEngine(CTL, RX, ARGS.wait_timeout)
    else:
        from transformers import AutoProcessor
        PROCESSOR = AutoProcessor.from_pretrained(ARGS.model)
        ENGINE = build_engine()
    task = asyncio.create_task(data_server())
    yield
    task.cancel()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ARGS.http_port, log_level="warning")
