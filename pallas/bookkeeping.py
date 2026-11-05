"""Pure bookkeeping for the Pallas connector (no vLLM / torch imports, unit-tested).

Source side: which KV blocks to stream after the trigger, and what to send at the handover.
Target side: how many tokens can be loaded beyond the locally prefilled prefix.
"""
import glob
import json
import os
from dataclasses import dataclass, field


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass
class SourceState:
    session: str
    request: object                         # live vllm Request: num_computed_tokens, all_token_ids, ...
    block_ids: list = field(default_factory=list)
    epoch: int = -1
    link: str = ""
    next_block: int = 0                     # first block not yet streamed
    triggered: bool = False
    handed_over: bool = False
    trigger_key: tuple = None               # (epoch, cancel) of the last trigger file applied


def apply_trigger(st, trigger, block_size):
    """Apply a trigger file once. A new epoch re-triggers (the predicted target changed: Pallas
    cancels the old attempt and starts a new one); {"cancel": true} stops streaming.
    Returns the ack to write, or None."""
    if trigger is None or st.handed_over:
        return None
    key = (int(trigger["epoch"]), bool(trigger.get("cancel")))
    if key == st.trigger_key:
        return None
    st.trigger_key = key
    if trigger.get("cancel"):
        st.triggered = False
        return {"epoch": key[0], "cancelled": True}
    return plan_trigger(st, trigger, block_size)


def plan_trigger(st, trigger, block_size):
    """At the trigger, the source snaps to the last complete block: prefix = [0, k).
    trigger["from_block"] = 0 forces streaming from the start (Full-Copy baseline)."""
    computed = st.request.num_computed_tokens
    k = (computed // block_size) * block_size
    if trigger.get("from_block") is not None:
        st.next_block = int(trigger["from_block"])
    else:
        st.next_block = k // block_size
    st.epoch, st.link, st.triggered = int(trigger["epoch"]), trigger["link"], True
    n_prompt = st.request.num_prompt_tokens
    j = max(0, k - n_prompt)
    return {"epoch": st.epoch, "k": k, "j": j, "num_prompt_tokens": n_prompt,
            "prefix_output_ids": list(st.request.all_token_ids[n_prompt:n_prompt + j])}


def plan_stream(st, num_scheduled, block_size):
    """Blocks that will be complete after this step and have not been sent: [(start_block, ids)]."""
    if not st.triggered or st.handed_over:
        return None
    full = min((st.request.num_computed_tokens + num_scheduled) // block_size, len(st.block_ids))
    if full <= st.next_block:
        return None
    start, ids = st.next_block, st.block_ids[st.next_block:full]
    st.next_block = full
    return start, ids


def plan_handover(st, num_scheduled, block_size):
    """Source finishes its current step: n_kv tokens will have KV. Send the residual blocks
    (including the last, partial one) and the final token snapshot."""
    n_prompt = st.request.num_prompt_tokens
    n_kv = st.request.num_computed_tokens + num_scheduled
    last = min(-(-n_kv // block_size), len(st.block_ids))
    residual = (st.next_block, st.block_ids[st.next_block:last]) if last > st.next_block else None
    st.next_block, st.handed_over = last, True
    return residual, {"epoch": st.epoch, "n_kv": n_kv, "num_prompt_tokens": n_prompt,
                      "output_ids": list(st.request.all_token_ids[n_prompt:n_kv])}


def rx_index(rx_dir, epoch):
    """Blocks received for this epoch: {block_index: (file, position in file)}."""
    have = {}
    for meta_path in glob.glob(os.path.join(rx_dir, "kv_*.json")):
        m = read_json(meta_path)
        if not m or m.get("epoch") != epoch:
            continue
        for i in range(m["n_blocks"]):
            have[m["start_block"] + i] = (meta_path[:-5] + ".bin", i)
    return have


def plan_load(rx_dir, epoch, num_computed, prompt_len, block_size):
    """Target: tokens loadable beyond the local prefix, or None while the snapshot is not complete.
    Returns (n_external, from_block, to_block)."""
    final = read_json(os.path.join(rx_dir, "final.json"))
    if not final or final.get("epoch") != epoch:
        return None
    usable = min(final["n_kv"], prompt_len - 1)            # vLLM computes at least the last token
    to_block = usable // block_size
    from_block = num_computed // block_size
    if to_block <= from_block:
        return 0, from_block, from_block
    have = rx_index(rx_dir, epoch)
    if any(b not in have for b in range(from_block, to_block)):
        return None
    return to_block * block_size - num_computed, from_block, to_block
