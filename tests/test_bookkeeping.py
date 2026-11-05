import json
import os
from types import SimpleNamespace

from pallas.bookkeeping import SourceState, apply_trigger, plan_handover, plan_load, plan_stream, plan_trigger
from pallas.rx import store_final, store_kv

BS = 16


def fake_request(prompt_len, n_out):
    ids = list(range(1000, 1000 + prompt_len)) + list(range(5000, 5000 + n_out))
    return SimpleNamespace(num_prompt_tokens=prompt_len, all_token_ids=ids,
                           num_computed_tokens=len(ids) - 1, prompt_token_ids=ids[:prompt_len])


def decode(st, steps, sent):
    """Run `steps` decode steps; record every block planned for streaming."""
    for _ in range(steps):
        r = st.request
        r.all_token_ids.append(r.all_token_ids[-1] + 1)
        while len(st.block_ids) * BS < len(r.all_token_ids):
            st.block_ids.append(100 + len(st.block_ids))
        p = plan_stream(st, 1, BS)
        if p:
            sent.extend(range(p[0], p[0] + len(p[1])))
        r.num_computed_tokens += 1


def test_trigger_snaps_to_block_and_reports_prefix_outputs():
    st = SourceState("s", fake_request(100, 31))              # computed = 130
    ack = plan_trigger(st, {"epoch": 1, "link": "x"}, BS)
    assert ack["k"] == 128 and ack["j"] == 28 and len(ack["prefix_output_ids"]) == 28
    assert st.next_block == 8


def test_suffix_blocks_sent_once_contiguously_until_handover():
    st = SourceState("s", fake_request(100, 31))
    st.block_ids = [100 + i for i in range(-(-len(st.request.all_token_ids) // BS))]
    plan_trigger(st, {"epoch": 1, "link": "x"}, BS)
    sent = []
    decode(st, 70, sent)
    residual, snap = plan_handover(st, 1, BS)
    if residual:
        sent.extend(range(residual[0], residual[0] + len(residual[1])))
    n_kv = snap["n_kv"]
    assert sent == list(range(8, -(-n_kv // BS)))              # blocks [k/16, ceil(n/16)) exactly once
    assert len(snap["output_ids"]) == n_kv - 100


def test_fullcopy_streams_from_block_zero():
    st = SourceState("s", fake_request(100, 31))
    st.block_ids = [100 + i for i in range(9)]
    plan_trigger(st, {"epoch": 1, "link": "x", "from_block": 0}, BS)
    residual, _ = plan_handover(st, 1, BS)
    assert residual[0] == 0 and len(residual[1]) == 9


def test_retrigger_on_new_epoch_and_cancel():
    st = SourceState("s", fake_request(100, 31))
    st.block_ids = [100 + i for i in range(9)]
    assert apply_trigger(st, {"epoch": 1, "link": "wrong"}, BS)["k"] == 128
    assert apply_trigger(st, {"epoch": 1, "link": "wrong"}, BS) is None        # same file: applied once
    sent = []
    decode(st, 40, sent)
    assert sent and st.link == "wrong"
    assert apply_trigger(st, {"epoch": 1, "cancel": True}, BS)["cancelled"]
    sent2 = []
    decode(st, 20, sent2)
    assert sent2 == [] and not st.triggered                                      # cancelled: nothing streamed
    ack = apply_trigger(st, {"epoch": 2, "link": "right"}, BS)                   # new target, new prefix
    assert ack["epoch"] == 2 and st.link == "right" and st.next_block == ack["k"] // BS
    assert ack["k"] == (st.request.num_computed_tokens // BS) * BS


def test_target_load_plan(tmp_path):
    rx = str(tmp_path)
    meta = {"layers": ["l0"], "block_shape": [2], "dtype": "bfloat16", "epoch": 3}
    store_kv(rx, dict(meta, start_block=8, n_blocks=4), b"\0" * 16)
    # snapshot not in yet -> wait
    assert plan_load(rx, 3, 128, 200, BS) is None
    store_final(rx, {"epoch": 3, "n_kv": 190, "num_prompt_tokens": 100, "output_ids": [0] * 90})
    # 190 tokens with KV, prompt of 191 at activation: loadable up to block 11 (176 tokens)
    store_kv(rx, dict(meta, start_block=12, n_blocks=0), b"")
    n_ext, fb, tb = plan_load(rx, 3, 128, 191, BS)
    assert (n_ext, fb, tb) == (176 - 128, 8, 11)
    # a missing block -> keep waiting (never inject a gap)
    os.remove(os.path.join(rx, "kv_000008_4.json"))
    assert plan_load(rx, 3, 128, 191, BS) is None
    # a stale epoch's snapshot is ignored
    assert plan_load(rx, 4, 128, 191, BS) is None
