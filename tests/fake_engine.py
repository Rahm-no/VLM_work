"""A fake vLLM engine for whole-system tests without a GPU (PALLAS_FAKE_ENGINE=1).

It drives the REAL PallasConnector (scheduler and worker sides) exactly as vLLM's scheduler would:
prefix-cache hits, external tokens (with retry on None), block allocation, metadata per step,
start_load_kv before and wait_for_save after each forward. The "model" is deterministic:
  * images expand to IMG_TOKENS placeholder tokens and cost IMG_ENCODE_S to encode
  * prefill costs tokens / PREFILL_TOK_S, decode emits DECODE_TOK_S tokens/s
  * the KV of a full block is a function of every token up to the end of that block, and the next
    token is a function of the whole sequence
So a target that loaded wrong or misplaced KV is detected (KV check) and produces a different token.
"""
import asyncio
import hashlib
import os
import random
from types import SimpleNamespace

import torch

from tests import vllm_stub

vllm_stub.install()
from pallas.connector import PallasConnector  # noqa: E402

BS, NB, LAYERS = 16, 4096, 2
IMG_ID, IMG_TOKENS, IMG_ENCODE_S = 7, 600, 0.25
PREFILL_TOK_S, DECODE_TOK_S = 6000.0, 30.0


def _h(tokens):
    return int.from_bytes(hashlib.blake2b(bytes(str(tokens), "utf8"), digest_size=8).digest(), "big")


def block_kv(tokens_upto_block_end):
    g = torch.Generator().manual_seed(_h(tokens_upto_block_end) % (2 ** 63))
    return torch.randn((2, BS, 2, 4), generator=g).to(torch.bfloat16)


def next_token(seq):
    return 100 + _h(seq) % 1000


class FakeTokenizer:
    vocab_size = 32000

    def encode(self, text, add_special_tokens=False):
        return [IMG_ID if w == "<img>" else 10 + _h(w) % 30000 for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        out = []
        for m in msgs:
            out.append(f"<{m['role']}>")
            for c in m["content"]:
                out.append("<img>" if c["type"] == "image" else c["text"])
        return " ".join(out + (["<assistant>"] if add_generation_prompt else []))


class FakeEngine:
    def __init__(self, ctl_dir, rx_dir, wait_timeout_s):
        cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=BS), kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            kv_connector_extra_config={"ctl_dir": ctl_dir, "rx_dir": rx_dir, "wait_timeout_s": wait_timeout_s}))
        kvc = SimpleNamespace(num_blocks=NB)
        self.sched, self.work = PallasConnector(cfg, "scheduler", kvc), PallasConnector(cfg, "worker", kvc)
        self.kv = {f"layers.{i}.attn": torch.zeros((2, NB, BS, 2, 4), dtype=torch.bfloat16) for i in range(LAYERS)}
        self.work.register_kv_caches(self.kv)
        self.free = list(range(NB))
        random.Random(os.getpid()).shuffle(self.free)
        self.prefix_cache = {}                 # hash(tokens[:end of block]) -> block id
        self.in_use = {}                       # request id -> its blocks
        self.aborted = set()
        self.lock = asyncio.Lock()             # one engine step at a time
        self.kv_mismatch = 0

    def _release(self):
        """Blocks neither held by a request nor cached go back to the free list (as in vLLM)."""
        keep = set(self.prefix_cache.values()) | {b for v in self.in_use.values() for b in v}
        free = set(self.free)
        self.free.extend(b for b in range(NB) if b not in keep and b not in free)

    async def reset_prefix_cache(self):
        self.prefix_cache.clear()
        self._release()

    async def abort(self, rid):
        self.aborted.add(rid)

    def _step(self, so, compute):
        meta = self.sched.build_connector_meta(so)
        self.work.bind_connector_metadata(meta)
        self.work.start_load_kv(None)
        compute()
        self.work.wait_for_save()
        self.work.clear_connector_metadata()

    def _write(self, seq, blocks, upto):
        """'Forward': KV of every block touched by tokens [.., upto)."""
        for b in range((upto - 1) // BS + 1):
            end = min((b + 1) * BS, upto)
            for t in self.kv.values():
                t[:, blocks[b]] = block_kv(tuple(seq[:end]))

    async def generate(self, prompt, sp, rid):
        try:
            async for out in self._generate(prompt, sp, rid):
                yield out
        finally:                                   # finished, aborted or cancelled: free its blocks
            self.in_use.pop(rid, None)
            self._release()

    async def _generate(self, prompt, sp, rid):
        ids = []
        for t in prompt["prompt_token_ids"]:
            ids.extend([IMG_ID] * IMG_TOKENS if t == IMG_ID else [t])
        params = (sp.extra_args or {}).get("kv_transfer_params")
        req = SimpleNamespace(request_id=rid, kv_transfer_params=params, prompt_token_ids=list(ids),
                              num_prompt_tokens=len(ids), all_token_ids=list(ids), num_computed_tokens=0)
        # ---- schedule: local prefix hit, then external tokens (retry while the connector says None)
        hit_blocks = []
        for b in range((len(ids) - 1) // BS):
            bid = self.prefix_cache.get(_h(tuple(ids[:(b + 1) * BS])))
            if bid is None:
                break
            hit_blocks.append(bid)
        local = len(hit_blocks) * BS
        while True:
            n_ext, _ = self.sched.get_num_new_matched_tokens(req, local)
            if n_ext is not None:
                break
            await asyncio.sleep(0.005)
        blocks = hit_blocks + [self.free.pop() for _ in range(-(-len(ids) // BS) - len(hit_blocks))]
        self.in_use[rid] = blocks
        self.sched.update_state_after_alloc(req, None, n_ext)
        start = local + n_ext
        n_img = sum(1 for i in range(start, len(ids)) if ids[i] == IMG_ID) / IMG_TOKENS
        async with self.lock:
            await asyncio.sleep((len(ids) - start) / PREFILL_TOK_S + n_img * IMG_ENCODE_S)
            so = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id=rid, block_ids=(list(blocks),))],
                                 scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set()),
                                 num_scheduled_tokens={rid: len(ids) - start})

            def prefill():
                for b in range(start // BS):                 # loaded or cached blocks must be right
                    want = block_kv(tuple(ids[:(b + 1) * BS]))
                    if not torch.equal(self.kv["layers.0.attn"][:, blocks[b]], want):
                        self.kv_mismatch += 1
                self._write(ids, blocks, len(ids))
            self._step(so, prefill)
        req.num_computed_tokens = len(ids)
        for b in range(len(ids) // BS):
            self.prefix_cache[_h(tuple(ids[:(b + 1) * BS]))] = blocks[b]
        n_out, max_tokens = 0, sp.max_tokens
        while True:
            tok = next_token(tuple(req.all_token_ids))
            req.all_token_ids.append(tok)
            n_out += 1
            yield SimpleNamespace(prompt_token_ids=ids, finished=n_out >= max_tokens or rid in self.aborted,
                                  outputs=[SimpleNamespace(token_ids=[tok])])
            if n_out >= max_tokens or rid in self.aborted:
                break
            new = []
            if len(req.all_token_ids) > len(blocks) * BS:
                new = [self.free.pop()]
                blocks.extend(new)
            async with self.lock:
                await asyncio.sleep(1.0 / DECODE_TOK_S)
                so = SimpleNamespace(scheduled_new_reqs=[], num_scheduled_tokens={rid: 1},
                                     scheduled_cached_reqs=SimpleNamespace(req_ids=[rid], new_block_ids=[(new,) if new else None],
                                                                           resumed_req_ids=set()))
                self._step(so, lambda: self._write(req.all_token_ids, blocks, len(req.all_token_ids)))
            req.num_computed_tokens += 1
            if rid in self.aborted:
                break
        self.sched.request_finished(req, blocks)
        self.in_use.pop(rid, None)
        self._release()
