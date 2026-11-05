#!/usr/bin/env python3
"""
Per-phase timing of VLM inference in vLLM: preprocess, vision encode, prefill, decode.

  preprocess : HF processor on CPU (vLLM uses the same processor)
  encode     : the model's image-embedding method inside vLLM, timed with CUDA events
  prefill    : time-to-first-token (max_tokens=1)  -  encode  -  preprocess
               (includes a few ms of vLLM scheduling overhead)
  decode     : (time for N tokens - time for 1 token) / (N - 1)

Caches are disabled so every run recomputes everything.

Usage (one GPU):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python vllm_phase_bench.py \
      --model Qwen/Qwen2-VL-7B-Instruct --images ./imgs --out qwen_vllm.csv
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python vllm_phase_bench.py \
      --model google/gemma-3-4b-it --images ./imgs --out gemma_vllm.csv

Also measures text-only prefill speed (tokens/s) at a few lengths (--text-lengths).

Notes
- Internal names change between vLLM versions. The script hooks whichever of
  `get_multimodal_embeddings` / `embed_multimodal` the model class has. If neither is
  found it still reports TTFT and decode, and prints a warning (encode = n/a).
- Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so the hook runs in this process.
"""
import argparse
import csv
import glob
import os
import statistics as st
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from PIL import Image

ENCODE_MS = []


def hook_encoder(model_id):
    """Wrap the vLLM model class's image-embedding method with CUDA-event timing."""
    import importlib
    candidates = {
        "qwen2-vl": ("vllm.model_executor.models.qwen2_vl", "Qwen2VLForConditionalGeneration"),
        "qwen2.5-vl": ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        "qwen3-vl": ("vllm.model_executor.models.qwen3_vl", "Qwen3VLForConditionalGeneration"),
        "gemma-3": ("vllm.model_executor.models.gemma3_mm", "Gemma3ForConditionalGeneration"),
        "llava": ("vllm.model_executor.models.llava", "LlavaForConditionalGeneration"),
    }
    key = next((k for k in candidates if k in model_id.lower()), None)
    if key is None:
        print("WARNING: unknown model family, encode time will be n/a")
        return False
    mod_name, cls_name = candidates[key]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    for meth in ("get_multimodal_embeddings", "embed_multimodal"):
        if hasattr(cls, meth):
            orig = getattr(cls, meth)

            def timed(self, *a, __orig=orig, **kw):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = __orig(self, *a, **kw)
                end.record()
                torch.cuda.synchronize()
                ENCODE_MS.append(start.elapsed_time(end))
                return out

            setattr(cls, meth, timed)
            print(f"hooked {cls_name}.{meth}")
            return True
    print("WARNING: no image-embedding method found to hook; encode = n/a")
    return False


def build_llm(model_id, max_len, max_pixels, max_images=1):
    from vllm import LLM
    kw = dict(model=model_id, max_model_len=max_len, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": max_images}, gpu_memory_utilization=0.85, dtype="bfloat16")
    if "qwen" in model_id.lower() and max_pixels:
        kw["mm_processor_kwargs"] = {"max_pixels": max_pixels}
    for cache_kw in ({"mm_processor_cache_gb": 0}, {"disable_mm_preprocessor_cache": True}, {}):
        try:
            return LLM(**kw, **cache_kw)
        except TypeError:
            continue
    raise RuntimeError("could not create LLM")


def prompt_for(model_id, processor, question, n_images=1):
    msgs = [{"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": question}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def run(llm, prompt, image, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    req = {"prompt": prompt}
    if image is not None:
        req["multi_modal_data"] = {"image": image}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([req], sp, use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, len(out[0].prompt_token_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-pixels", type=int, default=4096 * 28 * 28, help="Qwen cap (default = 4096 tokens)")
    ap.add_argument("--counts", default="", help="also time prompts with this many images each, e.g. 1,3,6,10")
    ap.add_argument("--sets", type=int, default=3, help="different image sets per count (averaged)")
    ap.add_argument("--text-lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--out", default="vllm_phases.csv")
    a = ap.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    hooked = hook_encoder(a.model)
    counts = [int(x) for x in a.counts.replace(" ", ",").split(",") if x]
    llm = build_llm(a.model, a.max_len, a.max_pixels, max(counts + [1]))
    prompt = prompt_for(a.model, processor, a.question)

    files = sorted(f for f in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if counts:
        files_all, files = files, files[:4]                    # per-image rows: first 4 only
    rows = []
    warm = Image.open(files[0]).convert("RGB")
    run(llm, prompt, warm, 1)                                  # warm-up
    for f in files:
        img = Image.open(f).convert("RGB")
        pre, enc, ttft, full = [], [], [], []
        ntok = None
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            processor(text=[prompt], images=[img], return_tensors="pt")
            pre.append((time.perf_counter() - t0) * 1000)
            ENCODE_MS.clear()
            t1, ntok = run(llm, prompt, img, 1)
            ttft.append(t1)
            enc.append(sum(ENCODE_MS) if ENCODE_MS else float("nan"))
            tN, _ = run(llm, prompt, img, a.decode_tokens)
            full.append(tN)
        med = st.median
        e = med(enc) if hooked else float("nan")
        row = {"image": os.path.basename(f), "prompt_tokens": ntok,
               "preprocess_ms": round(med(pre), 2), "encode_ms": round(e, 2),
               "ttft_ms": round(med(ttft), 2),
               # generate() also runs the HF processor (cache off), so subtract it too
               "prefill_ms": round(med(ttft) - e - med(pre), 2) if hooked else float("nan"),
               "decode_ms_per_token": round((med(full) - med(ttft)) / (a.decode_tokens - 1), 2)}
        rows.append(row)
        print(row)

    # several images in one prompt: distinct images, averaged over --sets different sets
    for n in counts:
        if n > len(files_all):
            print(f"skip {n} images: only {len(files_all)} files"); continue
        p_n = prompt_for(a.model, processor, a.question, n)
        per = {"preprocess_ms": [], "encode_ms": [], "ttft_ms": []}
        ntok = None
        for s_i in range(a.sets):
            imgs = [Image.open(files_all[(s_i * n + j) % len(files_all)]).convert("RGB") for j in range(n)]
            run(llm, p_n, imgs, 1)                             # warm-up for this shape
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                processor(text=[p_n], images=imgs, return_tensors="pt")
                per["preprocess_ms"].append((time.perf_counter() - t0) * 1000)
                ENCODE_MS.clear()
                t1, ntok = run(llm, p_n, imgs, 1)
                per["ttft_ms"].append(t1)
                per["encode_ms"].append(sum(ENCODE_MS) if ENCODE_MS else float("nan"))
        m = {k: st.mean(v) for k, v in per.items()}
        row = {"image": f"{n}_IMAGES", "prompt_tokens": ntok, "n_images": n,
               "preprocess_ms": round(m["preprocess_ms"], 1), "encode_ms": round(m["encode_ms"], 1),
               "ttft_ms": round(m["ttft_ms"], 1),
               "prefill_ms": round(m["ttft_ms"] - m["encode_ms"] - m["preprocess_ms"], 1),
               "encode_share_of_gpu_%": round(100 * m["encode_ms"] / (m["ttft_ms"] - m["preprocess_ms"]), 1),
               "runs": len(per["ttft_ms"])}
        rows.append(row)
        print(row)

    # text-only prefill speed
    tok = processor.tokenizer
    for L in [int(x) for x in a.text_lengths.split(",") if int(x) < a.max_len - 64]:
        text = tok.decode(tok("hello " * (L + 50))["input_ids"][:L])
        ts = [run(llm, text, None, 1)[0] for _ in range(a.repeats)]
        rows.append({"image": f"TEXT_ONLY_{L}", "prompt_tokens": L, "ttft_ms": round(st.median(ts), 2),
                     "prefill_ms": round(st.median(ts), 2),
                     "prefill_tok_per_s": round(L / (st.median(ts) / 1000))})
        print(rows[-1])

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
