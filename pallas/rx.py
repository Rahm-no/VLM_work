"""Target side of the link: store received KV blocks and the final snapshot where the connector reads them."""
import os

from .bookkeeping import atomic_write_json


def store_kv(rx_dir, header, payload):
    os.makedirs(rx_dir, exist_ok=True)
    base = os.path.join(rx_dir, f"kv_{header['start_block']:06d}_{header['n_blocks']}")
    with open(base + ".bin.tmp", "wb") as f:
        f.write(payload)
    os.replace(base + ".bin.tmp", base + ".bin")
    atomic_write_json(base + ".json", {k: header[k] for k in
                                       ("epoch", "start_block", "n_blocks", "layers", "block_shape", "dtype")})


def store_final(rx_dir, header):
    os.makedirs(rx_dir, exist_ok=True)
    atomic_write_json(os.path.join(rx_dir, "final.json"),
                      {k: header[k] for k in ("epoch", "n_kv", "num_prompt_tokens", "output_ids")})
