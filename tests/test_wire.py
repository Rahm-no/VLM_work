import socket

import pytest
import torch

from pallas import wire

LAYOUTS = {"flash_attn": lambda nb: (2, nb, 16, 4, 8),      # [2, num_blocks, block, heads, dim]
           "flashinfer": lambda nb: (nb, 2, 16, 4, 8)}      # [num_blocks, 2, block, heads, dim]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_blocks_roundtrip_into_other_slots(layout):
    nb = 40
    src = {f"layer.{i}": torch.randn(LAYOUTS[layout](nb), dtype=torch.bfloat16) for i in range(3)}
    dst = {k: torch.zeros_like(v) for k, v in src.items()}
    names = sorted(src)
    bdim = wire.block_dim(src[names[0]], nb)
    payload, meta = wire.pack_blocks(src, names, bdim, [5, 6, 7])
    blocks = wire.unpack_blocks(payload, meta)
    wire.write_blocks(dst, bdim, blocks, [30, 2, 11])           # target allocates different blocks
    for n in names:
        for s, d in zip([5, 6, 7], [30, 2, 11]):
            assert torch.equal(src[n].select(bdim, s), dst[n].select(bdim, d))


def test_frame_roundtrip_over_socket():
    import threading
    a, b = socket.socketpair()
    payload = bytes(range(256)) * 1000
    th = threading.Thread(target=wire.send_frame, args=(a, {"type": "kv", "x": 1}, payload))
    th.start()
    h, p = wire.recv_frame(b)
    th.join()
    assert h["type"] == "kv" and h["payload_len"] == len(payload) and p == payload
