"""Messages on the inter-gNB link, and the byte format of KV blocks.

A frame is: 4-byte big-endian header length | JSON header | payload (header["payload_len"] bytes).

Message types (all go through the throttled link):
  prefix   source -> target at the trigger: unexpanded prompt ids, the j output ids generated so
           far, and the raw images (a VLM needs them to rebuild image tokens). Target prefills [0, k).
  kv       source -> target: KV of whole blocks [start_block, start_block + n_blocks), every layer.
  final    source -> target at the handover: n_kv (tokens whose KV the target will hold) and the
           output ids up to n_kv (Pallas' final token snapshot).
  full     source -> target at the handover, Recomputation baseline: the whole sequence + images.
"""
import json
import socket
import struct

HDR = struct.Struct(">I")


def encode_frame(header, payload=b""):
    header = dict(header, payload_len=len(payload))
    h = json.dumps(header).encode()
    return HDR.pack(len(h)) + h + payload


def send_frame(sock, header, payload=b""):
    sock.sendall(encode_frame(header, payload))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("link closed")
        buf += chunk
    return bytes(buf)


def recv_frame(sock):
    (hlen,) = HDR.unpack(_recv_exact(sock, HDR.size))
    header = json.loads(_recv_exact(sock, hlen))
    return header, _recv_exact(sock, header["payload_len"])


async def read_frame(reader):
    (hlen,) = HDR.unpack(await reader.readexactly(HDR.size))
    header = json.loads(await reader.readexactly(hlen))
    return header, await reader.readexactly(header["payload_len"])


def connect(addr, timeout=10.0):
    host, port = addr.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(None)
    return s


# ---------------------------------------------------------------- KV block bytes
def block_dim(kv, num_blocks):
    """Index of the 'block' dimension in a layer's paged KV tensor (layout differs by backend)."""
    dims = [d for d, s in enumerate(kv.shape) if s == num_blocks]
    if not dims:
        raise ValueError(f"no dimension of size num_blocks={num_blocks} in KV shape {tuple(kv.shape)}")
    return dims[0]


def pack_blocks(kv_caches, layer_names, bdim, block_ids):
    """Copy the given blocks of every layer to host memory; returns (payload, meta)."""
    import torch
    idx = torch.tensor(block_ids, dtype=torch.long, device=kv_caches[layer_names[0]].device)
    parts, shape, dtype = [], None, None
    for name in layer_names:
        t = kv_caches[name].index_select(bdim, idx).movedim(bdim, 0).contiguous().cpu()
        shape, dtype = list(t.shape[1:]), str(t.dtype).replace("torch.", "")
        parts.append(t.view(torch.uint8).numpy().tobytes())
    return b"".join(parts), {"layers": layer_names, "block_shape": shape, "dtype": dtype,
                             "n_blocks": len(block_ids)}


def unpack_blocks(payload, meta):
    """Inverse of pack_blocks: {layer: tensor [n_blocks, *block_shape]} on CPU."""
    import torch
    dtype = getattr(torch, meta["dtype"])
    per = meta["n_blocks"]
    for s in meta["block_shape"]:
        per *= s
    nbytes = per * torch.tensor([], dtype=dtype).element_size()
    out = {}
    for i, name in enumerate(meta["layers"]):
        raw = torch.frombuffer(bytearray(payload[i * nbytes:(i + 1) * nbytes]), dtype=torch.uint8)
        out[name] = raw.view(dtype).view(meta["n_blocks"], *meta["block_shape"])
    return out


def write_blocks(kv_caches, bdim, blocks_by_layer, dst_block_ids):
    """Inject received blocks into the paged KV cache at dst_block_ids."""
    import torch
    first = next(iter(kv_caches.values()))
    idx = torch.tensor(dst_block_ids, dtype=torch.long, device=first.device)
    for name, t in blocks_by_layer.items():
        kv = kv_caches[name]
        kv.index_copy_(bdim, idx, t.to(device=kv.device, dtype=kv.dtype).movedim(0, bdim))
