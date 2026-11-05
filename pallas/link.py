#!/usr/bin/env python3
"""The inter-gNB link: a TCP relay from the source cell to the target cell's data port that
carries the real bytes at a set goodput. All connections share one token bucket (one link).

  python3 -m pallas.link --listen 0.0.0.0:9100 --to 127.0.0.1:9201 --mbps 300
  python3 -m pallas.link --listen 0.0.0.0:9100 --to 127.0.0.1:9201 --trace snob5g.csv   # time-varying

  python3 -m pallas.link --listen 0.0.0.0:9300 --to sink --mbps 300 --stats-port 9399
      a link to a cell that is not part of the experiment (the wrong target when the prediction is
      wrong): bytes cross at the link rate and are discarded.

Trace format: CSV "t_seconds,mbps" (rate holds until the next row; loops).
Control port (HTTP):  GET /            -> {"bytes", "elapsed_s", "rate_now_mbps"}
                      GET /set?mbps=X  -> change the goodput (used to sweep bandwidths)
"""
import argparse
import asyncio
import csv
import json
import time


class TokenBucket:
    """Shared rate limiter. rate_fn(t) returns bits/s at elapsed time t. Tokens may go negative
    (debt): a sender takes its chunk at once and then sleeps until the debt is repaid, so the long-run
    rate is exact and the cost is one wake-up per chunk."""

    def __init__(self, rate_fn, burst_bytes=16 * 1024):
        self.rate_fn, self.burst = rate_fn, burst_bytes
        self.t0 = self.last = time.perf_counter()
        self.tokens = 0.0
        self.lock = asyncio.Lock()
        self.bytes = 0

    async def take(self, n):
        """Wait until n bytes may cross the link."""
        async with self.lock:
            now = time.perf_counter()
            rate = self.rate_fn(now - self.t0) / 8.0          # bytes/s
            self.tokens = min(self.burst, self.tokens + (now - self.last) * rate) - n
            self.last = now
            if self.tokens < 0:
                await asyncio.sleep(-self.tokens / rate)


def rate_from_args(mbps, trace):
    if not trace:
        return lambda t: mbps * 1e6
    rows = [(float(r[0]), float(r[1]) * 1e6) for r in csv.reader(open(trace)) if r and not r[0].startswith("#")]
    period = rows[-1][0] + (rows[-1][0] - rows[-2][0] if len(rows) > 1 else 1.0)

    def f(t):
        t %= period
        cur = rows[0][1]
        for ts, r in rows:
            if ts > t:
                break
            cur = r
        return cur
    return f


async def pipe(reader, writer, bucket, delay_s, chunk=256 * 1024):
    try:
        while data := await reader.read(chunk):
            await bucket.take(len(data))
            if delay_s:
                await asyncio.sleep(delay_s)
            writer.write(data)
            await writer.drain()
            bucket.bytes += len(data)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def serve(listen, to, rate_fn, delay_s, stats_port):
    bucket = TokenBucket(rate_fn)
    th, tp = to.rsplit(":", 1) if to != "sink" else (None, None)

    async def handle(r, w):
        if to == "sink":
            try:
                while data := await r.read(256 * 1024):
                    await bucket.take(len(data))
                    bucket.bytes += len(data)
            except ConnectionError:
                pass
            w.close()
            return
        tr, tw = await asyncio.open_connection(th, int(tp))
        await asyncio.gather(pipe(r, tw, bucket, delay_s), pipe(tr, w, bucket, 0.0))

    async def stats(r, w):
        req = (await r.read(1024)).decode(errors="ignore").split(" ")
        path = req[1] if len(req) > 1 else "/"
        if path.startswith("/set?mbps="):
            mbps = float(path.split("=", 1)[1])
            bucket.rate_fn = lambda t: mbps * 1e6
        body = json.dumps({"bytes": bucket.bytes, "elapsed_s": time.perf_counter() - bucket.t0,
                           "rate_now_mbps": rate_fn(time.perf_counter() - bucket.t0) / 1e6}).encode()
        w.write(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\n\r\n" + body)
        await w.drain()
        w.close()

    lh, lp = listen.rsplit(":", 1)
    srv = await asyncio.start_server(handle, lh, int(lp))
    st = await asyncio.start_server(stats, lh, stats_port)
    async with srv, st:
        await asyncio.gather(srv.serve_forever(), st.serve_forever())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True)
    ap.add_argument("--to", required=True, help="target cell data address host:port, or 'sink'")
    ap.add_argument("--mbps", type=float, default=300.0)
    ap.add_argument("--trace", help="CSV t_seconds,mbps for a time-varying link")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="one-way propagation delay")
    ap.add_argument("--stats-port", type=int, default=9199)
    a = ap.parse_args()
    asyncio.run(serve(a.listen, a.to, rate_from_args(a.mbps, a.trace), a.delay_ms / 1000, a.stats_port))


if __name__ == "__main__":
    main()
