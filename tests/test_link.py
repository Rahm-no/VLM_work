import asyncio
import socket
import threading
import time

from pallas import link


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _transfer_time(mbps, nbytes):
    sink_port, link_port, ctl_port = _free_port(), _free_port(), _free_port()
    received = []

    def sink():
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", sink_port))
        srv.listen()
        c, _ = srv.accept()
        n = 0
        while d := c.recv(1 << 20):
            n += len(d)
        received.append((n, time.perf_counter()))

    threading.Thread(target=sink, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(link.serve(f"127.0.0.1:{link_port}", f"127.0.0.1:{sink_port}",
                                                           lambda t: mbps * 1e6, 0.0, ctl_port)), daemon=True).start()
    time.sleep(0.3)
    c = socket.create_connection(("127.0.0.1", link_port))
    t0 = time.perf_counter()
    c.sendall(b"x" * nbytes)
    c.close()
    for _ in range(100):
        if received:
            break
        time.sleep(0.05)
    n, t1 = received[0]
    assert n == nbytes
    return t1 - t0


def test_link_carries_bytes_at_the_set_rate():
    dt = _transfer_time(40, 2_000_000)                    # 16 Mbit at 40 Mbps -> 0.4 s
    assert 0.33 <= dt <= 0.55, dt


def test_link_reaches_paper_goodputs():
    # regression: the first relay topped out near 400 Mbps (one wake-up per 16 KB)
    for mbps in (500, 1000):
        dt = _transfer_time(mbps, 25_000_000)            # 200 Mbit
        rate = 25_000_000 * 8 / dt / 1e6
        assert 0.85 * mbps <= rate <= 1.1 * mbps, (mbps, rate)
