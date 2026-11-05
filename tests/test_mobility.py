import numpy as np

from pallas import mobility as m


def test_straight_path_cvh_exact():
    # 10 m/s along x from x=5 in 100 m cells: handover at x=100 after 9.5 s, CVH exact and right
    xy = np.stack([5 + 10 * np.arange(0, 15, m.TICK), np.full(150, 50.0)], 1)
    P = xy.copy()
    cells, evs = m.handovers(P, 100.0)
    assert len(evs) == 1 and abs(evs[0][1] - 9.5) < 1e-6 and evs[0][3] == (1, 0)
    V = np.zeros_like(P)
    V[5:] = (P[5:] - P[:-5]) / 0.5
    tgt, tc, valid = m.cvh(P, V, cells, 100.0)
    assert all(tuple(tgt[j]) == (1, 0) for j in range(5, 95))
    assert abs(50 * m.TICK + tc[50] - 9.5) < 1e-6


def test_traces_relative_times_and_oracle():
    trs = m.prediction_traces(m.synthetic_paths(3, 1), 100.0, 2, 0)
    for tr in trs:
        ts = [t[0] for t in tr["ticks"]]
        assert ts == sorted(ts) and ts[-1] <= 1e-6 and ts[-1] > -m.TICK - 1e-6
    o = m.oracle_trace(2.0)
    assert m.prediction_at(o, -2.5) is None and m.prediction_at(o, -1.0) == ("T", True, 0.0)
