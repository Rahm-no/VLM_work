"""Pallas cost model (Eqs. 1 to 9) and window selection (Algorithm 1), as written in the paper.

Symbols (paper notation):
  s_KV     KV bytes per token                          Eq. 1: 2 * L * H_kv * d_h * b
  K_curr   tokens in the session now (prompt + generated so far)
  r_p      target prefill throughput (tokens/s), EWMA-smoothed
  r_d      source decode throughput for this request (tokens/s), EWMA-smoothed
  B        inter-gNB goodput (bits/s), EWMA-smoothed
  T_w      prefetching window = predicted handover time - trigger time
"""
from dataclasses import dataclass


@dataclass
class PallasParams:
    alpha: float = 0.8          # Eq. 8 weight on SIT (paper value)
    t_max: float = 5.0          # largest window considered (s)   -- not given in the paper
    delta_t: float = 0.05       # window grid step (s)            -- not given in the paper
    t0: float = 0.0             # window-independent activation latency (s) -- not given; does not change argmin
    ewma: float = 0.3           # smoothing factor                -- not given in the paper
    block: int = 16             # vLLM PagedAttention block size (paper)


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    """Eq. 1."""
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def t_hist(l_hist, r_p, t_load=0.0):
    """Eq. 4: time for the target to prefill the history (prefix)."""
    return t_load + l_hist / r_p


def s_inc(t_w, r_d, s_kv):
    """Eq. 3: KV bytes of the suffix generated at the source during the window."""
    return r_d * t_w * s_kv


def t_res(t_w, r_d, s_kv, b_bps):
    """Eq. 5: residual suffix transfer left at the handover."""
    return max(0.0, s_inc(t_w, r_d, s_kv) * 8 / b_bps - t_w)


def sit(t_w, l_hist, r_p, r_d, s_kv, b_bps, t0=0.0):
    """Eq. 6: service interruption time."""
    return t0 + max(0.0, t_hist(l_hist, r_p) - t_w, t_res(t_w, r_d, s_kv, b_bps))


def t_early(t_w, l_hist, r_p):
    """Eq. 7: how long migrated state waits at the target before it is needed."""
    return max(0.0, t_w - t_hist(l_hist, r_p))


def objective(t_w, l_hist, r_p, r_d, s_kv, b_bps, p):
    """Eq. 8."""
    return p.alpha * sit(t_w, l_hist, r_p, r_d, s_kv, b_bps, p.t0) + (1 - p.alpha) * t_early(t_w, l_hist, r_p)


def choose_window(t_remain, k_curr, r_p, r_d, s_kv, b_bps, p):
    """Algorithm 1, lines 3 to 6 (with Eq. 9 for the history length). Returns (T_w*, J*, L_hist*)."""
    t_limit = max(0.0, min(p.t_max, t_remain))
    n = int(t_limit / p.delta_t + 1e-9)
    candidates = [i * p.delta_t for i in range(n + 1)]
    if t_limit - candidates[-1] > 1e-9:
        candidates.append(t_limit)                                  # W = {0, dT, ..., T_limit}
    best = None
    for t_w in candidates:
        l_hist = k_curr + r_d * (t_remain - t_w)                    # Eq. 9
        j = objective(t_w, l_hist, r_p, r_d, s_kv, b_bps, p)
        if best is None or j < best[1]:
            best = (t_w, j, l_hist)
    return best


def should_trigger(t_remain, k_curr, r_p, r_d, s_kv, b_bps, p):
    """Algorithm 1, line 7: start preparation once the remaining time reaches the chosen window."""
    t_w, _, _ = choose_window(t_remain, k_curr, r_p, r_d, s_kv, b_bps, p)
    return t_remain <= t_w + 1e-9, t_w


class Ewma:
    def __init__(self, alpha, value=None):
        self.alpha, self.value = alpha, value

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value
