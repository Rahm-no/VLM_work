from pallas.model import PallasParams, choose_window, kv_bytes_per_token, should_trigger, sit, t_early, t_res

P = PallasParams()
S_KV = kv_bytes_per_token(28, 4, 128)          # Qwen2-VL-7B: 56 KiB
B = 300e6


def test_eq1_qwen2vl():
    assert S_KV == 57344


def test_eq1_paper_example_qwen3_32b():
    assert kv_bytes_per_token(64, 8, 128) == 262144          # paper: ~256 KiB per token


def test_sit_zero_when_window_covers_prefill_and_suffix_small():
    assert sit(t_w=2.0, l_hist=3000, r_p=6000, r_d=30, s_kv=S_KV, b_bps=B) == 0.0


def test_residual_grows_with_window_on_slow_link():
    assert t_res(4.0, 30, S_KV, 10e6) > t_res(1.0, 30, S_KV, 10e6) > 0


def test_longer_history_needs_longer_window():
    short, _, _ = choose_window(5.0, 1000, 6000, 30, S_KV, B, P)
    long, _, _ = choose_window(5.0, 9000, 6000, 30, S_KV, B, P)
    assert long > short


def test_window_close_to_prefill_time():
    t_w, _, l_hist = choose_window(5.0, 6000, 6000, 30, S_KV, B, P)
    assert abs(t_w - l_hist / 6000) <= P.delta_t + 1e-9


def test_trigger_only_when_remaining_time_reaches_window():
    fire_late, _ = should_trigger(4.0, 3000, 6000, 30, S_KV, B, P)
    fire_now, _ = should_trigger(0.4, 3000, 6000, 30, S_KV, B, P)
    assert not fire_late and fire_now


def test_triggers_when_remaining_time_is_shorter_than_needed_window():
    # regression: T_limit must be a candidate, else Pallas never fires once T_remain < T_hist
    for t_remain in (0.197, 0.123, 0.051, 0.01):
        fire, t_w = should_trigger(t_remain, 3000, 6000, 30, S_KV, B, P)
        assert fire and abs(t_w - t_remain) < 1e-9


def test_t_early_penalises_too_long_window():
    assert t_early(3.0, 3000, 6000) > 0 and t_early(0.4, 3000, 6000) == 0
