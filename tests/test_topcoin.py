"""v16 top-coin simulator (research/topcoin.py): entries, the three exits, costs and funding on a tiny hand-made panel."""
import math

import numpy as np

from research import topcoin
from research.topcoin import MAKER_COST, TAKER_COST, FILL_GAP, Rules

T, N = 60, 3


def panel(alpha_rows=None) -> topcoin.Panel:
    close = np.full((T, N), 100.0)
    a = np.tile([0.01, 0.0, -0.01], (T, 1)) if alpha_rows is None else alpha_rows
    return topcoin.Panel(times=np.arange(T) * 14_400_000, symbols=["A", "B", "C"], open=close.copy(), high=close * 1.001,
                         low=close * 0.999, close=close, sigma=np.full((T, N), 0.01), funding=np.zeros((T, N)), alpha=a)


def test_market_entry_held_for_the_horizon_then_sold_with_funding_and_costs():
    p = panel()
    p.close[1:, 0] = 110.0
    p.open[1:, 0] = 110.0
    p.funding[:, 0] = 0.0001
    rules = Rules(limit_k=None, stop_k=None, score_exit=False, min_edge=None, hold_bars=5)
    res = topcoin.simulate(p, lambda t: 0 if t < 5 else 1, rules, 0, 10)
    first = res.trades[0]
    assert first["symbol"] == "A" and first["why"] == "time" and first["bars"] == 5 and first["exit_ms"] == p.times[5]
    assert math.isclose(first["net"], (1 - TAKER_COST) ** 2 * 1.10 * (1 - 0.0001) ** 5 - 1)


def test_still_first_at_expiry_keeps_the_coin_without_paying_again():
    p = panel()
    rules = Rules(limit_k=None, stop_k=None, score_exit=False, min_edge=None, hold_bars=5)
    res = topcoin.simulate(p, topcoin.top_picker(p), rules, 0, 20)
    assert [tr["why"] for tr in res.trades] == ["open"] and res.orders == 1
    assert math.isclose(res.equity[-1], 1000 * (1 - TAKER_COST))


def test_limit_fills_only_when_the_low_passes_it_and_is_cancelled_after_the_wait():
    p = panel()
    limit = 100 * (1 - 0.25 * 0.01)
    p.low[1:3, 0] = limit * (1 - FILL_GAP / 2)            # touches but does not pass: no fill in bars 1 and 2
    p.low[3, 0] = limit * (1 - FILL_GAP * 2)               # the order placed at close 2 fills in bar 3
    rules = Rules(stop_k=None, score_exit=False, min_edge=None, hold_bars=50)
    res = topcoin.simulate(p, topcoin.top_picker(p), rules, 0, 5)
    assert res.orders == 2 and res.fills == 1                  # the first order expired after two bars unfilled
    assert math.isclose(res.trades[0]["entry"], 100 * (1 - 0.25 * 0.01)) and res.trades[0]["entry_ms"] == p.times[3]
    assert math.isclose(res.equity[-1], 1000 * (1 - MAKER_COST) * 100 / limit)


def test_stop_sells_at_the_stop_or_the_open_below_it():
    p = panel()
    rules = Rules(limit_k=None, score_exit=False, min_edge=None, hold_bars=50, stop_k=1.0)
    stop = 100 * (1 - 0.01 * math.sqrt(50))
    p.low[4, 0] = stop - 0.5
    res = topcoin.simulate(p, lambda t: 0 if t == 0 else None, rules, 0, 6)
    assert res.trades[0]["why"] == "stop" and math.isclose(res.trades[0]["exit"], stop)
    p.open[4, 0] = stop - 0.3
    res = topcoin.simulate(p, lambda t: 0 if t == 0 else None, rules, 0, 6)
    assert math.isclose(res.trades[0]["exit"], stop - 0.3)


def test_score_exit_and_min_edge():
    a = np.tile([0.01, 0.0, -0.01], (T, 1))
    a[3] = [-0.02, 0.0, 0.01]                              # A drops to the bottom at close 3
    p = panel(a)
    rules = Rules(limit_k=None, stop_k=None, min_edge=None, hold_bars=50)
    res = topcoin.simulate(p, lambda t: 0 if t == 0 else None, rules, 0, 6)
    assert res.trades[0]["why"] == "score" and res.trades[0]["exit_ms"] == p.times[3]
    small = panel(np.tile([0.001, 0.0, -0.01], (T, 1)))   # below the round-trip cost: no order
    assert topcoin.simulate(small, topcoin.top_picker(small), topcoin.RULES_A, 0, 10).orders == 0


def test_bottom_half_and_holds():
    p = panel()
    assert topcoin.bottom_half(p, 0, 2) and not topcoin.bottom_half(p, 0, 0)
    p.close[5:, 1] = 120.0
    assert math.isclose(topcoin.hold(p, 0, 10, "B")[-1], 1.2)
    assert math.isclose(topcoin.hold(p, 0, 10)[-1], 1 + 0.2 / 3)


def test_shuffled_scores_keep_each_close_and_month_only_cut_the_link_to_coins():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(T, N))
    a[:, 2] = np.where(np.arange(T) < 30, a[:, 2], np.nan)          # C leaves the universe in the second month
    p = panel(a)
    months = np.where(np.arange(T) < 30, "m1", "m2")
    null = topcoin.permuted(p, months, seed=3)
    for t in range(T):
        assert np.allclose(np.sort(null.alpha[t][np.isfinite(null.alpha[t])]), np.sort(a[t][np.isfinite(a[t])]))
    for rows in (slice(0, 30), slice(30, T)):                          # one mapping per month
        block, orig = null.alpha[rows], a[rows]
        moved = [next(k for k in range(N) if np.allclose(block[:, k], orig[:, i], equal_nan=True)) for i in range(N)
                 if np.isfinite(orig[:, i]).any()]
        assert len(set(moved)) == len(moved)
    assert np.isnan(null.alpha[30:]).sum() == 30                       # still two coins in the second month
