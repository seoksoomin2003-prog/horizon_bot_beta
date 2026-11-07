import numpy as np
import pytest

from research import h4_data
from research.h4_rules import RULES, Rule, Signals, signals
from research.h4_sim import BAR_MS, Market, benchmark, block_interval, max_drawdown, run, weekly_returns

T0 = 1_600_000_000_000 // BAR_MS * BAR_MS


def bars_from(rows):
    return np.array([[T0 + i * BAR_MS, o, h, l, c, 1.0] for i, (o, h, l, c) in enumerate(rows)], dtype=float)


def market(rows, enter_long=(), enter_short=(), exit_long=(), exit_short=(), atr=1.0, funding=(), symbol="X"):
    n = len(rows)

    def mask(idx):
        a = np.zeros(n, dtype=bool)
        a[list(idx)] = True
        return a
    s = Signals(np.full(n, atr), mask(enter_long), mask(enter_short), mask(exit_long), mask(exit_short))
    return Market(symbol, bars_from(rows), np.array(funding, dtype=float).reshape(-1, 2), s)


FLAT = (100.0, 100.5, 99.5, 100.0)


def go(m, rule=None, side_cost=0.0, markets=None):
    rule = rule or Rule("t", 2.0)
    return run(rule, markets or [m], T0, T0 + 100 * BAR_MS, side_cost=side_cost)


@pytest.mark.parametrize("key", list(RULES))
def test_signals_do_not_use_future_bars(key):
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 700)))
    rows = [(c, c * 1.01, c * 0.99, c) for c in close]
    full = signals(key, bars_from(rows))
    cut = signals(key, bars_from(rows[:500]))
    for name in ("atr", "enter_long", "enter_short", "exit_long", "exit_short"):
        np.testing.assert_array_equal(getattr(full, name)[:500], getattr(cut, name))


def test_entry_fills_at_next_open_sized_by_risk():
    rows = [FLAT, (110.0, 111, 109, 110), FLAT]
    r = go(market(rows, enter_long=[0]))
    p = r["positions"][0]
    assert p["entry"] == 110.0 and p["entry_ms"] == T0 + BAR_MS
    assert p["qty"] == pytest.approx(0.0075 * 1000 / 2.0)


def test_stop_inside_bar_and_gap_fill():
    inside = [FLAT, (100.0, 100.5, 97.0, 99.0), FLAT]
    assert go(market(inside, enter_long=[0]))["positions"][0]["exit"] == 98.0
    gap = [FLAT, FLAT, (90.0, 91.0, 89.0, 90.0), FLAT]
    p = go(market(gap, enter_long=[0]))["positions"][0]
    assert p["exit"] == 90.0 and p["reason"] == "stop_gap"


def test_short_stop_is_above_entry():
    rows = [FLAT, (100.0, 103.0, 99.0, 100.0), FLAT]
    p = go(market(rows, enter_short=[0]))["positions"][0]
    assert p["side"] == -1 and p["exit"] == 102.0


def test_stop_wins_over_a_profitable_bar():
    rows = [FLAT, (100.0, 120.0, 97.0, 119.0), FLAT]
    assert go(market(rows, enter_long=[0]))["positions"][0]["reason"] == "stop"


def test_costs_charged_on_both_sides():
    rows = [FLAT, FLAT, FLAT]
    r = go(market(rows, enter_long=[0], exit_long=[1]), side_cost=0.001)
    p = r["positions"][0]
    assert p["pnl"] == pytest.approx(-2 * p["qty"] * 100 * 0.001)
    assert r["cash"] == pytest.approx(1000 + p["pnl"])


def test_funding_paid_by_long_received_by_short():
    rows = [FLAT, FLAT, FLAT, FLAT]
    funding = [[T0 + 2 * BAR_MS, 0.001]]
    long_p = go(market(rows, enter_long=[0], funding=funding))["positions"][0]
    short_p = go(market(rows, enter_short=[0], funding=funding))["positions"][0]
    assert long_p["pnl"] == pytest.approx(-long_p["qty"] * 100 * 0.001)
    assert short_p["pnl"] == pytest.approx(short_p["qty"] * 100 * 0.001)


def test_position_limit_fills_in_listed_order():
    rows = [FLAT, FLAT, FLAT]
    ms = [market(rows, enter_long=[0], symbol=s) for s in "ABCD"]
    r = run(Rule("t", 2.0), ms, T0, T0 + 3 * BAR_MS, side_cost=0.0)
    assert [p["symbol"] for p in r["positions"]] == ["A", "B", "C"] and r["skipped"] == 1


def test_notional_cap_skips_oversized_entry():
    rows = [FLAT, FLAT, FLAT]
    r = run(Rule("t", 2.0), [market(rows, enter_long=[0], atr=0.1)], T0, T0 + 3 * BAR_MS, side_cost=0.0)
    assert r["positions"] == [] and r["skipped"] == 1


def test_trailing_stop_only_tightens_and_acts_next_bar():
    rows = [FLAT, (100.0, 110.0, 99.0, 110.0), (110.0, 110.5, 105.0, 105.0), (105.0, 105.0, 107.0, 105.0)]
    rows[3] = (105.0, 105.5, 104.0, 105.0)
    p = go(market(rows, enter_long=[0]), rule=Rule("t", 2.0, trailing=True))["positions"][0]
    # Closed at 110 with ATR 1 and multiple 2, so the stop becomes 108 and fills in the next bar's range.
    assert p["exit"] == 108.0


def test_time_limit_exits_at_next_open():
    rows = [FLAT] * 6
    p = go(market(rows, enter_long=[0]), rule=Rule("t", 2.0, max_bars=3))["positions"][0]
    assert p["entry_ms"] == T0 + BAR_MS and p["exit_ms"] == T0 + 4 * BAR_MS and p["reason"] == "signal"


def test_reversal_on_the_same_signal():
    rows = [FLAT] * 4
    ps = go(market(rows, enter_long=[0], enter_short=[1], exit_long=[1]))["positions"]
    assert [p["side"] for p in ps] == [1, -1] and ps[0]["exit_ms"] == ps[1]["entry_ms"]


def test_nothing_traded_outside_the_window():
    rows = [FLAT] * 10
    r = run(Rule("t", 2.0), [market(rows, enter_long=[1])], T0 + 5 * BAR_MS, T0 + 10 * BAR_MS, side_cost=0.0)
    assert r["positions"] == []


def test_benchmark_equal_weight_return():
    rows_a = [(100.0, 110, 100, 110.0), (110.0, 110, 110, 110.0)]
    rows_b = [(100.0, 100, 90, 90.0), (90.0, 90, 90, 90.0)]
    curve = benchmark([market(rows_a, symbol="A"), market(rows_b, symbol="B")], T0, T0 + 2 * BAR_MS, side_cost=0.0)
    assert curve[0][1] == pytest.approx(1000.0) and curve[1][1] == pytest.approx(1000.0)


def test_statistics():
    curve = [(1, 1100.0), (2, 880.0), (3, 900.0)]
    assert max_drawdown(curve, 1000.0) == pytest.approx(-0.2)
    assert weekly_returns(curve, 1000.0) == pytest.approx([-0.1])
    low, high = block_interval([0.01] * 20)
    assert low == pytest.approx(0.01) and high == pytest.approx(0.01)


def test_validation_period_is_closed():
    with pytest.raises(PermissionError):
        h4_data.load("BTCUSDT", end_ms=h4_data.DEV_END_MS + 1)
    with pytest.raises(PermissionError):
        h4_data.download(end_ms=h4_data.DEV_END_MS + 1)
