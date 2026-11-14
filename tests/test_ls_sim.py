"""The v12 long-short basket simulator: picks, money, costs and funding come out as the registration says."""

import numpy as np
import pandas as pd
import pytest

from research import ls_sim
from research.h4_data import BAR_MS

COINS = ["AAA", "BBB", "CCC", "DDD"]


def panel(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=[i * BAR_MS for i in range(len(rows))], columns=COINS, dtype=float)


def predictions(by_time: dict[int, list[float]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "close_ms": t * BAR_MS, "p": p} for t, ps in by_time.items() for s, p in zip(COINS, ps)])


def test_targets_take_the_extremes_with_equal_money_and_order_ties_by_name():
    p = pd.Series({"B": 0.5, "A": 0.5, "C": 0.9, "D": 0.1, "E": 0.5})
    assert ls_sim.targets(p, 2, min_coins=4) == {"D": -0.25, "A": -0.25, "E": 0.25, "C": 0.25}
    assert ls_sim.targets(p, 2, min_coins=6) == {}


def test_a_tenth_of_the_coins_is_rounded():
    weights = lambda n: ls_sim.targets(pd.Series({f"C{i:02d}": i / 100 for i in range(n)}), None)
    assert len(weights(19)) == 0 and len(weights(24)) == 4 and len(weights(25)) == 6 and len(weights(46)) == 10
    assert sum(w for w in weights(46).values() if w > 0) == pytest.approx(0.5)
    assert sum(w for w in weights(46).values() if w < 0) == pytest.approx(-0.5)


def test_flat_prices_lose_exactly_the_costs():
    closes, flat = panel([[10, 10, 10, 10]] * 4), panel([[0, 0, 0, 0]] * 4)
    pred = predictions({0: [0.9, 0.8, 0.2, 0.1], 1: [0.9, 0.8, 0.2, 0.1]})       # same picks twice, then nothing
    bars = ls_sim.run(pred, closes, flat, 1, 1, side_cost=0.001, min_coins=4)
    assert bars["traded"].iloc[0] == pytest.approx(1.0)             # opens 0.5 long and 0.5 short
    assert bars["equity"].iloc[0] == pytest.approx(999.0)
    assert bars["traded"].iloc[1] == pytest.approx(1 / 999)          # the same coins: only trims 500 to 499.5 on each side
    assert bars["traded"].iloc[2] == pytest.approx(999 / 998.999)   # no prediction: closes the 999 held, equity is 998.999
    assert bars["gross"].iloc[2] == 0.0


def test_gains_funding_and_drift_are_booked_on_the_right_side():
    closes = panel([[10, 10, 10, 10], [11, 10, 10, 8], [11, 10, 10, 8]])
    funding = panel([[0.001, 0, 0, 0.002], [0, 0, 0, 0], [0, 0, 0, 0]])
    pred = predictions({0: [0.9, 0.5, 0.5, 0.1]})
    bars = ls_sim.run(pred, closes, funding, 1, 1, side_cost=0.0, min_coins=4)
    first = bars.iloc[0]
    assert first["long"] == pytest.approx(0.05) and first["short"] == pytest.approx(0.10)     # long +10%, short -20%
    assert first["funding"] == pytest.approx(-0.5 * 0.001 + 0.5 * 0.002)                        # the long pays, the short receives
    assert first["ret"] == pytest.approx(0.05 + 0.10 + 0.0005)
    assert first["market"] == pytest.approx((0.1 - 0.2) / 4)
    assert bars["traded"].iloc[1] == pytest.approx((550 + 400) / bars["equity"].iloc[0])        # drifted money is what gets closed


def test_two_baskets_take_half_each_and_alternate():
    closes, flat = panel([[10, 10, 10, 10]] * 5), panel([[0, 0, 0, 0]] * 5)
    pred = predictions({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
    bars = ls_sim.run(pred, closes, flat, 2, 1, side_cost=0.0, min_coins=4)
    assert list(bars["traded"].round(6)) == [0.5, 0.5, 0.5, 0.5]       # each opens at its turn and closes two bars later
    assert list(bars["gross"].round(6)) == [0.5, 1.0, 0.5, 0.0]


def test_the_last_bar_pays_for_closing():
    closes, flat = panel([[10, 10, 10, 10]] * 2), panel([[0, 0, 0, 0]] * 2)
    bars = ls_sim.run(predictions({0: [0.9, 0.8, 0.2, 0.1]}), closes, flat, 1, 1, side_cost=0.001, min_coins=4)
    assert len(bars) == 1 and bars["traded"].iloc[0] == pytest.approx(2.0) and bars["equity"].iloc[0] == pytest.approx(998.0)


def test_later_prices_never_change_earlier_bars():
    rng = np.random.default_rng(3)
    prices = np.exp(np.cumsum(rng.normal(0, 0.02, (40, 4)), axis=0))
    pred = predictions({t: list(rng.random(4)) for t in range(38)})
    flat = panel([[0, 0, 0, 0]] * 40)
    changed = prices.copy()
    changed[21:] *= 1.5
    a = ls_sim.run(pred, panel(prices.tolist()), flat, 2, 1, min_coins=4)
    b = ls_sim.run(pred, panel(changed.tolist()), flat, 2, 1, min_coins=4)
    assert a.iloc[:20].equals(b.iloc[:20]) and not a.iloc[20:].equals(b.iloc[20:])


def test_summary_reads_the_gates():
    month = 31 * 6
    bars = pd.DataFrame({"open_ms": [i * BAR_MS for i in range(3 * month)], "ret": [0.001] * month + [-0.0005] * month + [0.001] * month,
                         "market": np.tile([0.01, -0.01], 3 * month // 2), "traded": 0.5, "gross": 1.0,
                         "long": 0.0, "short": 0.0, "funding": 0.0, "cost": -0.0005})
    bars["equity"] = 1000 * (1 + bars["ret"]).cumprod()
    s = ls_sim.summary(bars, recent_ms=2 * month * BAR_MS)
    assert s["total"] == pytest.approx(bars["equity"].iloc[-1] / 1000 - 1) and s["months"] >= 3
    assert s["recent"] == pytest.approx(1.001 ** month - 1)
    assert s["max_drawdown"] == pytest.approx(1 - 0.9995 ** month)
    assert abs(s["beta"]) < 0.01 and s["turnover_per_day"] == pytest.approx(3.0)
    assert s["break_even_side_cost"] == pytest.approx((bars["ret"].sum() + 0.0005 * len(bars)) / (0.5 * len(bars)))
