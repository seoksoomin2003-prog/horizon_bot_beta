"""v17 (research/timed.py): the market forecast's timing and the long/short-or-flat simulator on tiny hand-made data."""
import math

import numpy as np
import pandas as pd

from research import timed
from research.h4_data import BAR_MS, utc_ms
from research.topcoin import TAKER_COST, Panel

T, N = 40, 3
C_RT = timed.ROUND_TRIP


def panel(sigma: float = 0.02) -> Panel:
    close = np.full((T, N), 100.0)
    return Panel(times=np.arange(T) * BAR_MS, symbols=["A", "B", "C"], open=close.copy(), high=close.copy(), low=close.copy(),
                 close=close, sigma=np.full((T, N), sigma), funding=np.zeros((T, N)), alpha=np.zeros((T, N)))


def test_forecast_uses_only_answered_closes_and_is_zero_without_enough_of_them():
    idx = np.arange(utc_ms("2021-01-01"), utc_ms("2022-03-01"), BAR_MS)
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    m_hat, slopes = timed.forecast(market, utc_ms("2022-01-01"), utc_ms("2022-03-01"))
    assert m_hat[idx < utc_ms("2022-01-01")].isna().all() and m_hat[idx >= utc_ms("2022-01-01")].notna().all()
    later = market.copy()
    later[idx >= utc_ms("2022-02-01")] = 0.05                 # the future must not change January's forecast
    m2, s2 = timed.forecast(later, utc_ms("2022-01-01"), utc_ms("2022-03-01"))
    jan = (idx >= utc_ms("2022-01-01")) & (idx < utc_ms("2022-02-01"))
    assert np.allclose(m_hat[jan], m2[jan]) and s2[utc_ms("2022-01-01")] == slopes[utc_ms("2022-01-01")]
    short = market[idx >= utc_ms("2021-12-01")]                # under MIN_ROWS answered closes: no forecast
    m3, s3 = timed.forecast(short, utc_ms("2022-01-01"), utc_ms("2022-02-01"))
    assert set(s3.values()) == {0.0}


def test_forecast_finds_a_planted_trend():
    idx = np.arange(utc_ms("2021-01-01"), utc_ms("2022-02-01"), BAR_MS)
    regime = np.where((np.arange(len(idx)) // 600) % 2 == 0, 0.003, -0.003)    # long up and down runs
    market = pd.Series(regime + np.random.default_rng(1).normal(0, 0.01, len(idx)), index=idx)
    _, slopes = timed.forecast(market, utc_ms("2022-01-01"), utc_ms("2022-02-01"))
    assert slopes[utc_ms("2022-01-01")] > 0


def test_long_enters_above_cost_sized_by_volatility_pays_funding_and_exits_when_mu_turns_negative():
    p = panel(sigma=0.02)
    p.close[1:, 0] = 110.0
    p.funding[:, 0] = 0.0001
    MU = np.full((T, N), 0.0)
    MU[:, 0] = np.where(np.arange(T) < 5, 0.01, -0.001)
    res = timed.simulate(p, MU, timed.RULES_C, 0, 10)
    tr = res.trades[0]
    assert tr["symbol"] == "A" and tr["side"] == 1 and tr["why"] == "signal" and tr["exit_ms"] == p.times[5]
    w = timed.TARGET_BAR_VOL / 0.02
    q = w * 1000 / 100
    expected = 1000 - w * 1000 * TAKER_COST - q * 100 * 0.0001 + q * 10 - 4 * q * 110 * 0.0001 - q * 110 * TAKER_COST
    assert math.isclose(res.equity[-1], expected) and math.isclose(tr["net"], expected / 1000 - 1)


def test_flat_when_no_coin_beats_the_cost():
    p = panel()
    MU = np.full((T, N), C_RT * 0.9)
    res = timed.simulate(p, MU, timed.RULES_C, 0, 10)
    assert res.orders == 0 and res.equity[-1] == 1000


def test_switches_only_when_the_gap_beats_twice_the_round_trip():
    p = panel()
    MU = np.zeros((T, N))
    MU[:, 0] = 0.01
    MU[3:, 1] = 0.01 + 2 * C_RT * 0.9               # higher, but not by enough
    MU[6:, 1] = 0.01 + 2 * C_RT * 1.1
    res = timed.simulate(p, MU, timed.RULES_C, 0, 10)
    assert [(tr["symbol"], tr["why"]) for tr in res.trades] == [("A", "switch"), ("B", "open")]
    assert res.trades[0]["exit_ms"] == p.times[6]


def test_short_gains_on_a_fall_and_receives_positive_funding_only_under_rule_d():
    p = panel(sigma=0.05)                              # 0.05 > target: notional below the equity
    p.close[1:, 2] = 90.0
    p.funding[:, 2] = 0.0002
    MU = np.full((T, N), 0.0)
    MU[:, 2] = -0.02
    assert timed.simulate(p, MU, timed.RULES_C, 0, 3).orders == 0
    res = timed.simulate(p, MU, timed.RULES_D, 0, 3)
    w = timed.TARGET_BAR_VOL / 0.05
    q = w * 1000 / 100
    expected = 1000 - w * 1000 * TAKER_COST + q * 100 * 0.0002 + q * 10 + q * 90 * 0.0002 + q * 90 * 0.0002
    assert res.trades[0]["side"] == -1 and math.isclose(res.equity[-1], expected)


def test_best_ranks_by_risk_adjusted_score_and_shifted_keeps_the_values():
    mu_row = np.array([0.03, 0.02, np.nan])
    sig = np.array([0.06, 0.02, 0.01])
    assert timed.best(mu_row, sig, np.full(3, 100.0), (1,))[0] == 1          # 0.02 / 0.02 beats 0.03 / 0.06
    m = np.arange(5000, dtype=float)
    s = timed.shifted(m, 100, 4999, seed=3)
    assert np.array_equal(s[:100], m[:100]) and sorted(s[100:]) == sorted(m[100:]) and not np.array_equal(s, m)
