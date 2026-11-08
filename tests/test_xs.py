import numpy as np
import pytest

from research.h4_sim import BAR_MS
from research.xs_sim import UNIVERSE, WEEK_BARS, WEEK_MS, Coin, mondays, pick, run

MONDAY = 1_704_067_200_000          # 2024-01-01 00:00 UTC, a Monday
START = MONDAY + 5 * WEEK_MS        # more than 28 days of history before it


def coin(symbol, growth, n_weeks=8, funding_rate=0.0, jump=None):
    n = n_weeks * WEEK_BARS
    close = 100 * np.cumprod(np.full(n, 1 + growth))
    if jump:
        at, factor = jump
        close[at:] *= factor
    open_ = np.concatenate([[100.0], close[:-1]])
    t = MONDAY + np.arange(n) * BAR_MS
    bars = np.column_stack([t, open_, np.maximum(open_, close), np.minimum(open_, close), close, np.ones(n)])
    stamps = MONDAY + np.arange(0, n * BAR_MS, 8 * 3_600_000)
    funding = np.column_stack([stamps, np.full(len(stamps), funding_rate)])
    return Coin(symbol, bars, funding)


def test_mondays_are_mondays_with_whole_weeks():
    weeks = mondays(MONDAY + 1, MONDAY + 3 * WEEK_MS)
    assert weeks == [MONDAY + WEEK_MS, MONDAY + 2 * WEEK_MS]


def test_momentum_buys_the_strongest_and_reversal_the_weakest():
    scores = {s: v for s, v in zip(UNIVERSE[:5], [0.1, -0.2, 0.3, 0.0, -0.1])}
    assert pick("X1", scores) == (["SOLUSDT", "AVAXUSDT"], ["XRPUSDT", "BNBUSDT"])
    assert pick("X2", scores) == (["XRPUSDT", "BNBUSDT"], ["SOLUSDT", "AVAXUSDT"])


def test_ties_follow_universe_order():
    scores = {s: 0.0 for s in UNIVERSE[:4]}
    assert pick("X1", scores) == (["AVAXUSDT", "DOGEUSDT"], ["SOLUSDT", "XRPUSDT"])


def test_ranking_uses_only_bars_before_the_monday_open():
    a = coin("SOLUSDT", 0.001)
    b = coin("SOLUSDT", 0.001, jump=(int((START - MONDAY) // BAR_MS), 5.0))   # jumps exactly at the Monday open
    i = a.week(START)
    assert a.score("X1", i, START) == pytest.approx(b.score("X1", i, START))
    assert a.score("X2", i, START) == pytest.approx(b.score("X2", i, START))


def test_trending_spread_earns_and_costs_are_charged():
    coins = [coin(s, g) for s, g in zip(UNIVERSE[:4], [0.002, 0.001, -0.001, -0.002])]
    free = run("X1", coins, START, START + 2 * WEEK_MS, side_cost=0.0)
    paid = run("X1", coins, START, START + 2 * WEEK_MS, side_cost=0.001)
    assert free["cash"] > 1000 and paid["cash"] < free["cash"] and free["hit_rate"] == 1.0
    assert len(free["curve"]) == 2 * WEEK_BARS and free["traded_weeks"] == 2
    # Entry and exit at 1x gross: about 2 x 1000 x 0.1% plus the small rebalance between weeks.
    assert free["cash"] - paid["cash"] == pytest.approx(2.0, abs=0.3)


def test_market_moving_together_gives_no_profit():
    coins = [coin(s, 0.002) for s in UNIVERSE[:4]]
    r = run("X1", coins, START, START + WEEK_MS, side_cost=0.0)
    assert r["cash"] == pytest.approx(1000.0) and r["bench"][0] > 0


def test_funding_paid_by_longs_and_received_by_shorts():
    rich = [coin(s, 0.0, funding_rate=0.001) for s in UNIVERSE[:2]]
    poor = [coin(s, 0.0, funding_rate=-0.001) for s in UNIVERSE[2:4]]
    r = run("X3", rich + poor, START, START + WEEK_MS, side_cost=0.0)
    # Carry shorts the high payers and buys the negative ones: 21 settlements a week, 1000 notional in all, 0.1% each.
    assert r["cash"] == pytest.approx(1000 + 21 * 1000 * 0.001)


def test_fewer_than_four_coins_stays_flat():
    coins = [coin(s, 0.001) for s in UNIVERSE[:3]]
    r = run("X1", coins, START, START + WEEK_MS)
    assert r["traded_weeks"] == 0 and r["cash"] == 1000.0


def test_young_coin_is_not_ranked():
    c = coin("SOLUSDT", 0.001)
    assert c.week(MONDAY + 4 * WEEK_MS) is None and c.week(START) is not None


def test_account_wiped_out_by_a_short_squeeze_stops_as_ruined():
    # Reversal shorts last week's strongest coin; it then jumps 6x inside the week.
    jump_bar = int((START - MONDAY) // BAR_MS) + 5
    coins = [coin(UNIVERSE[0], 0.002, jump=(jump_bar, 6.0)), coin(UNIVERSE[1], 0.001), coin(UNIVERSE[2], -0.001), coin(UNIVERSE[3], -0.002)]
    r = run("X2", coins, START, START + 2 * WEEK_MS, side_cost=0.0)
    assert r["ruined_ms"] is not None and r["cash"] == 0.0 and r["weeks"][-1] == -1.0
    assert all(v >= 0 for _, v in r["curve"])
