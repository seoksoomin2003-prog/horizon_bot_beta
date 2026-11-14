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


# v13: keeping a coin while it stays in a band, and limit orders

def test_choose_keeps_inside_the_band_and_fills_the_empty_places():
    p = pd.Series({f"C{i}": i / 10 for i in range(10)})                 # C9 has the highest p
    held = {"C7": 50.0, "C5": 50.0, "C2": -50.0, "C6": -50.0}
    closing, opening, k = ls_sim.choose(p, held, 2, 0.3, min_coins=10)
    assert set(closing) == {"C5", "C6"}                                 # C7 is in the top three, C2 in the bottom three
    assert opening == {"C9": 1.0, "C0": -1.0} and k == 2                # one place a side is left to fill
    assert ls_sim.choose(p, held, 2, 0.3, min_coins=11) == (list(held), {}, 0)
    assert ls_sim.choose(None, held, 2, 0.3, min_coins=10) == (list(held), {}, 0)


def test_choose_never_keeps_more_than_k_and_a_fallen_long_can_turn_short():
    p = pd.Series({f"C{i}": i / 10 for i in range(10)})
    closing, opening, _ = ls_sim.choose(p, {"C9": 1.0, "C8": 1.0, "C7": 1.0, "C0": 1.0}, 2, 0.5, min_coins=10)
    assert set(closing) == {"C7", "C0"} and opening == {"C0": -1.0, "C1": -1.0}


def test_a_coin_that_stays_is_not_traded_again():
    closes, flat = panel([[10, 10, 10, 10]] * 4), panel([[0, 0, 0, 0]] * 4)
    pred = predictions({0: [0.9, 0.8, 0.2, 0.1], 1: [0.9, 0.8, 0.2, 0.1]})
    bars = ls_sim.run_held(pred, closes, flat, 1, 1, 0.25, side_cost=0.001, min_coins=4)
    assert list(bars["traded"].round(9)) == [1.0, 0.0, round(1000 / 999, 9)]      # opens, keeps, then closes what it holds
    assert bars["equity"].iloc[1] == pytest.approx(999.0)


def test_the_band_holds_a_coin_that_slipped_and_drops_one_that_left():
    closes, flat = panel([[10, 10, 10, 10]] * 5), panel([[0, 0, 0, 0]] * 5)
    pred = predictions({0: [0.9, 0.8, 0.2, 0.1], 1: [0.8, 0.9, 0.1, 0.2], 2: [0.2, 0.9, 0.8, 0.1]})
    wide = ls_sim.run_held(pred, closes, flat, 1, 1, 0.5, side_cost=0.0, min_coins=4)
    assert list(wide["traded"].round(6)) == [1.0, 0.0, 1.0, 1.0]        # AAA and DDD stay second, then BBB replaces AAA
    narrow = ls_sim.run_held(pred, closes, flat, 1, 1, 0.25, side_cost=0.0, min_coins=4)
    assert narrow["traded"].iloc[1] == pytest.approx(2.0)                # without the band both coins are replaced


def windows(limit, low, high, last) -> dict:
    return {"limit": panel(limit), "low": panel(low), "high": panel(high), "last": panel(last)}


def test_limit_orders_fill_only_when_the_price_passes_them():
    nan = float("nan")
    closes, flat = panel([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]]), panel([[0, 0, 0, 0]] * 3)
    pred = predictions({0: [0.9, 0.5, 0.5, 0.1]})                        # buy AAA, sell DDD, then close both
    w = windows(limit=[[10.1, nan, nan, 9.9], [10.0, nan, nan, 10.0], [nan] * 4],
                low=[[10.0, nan, nan, 9.8], [9.9, nan, nan, 9.9], [nan] * 4],          # AAA's buy is passed, DDD's sell is not
                high=[[10.2, nan, nan, 9.9], [10.004, nan, nan, 10.2], [nan] * 4],     # AAA's sell at 10 is not passed by 0.05%
                last=[[10.1, nan, nan, 9.9], [10.2, nan, nan, 10.1], [nan] * 4])
    bars = ls_sim.run_held(pred, closes, flat, 1, 1, 0.25, side_cost=0.001, windows=w, maker_cost=0.0002, min_coins=4)
    first, second = bars.iloc[0], bars.iloc[1]
    assert (first["entries"], first["filled"]) == (2, 1)                 # the short never opened
    assert first["long"] == pytest.approx(0.5 * (10 / 10.1 - 1)) and first["short"] == 0.0     # bought above the close
    assert first["cost"] == pytest.approx(-0.5 * 0.0002) and first["gross"] == pytest.approx(0.5 * 10 / 10.1)
    assert (second["exits"], second["exits_limit"]) == (1, 0)            # the exit missed and went to market at 10.2
    held = 500 * 10 / 10.1
    assert second["long"] * bars["equity"].iloc[0] == pytest.approx(held * 0.02)
    assert second["cost"] * bars["equity"].iloc[0] == pytest.approx(-held * 1.02 * 0.001)


def test_without_5m_bars_nothing_is_entered_and_exits_trade_at_the_close():
    nan = float("nan")
    closes, flat = panel([[10, 10, 10, 10]] * 3), panel([[0, 0, 0, 0]] * 3)
    w = windows(*[[[nan] * 4] * 3] * 4)
    bars = ls_sim.run_held(predictions({0: [0.9, 0.5, 0.5, 0.1]}), closes, flat, 1, 1, 0.25, windows=w, min_coins=4)
    assert bars["filled"].sum() == 0 and bars["traded"].sum() == 0 and bars["equity"].iloc[-1] == 1000.0


def test_held_run_never_reads_later_prices():
    rng = np.random.default_rng(5)
    prices = np.exp(np.cumsum(rng.normal(0, 0.02, (40, 4)), axis=0))
    pred = predictions({t: list(rng.random(4)) for t in range(38)})
    flat = panel([[0, 0, 0, 0]] * 40)
    changed = prices.copy()
    changed[21:] *= 1.5
    a = ls_sim.run_held(pred, panel(prices.tolist()), flat, 2, 1, 0.5, min_coins=4)
    b = ls_sim.run_held(pred, panel(changed.tolist()), flat, 2, 1, 0.5, min_coins=4)
    assert a.iloc[:20].equals(b.iloc[:20]) and not a.iloc[20:].equals(b.iloc[20:])


# v14: a book whose coins have a planned exit time

def scores(up: list[float], up_bars: list[int], down: list[float], down_bars: list[int]) -> pd.DataFrame:
    names = [f"C{i}" for i in range(len(up))]
    return pd.DataFrame({"up": up, "up_bars": up_bars, "down": down, "down_bars": down_bars}, index=names)


def test_plan_opens_the_first_of_each_order_with_their_own_exit_times():
    now = scores(up=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], up_bars=[2] * 5 + [6] * 5, down=[5, 5, 5, 5, 5, 5, 5, 5, -3, -9], down_bars=[18] * 9 + [42])
    closing, extend, opening, k = ls_sim.plan(now, {}, 100 * BAR_MS, 0.1, min_coins=10)
    assert (closing, extend, k) == ([], {}, 1)
    assert opening == {"C0": (1.0, 102 * BAR_MS), "C9": (-1.0, 142 * BAR_MS)}
    assert ls_sim.plan(now, {"C5": (1.0, 0)}, 100 * BAR_MS, 0.1, min_coins=11) == (["C5"], {}, {}, 0)


def test_plan_leaves_a_coin_alone_until_its_exit_time():
    now = scores(up=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], up_bars=[2] * 10, down=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], down_bars=[6] * 10)
    t = 100 * BAR_MS
    held = {"C9": (50.0, t + BAR_MS), "C8": (-50.0, t + BAR_MS)}            # both now sit at the wrong end of their order
    assert ls_sim.plan(now, held, t, 0.1, min_coins=10) == ([], {}, {}, 1)


def test_plan_at_the_exit_time_keeps_a_coin_inside_the_band_and_replaces_one_outside():
    now = scores(up=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], up_bars=[2, 6, 18, 42, 2, 2, 2, 2, 2, 2], down=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], down_bars=[6] * 10)
    t = 100 * BAR_MS
    closing, extend, opening, _ = ls_sim.plan(now, {"C2": (50.0, t), "C1": (-50.0, t - BAR_MS)}, t, 0.3, min_coins=10)
    assert extend == {"C2": t + 18 * BAR_MS}                                # third of ten for longs: inside 30%
    assert closing == ["C1"] and opening == {"C9": (-1.0, t + 6 * BAR_MS)}  # ninth for shorts: closed and replaced
    closing, extend, opening, _ = ls_sim.plan(now, {"C2": (50.0, t)}, t, 0.1, min_coins=10)
    assert closing == ["C2"] and extend == {} and opening["C0"] == (1.0, t + 2 * BAR_MS)


def test_plan_never_holds_a_coin_on_both_sides():
    now = scores(up=[9] + [0] * 9, up_bars=[2] * 10, down=[-9] + [0] * 9, down_bars=[2] * 10)      # C0 leads both orders
    _, _, opening, _ = ls_sim.plan(now, {}, 0, 0.1, min_coins=10)
    assert opening == {"C0": (1.0, 2 * BAR_MS), "C1": (-1.0, 2 * BAR_MS)}


def test_plan_keeps_a_coin_it_would_only_close_and_buy_back():
    now = scores(up=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], up_bars=[2] * 10, down=[-9, 8, 7, 6, 5, 4, 3, 2, 1, 0], down_bars=[6] * 10)
    t = 100 * BAR_MS
    held = {"C0": (-50.0, t + BAR_MS), "C1": (50.0, t)}      # C0 leads the longs but is held short, so C1 is next in line
    assert ls_sim.plan(now, held, t, 0.1, min_coins=10) == ([], {"C1": t + 2 * BAR_MS}, {}, 1)


def planned_case(turn_over: bool) -> pd.DataFrame:
    coins = [f"C{i}" for i in range(10)]
    grid = [i * BAR_MS for i in range(8)]
    closes, flat = pd.DataFrame(10.0, index=grid, columns=coins), pd.DataFrame(0.0, index=grid, columns=coins)
    row = lambda t, s, up, down: {"symbol": s, "close_ms": t, "up": up, "up_bars": 2, "down": down, "down_bars": 6}
    first = lambda i: (9 - i, 9 - i)                                  # C0 leads the longs, C9 the shorts
    later = lambda i: ((i if i > 0 else -1, 9 - i) if turn_over else first(i))     # C0 falls to the end of the longs
    pred = pd.DataFrame([row(t, s, *(first(i) if t == 0 else later(i))) for t in grid for i, s in enumerate(coins)])
    return ls_sim.run_planned(pred, closes, flat, 0.1, side_cost=0.001, min_coins=10)


def test_planned_book_keeps_a_coin_that_is_still_first_without_trading():
    bars = planned_case(turn_over=False)
    assert list(bars["traded"].round(2)) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]      # opens once, closes at the very end
    assert list(bars["kept"]) == [0, 0, 1, 0, 1, 0, 2] and bars["planned_bars"].iloc[0] == 8
    assert bars["equity"].iloc[-1] == pytest.approx(1000 - 1.0 - 1.0)       # the cost of opening 1,000 and closing 1,000


def test_planned_book_replaces_a_coin_only_when_its_period_is_over():
    bars = planned_case(turn_over=True)
    assert list(bars["traded"].round(2)) == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]      # C0 leaves after two bars, not before
    assert list(bars["exits"]) == [0, 0, 1, 0, 0, 0, 0] and list(bars["entries"]) == [2, 0, 1, 0, 0, 0, 0]


def test_planned_book_never_reads_later_prices():
    rng = np.random.default_rng(9)
    coins = [f"C{i}" for i in range(10)]
    grid = [i * BAR_MS for i in range(40)]
    prices = np.exp(np.cumsum(rng.normal(0, 0.02, (40, 10)), axis=0))
    flat = pd.DataFrame(0.0, index=grid, columns=coins)
    pred = pd.DataFrame([{"symbol": s, "close_ms": t, "up": rng.normal(), "up_bars": int(rng.choice([2, 6])),
                          "down": rng.normal(), "down_bars": int(rng.choice([2, 6]))} for t in grid for s in coins])
    changed = prices.copy()
    changed[21:] *= 1.5
    a = ls_sim.run_planned(pred, pd.DataFrame(prices, index=grid, columns=coins), flat, 0.5, min_coins=10)
    b = ls_sim.run_planned(pred, pd.DataFrame(changed, index=grid, columns=coins), flat, 0.5, min_coins=10)
    assert a.iloc[:20].equals(b.iloc[:20]) and not a.iloc[20:].equals(b.iloc[20:])
