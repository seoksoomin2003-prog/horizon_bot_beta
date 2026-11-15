"""The quant research layers (research/alpha.py, risk.py, portfolio.py): no look-ahead, and the arithmetic of the
cost-aware optimizer and the book."""
import math

import numpy as np
import pandas as pd
import pytest

from research import alpha, portfolio, risk
from research.h4_data import BAR_MS

DAY = 86_400_000
T0 = 1_600_000_000_000 // BAR_MS * BAR_MS


def test_sigma_and_beta_recover_known_values_and_ignore_the_future():
    rng = np.random.default_rng(1)
    n = 3000
    m = rng.normal(0, 0.01, n)
    rets = pd.DataFrame({"A": 2 * m + rng.normal(0, 0.005, n), "B": rng.normal(0, 0.03, n)}, index=T0 + BAR_MS * np.arange(n))
    sig, b = risk.sigma(rets), risk.beta(rets, pd.Series(m, index=rets.index))
    assert abs(sig["B"].iloc[-500:].mean() - 0.03) < 0.003
    assert abs(b["A"].iloc[-1] - 2) < 0.1 and abs(b["B"].iloc[-1]) < 0.5
    assert b["A"].iloc[:risk.BETA_MIN_BARS - 1].isna().all() and sig["A"].iloc[:risk.VOL_MIN_BARS - 1].isna().all()
    other = rets.copy()
    other.iloc[2000:] *= 5                                   # a different future
    assert np.allclose(risk.sigma(other).iloc[:2000], sig.iloc[:2000], equal_nan=True)
    assert np.allclose(risk.beta(other, pd.Series(m, index=rets.index)).iloc[:2000], b.iloc[:2000], equal_nan=True)


def test_portfolio_vol_is_the_market_part_plus_each_coins_own_part():
    w, sig, b = np.array([0.5, -0.5]), np.array([0.02, 0.04]), np.array([1.0, 1.0])
    own = np.maximum(sig ** 2 - 0.01 ** 2, risk.RESIDUAL_FLOOR * sig ** 2)
    assert math.isclose(risk.portfolio_vol(w, sig, b, 0.01), math.sqrt((0.25 * own).sum()))      # beta cancels


def bars(turnover_per_bar: float, first_ms: int, last_ms: int) -> np.ndarray:
    t = np.arange(first_ms, last_ms, BAR_MS)
    return np.column_stack([t, np.ones(len(t)), np.ones(len(t)), np.ones(len(t)), np.ones(len(t)), np.full(len(t), turnover_per_bar)])


def test_universe_uses_only_the_year_before_and_skips_new_or_stopped_coins():
    month = T0 + 800 * DAY
    data = {"BIG": bars(10, T0, month + 400 * DAY), "SMALL": bars(1, T0, month + 400 * DAY),
            "NEW": bars(100, month - 100 * DAY, month + 400 * DAY),          # listed less than half a year before
            "GONE": bars(100, T0, month - 10 * DAY),                          # stopped trading
            "LATER": np.vstack([bars(0.5, T0, month), bars(1000, month, month + 400 * DAY)])}     # big only afterwards
    assert alpha.universe(data, month, 2) == ["BIG", "SMALL"]
    assert alpha.universe(data, month, 3) == ["BIG", "SMALL", "LATER"]


def panel(n_close: int = 6, coins: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_close):
        for k in range(coins):
            rows.append({"symbol": f"C{k:02d}", "close_ms": T0 + c * BAR_MS, "x": rng.normal(), "fwd": rng.normal(0, 0.02),
                         "fund_hold": 0.001 * k})
    return pd.DataFrame(rows)


def test_target_is_the_return_after_funding_minus_the_median_at_that_close():
    d = alpha.add_target(panel())
    net = np.expm1(d["fwd"]) - d["fund_hold"]
    assert np.allclose(d["net"], net)
    assert np.allclose(d.groupby("close_ms")["target"].median(), 0)
    one = d[d["close_ms"] == T0]
    assert math.isclose(one["target"].iloc[3], net.iloc[3] - net.iloc[:12].median())


def test_scores_are_centred_ranks_with_unit_spread_and_need_enough_coins():
    d = panel()
    d.loc[0, "x"] = np.nan
    s = alpha.add_scores(d, ["x"])
    first = s[s["close_ms"] == T0]
    assert first["z_x"].iloc[0] == 0.0                     # a missing signal scores 0
    ranked = first["z_x"].iloc[1:]
    assert math.isclose(ranked.max(), math.sqrt(3)) and math.isclose(ranked.min(), -math.sqrt(3))
    assert math.isclose(ranked.mean(), 0, abs_tol=1e-12) and math.isclose(ranked.var(ddof=0), 12 / 10)       # (n + 1) / (n - 1): near 1 for many coins
    few = alpha.add_scores(panel(coins=alpha.MIN_COINS - 1), ["x"])
    assert few["z_x"].isna().all()


def test_ic_is_the_rank_correlation_per_close():
    d = alpha.add_target(panel())
    d["perfect"] = d["target"] ** 3                          # same order as the target
    d["backwards"] = -d["target"]
    assert np.allclose(alpha.ic_by_close(d, "perfect"), 1) and np.allclose(alpha.ic_by_close(d, "backwards"), -1)
    ic = alpha.ic_by_close(d, "x")
    one = d[d["close_ms"] == T0]
    assert math.isclose(ic.iloc[0], one["x"].rank().corr(one["target"].rank()))
    r = alpha.ic_report(ic, 2)
    assert math.isclose(r["mean"], ic.mean()) and r["spaced_n"] == 3


def test_signal_autocorrelation():
    d = panel(n_close=20)
    d["still"] = d["symbol"].str[1:].astype(int)            # never changes
    assert math.isclose(alpha.autocorr(d, "still", 3), 1.0)
    assert abs(alpha.autocorr(d, "x", 1)) < 0.3


def test_alpha_model_learns_only_from_answers_known_before_the_month_and_speaks_in_returns():
    rng = np.random.default_rng(2)
    n_close, coins, h = 400, 15, 3
    rows = []
    for c in range(n_close):
        z = rng.normal(size=coins)
        for k in range(coins):
            sig = 0.01 * (1 + k % 3)
            rows.append({"symbol": f"C{k}", "close_ms": T0 + c * BAR_MS, "s": z[k], "sigma": sig,
                         "fwd": 0.004 * z[k] * sig * math.sqrt(h) / 0.01 + rng.normal(0, 0.001), "fund_hold": 0.0})
    d = alpha.add_scores(alpha.add_target(pd.DataFrame(rows)), ["s"])
    month = T0 + 300 * BAR_MS
    model = alpha.AlphaModel(["s"], h)
    model.fit(d, month)
    assert model.train_rows == int((d["close_ms"] + h * BAR_MS <= month).sum())
    later = d.copy()
    later.loc[later["close_ms"] + h * BAR_MS > month, "target"] = 99.0      # answers not known by then change nothing
    other = alpha.AlphaModel(["s"], h)
    other.fit(later, month)
    assert np.allclose(other.w, model.w) and other.intercept == model.intercept
    test = d[d["close_ms"] >= month]
    a = model.predict(test)
    assert np.corrcoef(a, test["target"])[0, 1] > 0.9
    assert 0.8 < model.k < 1.2                                # already in return units: the calibration leaves it alone
    big = test["sigma"] == 0.03                              # a calmer coin with the same score expects less
    assert abs(a[big.to_numpy()]).mean() > 2 * abs(a[(test["sigma"] == 0.01).to_numpy()]).mean()


def test_the_trade_band_comes_from_the_cost():
    var, cost, lam = np.array([0.01]), 0.001, 10.0
    w = lambda a, w0: portfolio.solve(np.array([a]), var, np.array([w0]), cost, lam)[0]
    assert math.isclose(w(0.011, 0.0), (0.011 - cost) / (2 * lam * 0.01))
    assert math.isclose(w(-0.011, 0.0), (-0.011 + cost) / (2 * lam * 0.01))
    assert w(0.0009, 0.0) == 0.0                              # expected gain below the cost: no trade
    assert w(0.0, 0.004) == 0.004                             # alpha gone, but closing would cost more than the risk saved
    assert math.isclose(w(0.0, 0.03), cost / (2 * lam * 0.01))        # far outside the band: back to its edge


def test_optimize_keeps_beta_near_zero_caps_and_closes_coins_outside_the_universe():
    alpha_ = np.array([0.03, 0.02, -0.01, np.nan, 0.0])
    var, beta_ = np.full(5, 0.004), np.array([1.5, 1.0, 0.8, 1.0, 1.2])
    w0 = np.array([0.0, 0.0, 0.0, 0.05, 0.0])
    w = portfolio.optimize(alpha_, var, beta_, w0, cost=0.001, lam=10.0, cap=1.0, gross=10.0)
    assert w[3] == 0.0 and abs(w @ beta_) < 1e-6
    capped = portfolio.optimize(alpha_, var, beta_, w0, cost=0.001, lam=1.0, cap=0.1, gross=0.25)
    assert np.abs(capped).max() <= 0.1 + 1e-12 and math.isclose(np.abs(capped).sum(), 0.25, rel_tol=1e-6)
    plain = portfolio.optimize(alpha_, var, beta_, w0, cost=0.001, lam=10.0, cap=1.0, gross=10.0, neutral=False)
    over = np.array([0.05, 0.04, -0.03, -0.04, 0.0003])        # too big at lam 1: the risk penalty is raised until it fits
    near = portfolio.optimize(over, var, np.zeros(5), np.zeros(5), cost=0.001, lam=1.0, cap=1.0, gross=0.5)
    assert 0.45 < np.abs(near).sum() <= 0.5 + 1e-9
    lam = (over[0] - 0.001) / (2 * var[0] * near[0])            # the lam that gives coin 0 its weight gives the others theirs
    assert np.allclose(near[:4], portfolio.solve(over[:4], var[:4], np.zeros(4), 0.001, lam)) and near[4] == 0.0
    tiny = portfolio.optimize(np.array([0.0012, 0.03]), np.full(2, 0.004), np.zeros(2), np.zeros(2), cost=0.001, lam=10.0)
    assert tiny[0] == 0.0 and tiny[1] > portfolio.MIN_WEIGHT      # 0.25% of equity is not worth an order
    assert plain[0] > 0 and plain[2] < 0 and plain[4] == 0.0


def test_mark_pays_funding_by_side_and_lets_weights_drift():
    parts, w = portfolio.mark(np.array([0.5, -0.5]), np.array([0.10, 0.10]), np.array([0.001, 0.001]))
    assert math.isclose(parts["long"], 0.05) and math.isclose(parts["short"], -0.05)
    assert math.isclose(parts["funding"], 0.0)               # the long pays what the short receives
    assert np.allclose(w, [0.55, -0.55])
    parts, _ = portfolio.mark(np.array([0.5]), np.array([0.0]), np.array([0.001]))
    assert math.isclose(parts["funding"], -0.0005)


def test_book_pays_the_cost_of_every_trade_and_of_closing_at_the_end():
    idx = T0 + BAR_MS * np.arange(4)
    closes = pd.DataFrame({"A": [100, 110, 110, 110.0], "B": [100, 100, 100, 100.0]}, index=idx)
    zero = closes * 0
    sig = closes * 0 + 0.1                                    # per-bar sigma; horizon 1 gives var 0.01
    beta_ = closes * 0 + 1.0
    al = pd.DataFrame([{"symbol": "A", "close_ms": idx[0], "alpha": 0.011}, {"symbol": "B", "close_ms": idx[0], "alpha": np.nan}])
    book = portfolio.run(al, closes, zero, sig, beta_, 1, lam=10.0, side_cost=0.001, cap=1.0, gross=1.0)
    w = (0.011 - 0.001) / (2 * 10 * 0.01)                     # 0.05 of equity long A; beta neutrality has only A: mu pushes it to 0
    assert len(book) == 3
    neutral_off = portfolio.optimize(np.array([0.011]), np.array([0.01]), np.array([1.0]), np.zeros(1), 0.001, 10.0, 1.0, 1.0)
    assert abs(neutral_off[0]) < 1e-6 and book["traded"].sum() < 1e-6
    loose = portfolio.run(al, closes, zero, sig, beta_ * 0, 1, lam=10.0, side_cost=0.001, cap=1.0, gross=1.0)   # beta 0: no hedge
    first = loose.iloc[0]
    assert math.isclose(first["traded"], w) and math.isclose(first["long"], w * 0.10)
    assert math.isclose(first["ret"], w * 0.10 - w * 0.001)
    end = 1000 * (1 + w * 0.10 - w * 0.001)
    held = w * 1.1 / (1 + w * 0.10)
    assert math.isclose(loose["equity"].iloc[-1], end * (1 - held * 0.001))      # the drifted weight is closed at the end
    assert loose["traded"].iloc[1] == 0.0                     # no new alphas: nothing traded in between
