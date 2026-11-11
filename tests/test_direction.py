"""The v3 direction model: it learns a real signal, sees no future, and trains only on answers already known."""
import numpy as np
import pandas as pd

from research.direction import BAR_MS, FEATURES, fit, frame, predict, trades

T0 = 1_609_459_200_000 - 200 * BAR_MS     # 200 bars before 2021-01-01


def bars(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    t = T0 + np.arange(n) * BAR_MS
    return np.column_stack([t, close, close * 1.01, close * 0.99, close, rng.uniform(1, 2, n)])


FUNDING = np.array([[T0 + i * 8 * 3_600_000, 0.0001] for i in range(400)])


def test_fit_finds_a_real_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3))
    y = (rng.random(5000) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
    w = fit(x, y)
    assert w[1] > 1.0 and abs(w[2]) < 0.2 and abs(w[3]) < 0.2
    assert predict(w, np.array([[2.0, 0, 0]]))[0] > 0.9


def test_features_do_not_use_the_future():
    b = bars()
    changed = b.copy()
    changed[400:, 1:5] *= 3.0
    a, c = frame("X", b, FUNDING, None, 6), frame("X", changed, FUNDING, None, 6)
    row = 399     # last close before the change
    assert np.allclose(a.loc[row, FEATURES].to_numpy(float), c.loc[row, FEATURES].to_numpy(float), equal_nan=True)
    assert not np.isclose(a.loc[399, "fwd"], c.loc[399, "fwd"])    # only the answer looks ahead


def test_training_uses_only_answers_known_before_the_month(monkeypatch):
    import research.direction as d
    data = frame("X", bars(900), FUNDING, None, 6)
    sizes = []
    real_fit = d.fit
    monkeypatch.setattr(d, "fit", lambda x, y: sizes.append(len(x)) or real_fit(x, y))
    pred = d.walk_forward(data, 6, T0, int(data["close_ms"].iloc[500]), int(data["close_ms"].iloc[-1]), daily_only=False)
    usable = data.dropna(subset=FEATURES + ["fwd", "sd42"])
    expected = [int(((usable["close_ms"] + 6 * BAR_MS) <= m).sum()) for m in sorted(pred["month"].unique())]
    assert sizes == expected and len(sizes) >= 2
    assert (pred["close_ms"] >= pred["month"]).all()


def test_no_bet_when_the_expected_gain_is_below_cost():
    pred = pd.DataFrame({"p": [0.51, 0.9, 0.1], "sd42": [0.01, 0.01, 0.01], "fwd": [0.02, 0.02, 0.02],
                         "fund_hold": [0.0, 0.0001, 0.0001], "y": [1.0, 1.0, 1.0], "close_ms": [0, 0, 0]})
    t = trades(pred, 1, 0.001)
    assert list(t["dir"]) == [1.0, -1.0]
    assert abs(t["ret"].iloc[0] - (np.expm1(0.02) - 0.002 - 0.0001)) < 1e-12
    assert list(t["hit"]) == [1.0, 0.0]


def test_v4_fixed_threshold_replaces_the_cost_threshold():
    pred = pd.DataFrame({"p": [0.6, 0.9], "sd42": [0.02, 0.02], "fwd": [0.01, 0.01], "fund_hold": [0.0, 0.0],
                         "y": [1.0, 1.0], "close_ms": [0, 0]})
    # expected gains: 0.2 * 0.02 * 0.798 = 0.32% and 0.8 * 0.02 * 0.798 = 1.28%
    assert len(trades(pred, 1, 0.001)) == 2
    assert list(trades(pred, 1, 0.002, min_gain=0.005)["p"]) == [0.9]


def flow_rows(b, seed=2):
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1, 2, len(b))
    return np.column_stack([b[:, 0], vol, vol * rng.uniform(0.3, 0.7, len(b)), rng.normal(0, 1e-4, len(b))])


def test_flow_features_do_not_use_the_future():
    from research.direction import FLOW_FEATURES
    b = bars()
    f = flow_rows(b)
    changed = f.copy()
    changed[400:, 2] = changed[400:, 1]          # every later bar becomes all buying
    changed[400:, 3] = 0.01
    a, c = frame("X", b, FUNDING, None, 6, f), frame("X", b, FUNDING, None, 6, changed)
    cols = FEATURES + FLOW_FEATURES
    assert np.allclose(a.loc[399, cols].to_numpy(float), c.loc[399, cols].to_numpy(float), equal_nan=True)
    assert not np.isclose(a.loc[400, "taker_1"], c.loc[400, "taker_1"])
    assert abs(a.loc[399, "taker_1"] - (f[399, 2] / f[399, 1] - 0.5)) < 1e-12


def test_final_check_period_is_closed_to_flow_development_code():
    import pytest
    from research import h4_data
    with pytest.raises(PermissionError):
        h4_data.load_flow("BTCUSDT", end_ms=h4_data.VAL_END_MS + 1)
    with pytest.raises(PermissionError):
        h4_data.download_flow(end_ms=h4_data.VAL_END_MS + 1)


def test_external_daily_value_is_used_only_from_the_next_utc_day():
    from research.direction import DAY_MS, add_external
    day = T0 // DAY_MS * DAY_MS
    daily = np.array([[day, 100.0], [day + DAY_MS, 110.0], [day + 2 * DAY_MS, 121.0]])
    hourly = np.array([[day + h * 3_600_000, 50.0 + h] for h in range(80)])
    data = pd.DataFrame({"close_ms": [day + DAY_MS + 20 * 3_600_000, day + 2 * DAY_MS, day + 2 * DAY_MS + BAR_MS]})
    out = add_external(data, {"spx": daily, "vix": daily, "dxy": daily, "dvol": hourly})
    assert np.isnan(out["spx_r"].iloc[0])                        # only day 0 is known late on day 1, and it has no return yet
    assert np.isclose(out["spx_r"].iloc[1], np.log(1.1))         # day 1's close is usable from day 2 00:00
    assert np.isclose(out["vix"].iloc[2], np.log(110.0))         # day 2's own close is still unknown during day 2
    assert np.isclose(out["dvol"].iloc[1], np.log(50.0 + 47))    # the hour that closed exactly at the bar close
    assert np.isclose(out["dvol_change"].iloc[2], np.log(50.0 + 51) - np.log(50.0 + 27))


def test_transfer_learns_only_from_the_named_coins(monkeypatch):
    import research.direction as d
    a, b = frame("A", bars(900, seed=1), FUNDING, None, 1), frame("B", bars(900, seed=2), FUNDING, None, 1)
    data = pd.concat([a, b], ignore_index=True)
    sizes = []
    real_fit = d.fit
    monkeypatch.setattr(d, "fit", lambda x, y: sizes.append(len(x)) or real_fit(x, y))
    pred = d.walk_forward(data, 1, T0, int(a["close_ms"].iloc[500]), int(a["close_ms"].iloc[-1]), False, train_symbols={"A"})
    usable = a.dropna(subset=FEATURES + ["fwd", "sd42"])
    assert sizes == [int(((usable["close_ms"] + BAR_MS) <= m).sum()) for m in sorted(pred["month"].unique())]
    assert set(pred["symbol"]) == {"A", "B"} and (pred["thr"] > 0).all()


def test_limit_order_fills_only_when_the_price_trades_through():
    from research.direction import FILL_BARS, M5_MS, fill_windows, maker_trades
    T, X = 10 * BAR_MS, 11 * BAR_MS

    def window(start, limit, low, high, last):
        rows = [[start, limit, limit, limit, limit]] + [[start + k * M5_MS, limit, high, low, limit] for k in range(1, FILL_BARS + 1)]
        rows[-1][4] = last
        return rows

    def run(p, entry, exit_):
        w = fill_windows(np.array(window(T, *entry) + window(X, *exit_), dtype=float))
        w.index = pd.MultiIndex.from_product([["X"], w.index])
        sel = pd.DataFrame({"symbol": ["X"], "close_ms": [T], "p": [p], "fund_hold": [0.001]})
        return maker_trades(sel, w, 1).iloc[0]

    touched = run(0.6, (100, 100, 101, 100.5), (110, 109, 110, 108))       # a buy at 100 with low exactly 100: not filled
    assert not touched["entry_filled"] and not touched["exit_filled"]
    assert np.isclose(touched["net_1x"], 108 / 100.5 - 1 - 0.001 - 2 * 0.0010)   # market in at 100.5, market out at 108
    filled = run(0.6, (100, 99.9, 101, 100.5), (110, 109, 110.1, 108))
    assert filled["entry_filled"] and filled["exit_filled"]
    assert np.isclose(filled["net_1x"], 0.10 - 0.001 - 2 * 0.0002) and np.isclose(filled["net_2x"], 0.10 - 0.001 - 4 * 0.0002)
    short = run(0.4, (100, 99, 100.1, 99.5), (90, 89.9, 91, 92))            # sells above 100, buys back below 90
    assert short["entry_filled"] and short["exit_filled"] and np.isclose(short["move"], 0.10)
    assert np.isclose(short["net_1x"], 0.10 + 0.001 - 2 * 0.0002)           # a short receives the funding a long pays
    w = fill_windows(np.array(window(T, 100, 99, 101, 100)[:-1], dtype=float))   # a missing 5m bar: no window
    assert w.empty


def test_positioning_uses_only_values_recorded_before_the_close():
    from research.direction import M5_MS, POSITION_FEATURES, add_positioning
    data = frame("X", bars(), FUNDING, None, 1)
    close = data["close_ms"].to_numpy()
    rows = lambda at_close: np.array([[t - M5_MS, 100.0 * (1.01 ** i), 2.0, 1.5] for i, t in enumerate(close)]
                                     + [[t, at_close, 9.0, 9.0] for t in close], dtype=float)
    sort = lambda r: r[np.argsort(r[:, 0], kind="stable")]
    a, b = add_positioning(data, {"X": sort(rows(1.0))}), add_positioning(data, {"X": sort(rows(1e6))})
    features = sum(POSITION_FEATURES.values(), [])
    assert np.allclose(a[features].to_numpy(float), b[features].to_numpy(float), equal_nan=True)   # the value stamped at the close is ignored
    assert np.isclose(a.loc[300, "oi_change_1"], np.log(1.01)) and np.isclose(a.loc[300, "oi_change_6"], 6 * np.log(1.01))
    assert np.isclose(a.loc[300, "ls_accounts"], np.log(2.0)) and np.isclose(a.loc[300, "ls_top_change"], 0.0)
    stale = add_positioning(data, {"X": np.array([[close[300] - 4 * M5_MS, 100.0, 2.0, 1.5]])})
    assert stale[features].isna().all().all()                                # older than 15 minutes: not used


def test_market_feature_is_the_gap_to_the_median_coin():
    from research.direction import add_market
    data = pd.DataFrame({"symbol": list("ABC"), "close_ms": [1, 1, 1], "r1": [0.01, 0.03, -0.02], "r6": [0.0, 0.1, 0.2]})
    out = add_market(data)
    assert np.allclose(out["xs_r1"], [0.0, 0.02, -0.03]) and np.allclose(out["xs_r6"], [-0.1, 0.0, 0.1])


def test_archive_lists_no_file_from_the_final_check_period():
    import pytest
    from research import h4_data
    files = h4_data.archive_files("BTCUSDT")
    assert files["m5"][-1].endswith("BTCUSDT-5m-2026-08-05.zip") and files["m5"][-6].endswith("BTCUSDT-5m-2026-07.zip")
    assert not any("2026-08.zip" in f for f in files["m5"]) and files["metrics"][-1].endswith("2026-08-05.zip")
    with pytest.raises(PermissionError):
        h4_data.archive_files("BTCUSDT", end_ms=h4_data.VAL_END_MS + 1)
    kept = h4_data.parse_m5("open_time,open\n0,1,2,0.5,1.5,9\n2400000,1,2,0.5,1.5,9\n14400000,3,4,2,3.5,9\n")
    assert kept == [[0, 1.0, 2.0, 0.5, 1.5], [14400000, 3.0, 4.0, 2.0, 3.5]]           # only the first 40 minutes after a 4h close
    text = 'create_time,symbol,a,b,c,d,e,f\n2022-01-05 03:55:00,X,5,500,"","","",""\n2022-01-05 02:00:00,X,5,500,1,2,3,4\n2026-08-01 03:50:00,X,7,1,1.1,1.6,2.2,0.9\n'
    parsed = h4_data.parse_metrics(text)
    assert len(parsed) == 2 and parsed[0][1] == 5.0 and np.isnan(parsed[0][2]) and parsed[1][1:] == [7.0, 2.2, 1.6]


def test_stricter_fill_needs_the_price_to_pass_the_order_by_the_gap():
    from research.direction import FILL_BARS, M5_MS, fill_windows, maker_trades
    T, X = 10 * BAR_MS, 11 * BAR_MS
    rows = []
    for start, limit, low, high in ((T, 100, 99.96, 100.5), (X, 110, 109.5, 110.2)):
        rows += [[start, limit, limit, limit, limit]] + [[start + k * M5_MS, limit, high, low, limit] for k in range(1, FILL_BARS + 1)]
    w = fill_windows(np.array(rows, dtype=float))
    w.index = pd.MultiIndex.from_product([["X"], w.index])
    sel = pd.DataFrame({"symbol": ["X"], "close_ms": [T], "p": [0.6], "fund_hold": [0.0]})
    assert maker_trades(sel, w, 1).iloc[0]["entry_filled"]                      # 99.96 is below 100
    strict = maker_trades(sel, w, 1, gap=0.0005).iloc[0]
    assert not strict["entry_filled"] and strict["exit_filled"]                  # but not below 99.95; the exit passes 110.055


def test_basket_caps_each_trade_at_a_tenth_and_the_total_at_everything():
    from research.direction import basket_returns
    few = pd.DataFrame({"close_ms": [1, 1, 1], "net": [0.01, 0.02, -0.01]})
    many = pd.DataFrame({"close_ms": [2] * 40, "net": [0.01] * 40})
    r = basket_returns(pd.concat([few, many]), "net")
    assert np.isclose(r[1], 0.002) and np.isclose(r[2], 0.01)                    # 3 trades at 10% each; 40 trades share 100%


def test_relative_target_is_measured_against_the_reference_coins_only():
    from research.direction import add_relative
    data = pd.DataFrame({"symbol": ["A", "B", "C", "NEW"] * 2, "close_ms": [1] * 4 + [2] * 4,
                         "r1": [0.01, 0.03, -0.02, 0.50, 0.0, 0.0, 0.0, 0.0], "r6": [0.0] * 8,
                         "fwd": [0.02, -0.01, 0.00, 0.30, 0.01, 0.02, 0.03, -0.9]})
    out = add_relative(data, {"A", "B", "C"})
    first = out[out["close_ms"] == 1].set_index("symbol")
    assert np.allclose(first["xs_r1"], [0.0, 0.02, -0.03, 0.49])                  # NEW's 0.50 does not move the median (0.01)
    assert np.allclose(first["fwd"], [0.02, -0.01, 0.0, 0.30]) and np.allclose(first["fwd_abs"], [0.02, -0.01, 0.0, 0.30])   # median fwd is 0
    assert np.isclose(out[(out["close_ms"] == 2) & (out["symbol"] == "NEW")]["fwd"].iloc[0], -0.92)
    later = data.copy()
    later.loc[later["close_ms"] == 2, "r1"] = 9.0                                 # a later bar never changes an earlier feature
    assert np.allclose(add_relative(later, {"A", "B", "C"}).loc[:3, "xs_r1"], out.loc[:3, "xs_r1"])
