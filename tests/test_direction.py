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
