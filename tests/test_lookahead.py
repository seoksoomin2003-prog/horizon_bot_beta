"""Nothing decided at a time T may change when the data after T changes: the research walk-forward, the market-state
weights and the bot's own signals."""
import numpy as np
import pandas as pd

from research.direction import (BAR_MS, FEATURES, MARKET_FEATURES, RANK_FEATURES, REGIMES, add_ranks, add_regime, add_relative, frame,
                                walk_forward)
from tests.test_direction import T0, bars

CUT = 1100                          # bars from here on are replaced by a different future
LAST_SAME = T0 + CUT * BAR_MS       # the last close both versions share
REFERENCE = {"A", "B", "C"}
FUNDING = np.array([[T0 + i * 8 * 3_600_000, 0.0001] for i in range(800)])


def market(other_future: bool) -> pd.DataFrame:
    frames = []
    for k, name in enumerate(["A", "B", "C", "D"]):
        b = bars(1500, seed=k + 1)
        if other_future:
            rng = np.random.default_rng(100 + k)
            b[CUT:, 1:5] *= np.exp(np.cumsum(rng.normal(0, 0.03, len(b) - CUT)))[:, None]
            b[CUT:, 5] *= rng.uniform(0.2, 5.0, len(b) - CUT)
        frames.append(frame(name, b, FUNDING, None, 2))
    return pd.concat(frames, ignore_index=True)


def predictions(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    pred = walk_forward(data, 2, T0, T0 + 700 * BAR_MS, T0 + 1500 * BAR_MS, False, features=features, train_symbols=REFERENCE)
    return pred.set_index(["symbol", "close_ms"])[["p", "base", "thr", "thr95", "thr98"]]


def same_before_and_different_after(a: pd.DataFrame, b: pd.DataFrame) -> None:
    before = a.index.get_level_values("close_ms") <= LAST_SAME
    assert before.sum() > 1000 and (~before).sum() > 100
    assert a.index.equals(b.index)
    assert np.array_equal(a[before].to_numpy(), b[before].to_numpy())       # identical to the last bit
    assert not np.allclose(a[~before]["p"], b[~before]["p"])                # and the check can fail


def test_walk_forward_predictions_ignore_everything_after_them():
    """End to end for the v9 model: features, the market median, scaling, the fit and the confidence cutoffs."""
    run = lambda other: predictions(add_relative(market(other), REFERENCE), FEATURES + MARKET_FEATURES)
    same_before_and_different_after(run(False), run(True))


def test_rank_features_ignore_everything_after_them():
    run = lambda other: predictions(add_ranks(add_relative(market(other), REFERENCE), REFERENCE, at_least=3),
                                    FEATURES + MARKET_FEATURES + RANK_FEATURES)
    same_before_and_different_after(run(False), run(True))


def test_market_state_weights_ignore_everything_after_them():
    def run(other):
        data, names = add_regime(add_relative(market(other), REFERENCE), REFERENCE, FEATURES + MARKET_FEATURES)
        return data, predictions(data, FEATURES + MARKET_FEATURES + names)
    (a, pa), (b, pb) = run(False), run(True)
    same_before_and_different_after(pa, pb)
    known = a["regime"].notna()
    assert known.sum() > 2000 and set(a.loc[known, "regime"]) <= {0.0, 1.0, 2.0, 3.0}
    assert (a.loc[known, [f"is_{r}" for r in REGIMES]].sum(axis=1) == 1).all()      # exactly one state at a time
    row = a[known].iloc[-1]
    state = REGIMES[int(row["regime"])]
    assert row[f"r1@{state}"] == row["r1"] and all(row[f"r1@{r}"] == 0 for r in REGIMES if r != state)
    d = a[a["symbol"] == "D"].set_index("close_ms")["regime"]       # a coin outside the reference never sets the state
    assert d.dropna().equals(a[a["symbol"] == "A"].set_index("close_ms")["regime"].reindex(d.dropna().index))


def test_bot_signals_ignore_later_candles():
    from signal_engine import SignalEngine
    def candles(other):
        b = bars(400, seed=9)
        if other:
            b[300:, 1:5] *= np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.03, 100)))[:, None]
        return pd.DataFrame({"timestamp": pd.to_datetime(b[:, 0], unit="ms"), "open": b[:, 1], "high": b[:, 2], "low": b[:, 3],
                             "close": b[:, 4], "volume": b[:, 5]})
    engine = SignalEngine()
    a, b = engine.compute_all_signals(candles(False), "X"), engine.compute_all_signals(candles(True), "X")
    columns = ["adx", "atr", "atr_percentile", "ema50", "ema200", "rsi", "regime", "trend_bias", "trigger", "score", "confidence"]
    pd.testing.assert_frame_equal(a.loc[:299, columns], b.loc[:299, columns])
    assert not a.loc[300:, "close"].equals(b.loc[300:, "close"])
