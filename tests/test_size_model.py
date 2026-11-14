"""The v14 size model: it learns the size of the return, never from answers that are not known yet."""

import numpy as np
import pandas as pd
import pytest

from research import size_model
from research.h4_data import BAR_MS, utc_ms

START, SCORE, END = utc_ms("2021-01-01"), utc_ms("2021-07-01"), utc_ms("2021-10-01")


def rows(seed: int = 1, horizon: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = np.arange(START, END, BAR_MS)
    parts = []
    for symbol in ("AAA", "BBB", "CCC"):
        a, b = rng.normal(size=len(grid)), rng.normal(size=len(grid))
        fwd = 0.02 * a - 0.01 * b + rng.normal(0, 0.01, len(grid))
        fwd[-horizon:] = np.nan                                      # the last closes have no answer yet
        parts.append(pd.DataFrame({"symbol": symbol, "close_ms": grid, "a": a, "b": b, "fwd": fwd}))
    return pd.concat(parts, ignore_index=True)


def test_fit_finds_the_weights_and_the_level():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 2))
    x -= x.mean(axis=0)                                              # the model is given standardized features
    w, level = size_model.fit(x, 0.5 + x @ np.array([2.0, -1.0]) + rng.normal(0, 0.1, 5000), lam=1.0)
    assert w == pytest.approx([2.0, -1.0], abs=0.02) and level == pytest.approx(0.5, abs=0.02)


def test_walk_forward_predicts_the_size_and_also_the_rows_without_an_answer():
    data = rows()
    pred = size_model.walk_forward(data, 6, START, SCORE, END, ["a", "b"])
    assert pred["close_ms"].min() == SCORE and pred["close_ms"].max() == END - BAR_MS and pred["fwd"].isna().sum() == 18
    scored = pred.dropna(subset=["fwd"])
    assert np.corrcoef(scored["size"], scored["fwd"])[0, 1] > 0.8
    truth = data.set_index(["symbol", "close_ms"]).loc[list(zip(pred["symbol"], pred["close_ms"]))]
    assert np.abs(pred["size"].to_numpy() - (0.02 * truth["a"] - 0.01 * truth["b"]).to_numpy()).mean() < 0.002


def test_nothing_after_a_month_start_changes_that_month_predictions_of_the_past():
    data = rows()
    cut = utc_ms("2021-08-01")
    later = data["close_ms"] + 6 * BAR_MS > cut                      # answers not yet known on the first of August
    changed = data.copy()
    changed.loc[later, "fwd"] = changed.loc[later, "fwd"] * -3 + 0.5
    changed.loc[data["close_ms"] >= utc_ms("2021-09-01"), ["a", "b"]] += 1.0
    a = size_model.walk_forward(data, 6, START, SCORE, END, ["a", "b"])
    b = size_model.walk_forward(changed, 6, START, SCORE, END, ["a", "b"])
    before = a["close_ms"] < utc_ms("2021-09-01")
    assert (a.loc[before, "size"].to_numpy() == b.loc[before, "size"].to_numpy()).all()
    assert not np.allclose(a.loc[~before, "size"], b.loc[~before, "size"])


def test_combine_takes_the_largest_gain_and_loss_with_their_periods():
    frame = lambda sizes: pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "close_ms": [0, 0, 0], "size": sizes})
    out = size_model.combine({2: frame([0.01, -0.02, 0.0]), 6: frame([0.03, -0.01, 0.0]), 18: frame([-0.02, 0.04, np.nan])})
    out = out.set_index("symbol")
    assert list(out.index) == ["AAA", "BBB"]                          # CCC lacks one period
    assert (out.at["AAA", "up"], out.at["AAA", "up_bars"], out.at["AAA", "down"], out.at["AAA", "down_bars"]) == (0.03, 6, -0.02, 18)
    assert (out.at["BBB", "up"], out.at["BBB", "up_bars"], out.at["BBB", "down"], out.at["BBB", "down_bars"]) == (0.04, 18, -0.02, 2)
    tie = size_model.combine({2: frame([0.01, 0.0, 0.0]), 6: frame([0.01, 0.0, 0.0])}).set_index("symbol")
    assert tie.at["AAA", "up_bars"] == 2 and tie.at["AAA", "down_bars"] == 2


def test_largest_ranks_by_turnover_in_the_last_year_only():
    end = utc_ms("2026-08-06")
    bar = lambda t, close, volume: [t, close, close, close, close, volume]
    old, new = end - 400 * 86_400_000, end - 10 * 86_400_000
    bars = {"OLD": np.array([bar(old, 100, 1e9), bar(new, 1, 1)], float), "BIG": np.array([bar(new, 10, 500)], float),
            "MID": np.array([bar(new, 10, 100)], float), "LATE": np.array([bar(end, 100, 1e9)], float)}
    assert size_model.largest(bars, end, 2) == ["BIG", "MID"]
