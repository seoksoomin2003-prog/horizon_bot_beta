"""The live direction predictor: on-time predictions only, scored one bar later, and no orders anywhere."""
import json
import math

import numpy as np
import pandas as pd

from dashboard.server import control, prediction_summary
from predictor import live
from research.h4_data import BAR_MS, SYMBOLS

T0 = 1_700_000_000_000 // BAR_MS * BAR_MS


class FakeMarket:
    """Random-walk bars for every coin, already complete up to `last_close`."""

    def __init__(self, last_close: int, n: int = 400):
        rng = np.random.default_rng(3)
        self.bars, self.funding = {}, {}
        opens = last_close - BAR_MS * np.arange(n, 0, -1)
        for s in SYMBOLS:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            self.bars[s] = np.column_stack([opens, close, close * 1.01, close * 0.99, close, rng.uniform(1, 2, n)])
            self.funding[s] = np.array([[t, 0.0001] for t in range(int(opens[0]), last_close + 1, 8 * 3_600_000)], dtype=float)
        self.updated = []

    def update(self, now_ms):
        self.updated.append(now_ms)

    frames = live.Market.frames


def run_cycle(tmp_path, market, now_ms, made_ms):
    return live.cycle(market, live.Model(), tmp_path, now_ms, clock=lambda: made_ms / 1000)


def test_predicts_every_coin_once_on_time(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    market = FakeMarket(close)
    assert run_cycle(tmp_path, market, close + 60_000, close + 90_000) == f"predicted {len(SYMBOLS)} coins"
    rows = live.read_jsonl(tmp_path / "predictions.jsonl")
    assert {r["symbol"] for r in rows} == set(SYMBOLS) and {r["close_ms"] for r in rows} == {close}
    assert all(0 < r["p_up"] < 1 and r["signal"] == (r["expected_gain"] > 0.005) for r in rows)
    btc = next(r for r in rows if r["symbol"] == "BTCUSDT")
    assert math.isclose(btc["price"], market.bars["BTCUSDT"][-1, 4])
    assert run_cycle(tmp_path, market, close + 120_000, close + 130_000) == "already predicted"


def test_a_late_close_is_skipped_not_predicted_after_the_fact(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    out = run_cycle(tmp_path, FakeMarket(close), close + 60_000, close + live.MAX_LATE_MS + 1)
    assert out.startswith("too late") and not (tmp_path / "predictions.jsonl").exists()


def test_outcome_is_the_next_bar_close(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    later = FakeMarket(close)
    run_cycle(tmp_path, later, close + 60_000, close + 60_000)
    for s in SYMBOLS:                                # one more bar closes, 2% above the last
        last = later.bars[s][-1].copy()
        last[0] += BAR_MS
        last[1:5] *= 1.02
        later.bars[s] = np.vstack([later.bars[s], last])
    run_cycle(tmp_path, later, close + BAR_MS + 60_000, close + BAR_MS + 60_000)
    outcomes = {o["symbol"]: o for o in live.read_jsonl(tmp_path / "outcomes.jsonl")}
    assert set(outcomes) == set(SYMBOLS) and all(o["close_ms"] == close for o in outcomes.values())
    b = later.bars["ETHUSDT"]
    assert math.isclose(outcomes["ETHUSDT"]["log_return"], math.log(b[-1, 4] / b[-2, 4]))


def test_model_trains_only_on_answers_known_before_the_month(monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    now = int(pd.Timestamp("2024-03-10").value // 1_000_000)
    month = int(pd.Timestamp("2024-03-01").value // 1_000_000)
    market = FakeMarket(now // BAR_MS * BAR_MS, n=600)
    data = live.with_closes(market.frames(), market.bars)
    model = live.Model()
    model.refit_if_needed(data, now)
    usable = data.dropna(subset=live.FEATURES + ["fwd", "sd42"])
    assert model.month == month and model.train_rows == int((usable["close_ms"] + BAR_MS <= month).sum())


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dashboard_scores_predictions(tmp_path):
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    write(folder / "predictions.jsonl", [
        {"close_ms": 0, "symbol": "BTCUSDT", "price": 100.0, "p_up": 0.6, "expected_gain": 0.006, "signal": True},
        {"close_ms": 0, "symbol": "ETHUSDT", "price": 10.0, "p_up": 0.45, "expected_gain": 0.001, "signal": False},
        {"close_ms": BAR_MS, "symbol": "BTCUSDT", "price": 101.0, "p_up": 0.55, "expected_gain": 0.002, "signal": False}])
    write(folder / "outcomes.jsonl", [
        {"close_ms": 0, "symbol": "BTCUSDT", "next_price": 101.0, "log_return": math.log(1.01)},
        {"close_ms": 0, "symbol": "ETHUSDT", "next_price": 10.5, "log_return": math.log(1.05)}])
    q = prediction_summary(tmp_path, now_ms=BAR_MS)
    assert (q["scored"], q["hits"], q["ups"], q["signals"], q["signal_hits"]) == (2, 1, 2, 1, 1)
    assert math.isclose(q["signal_net"], 0.01 - 0.002)
    assert [r["symbol"] for r in q["latest"]] == ["BTCUSDT"] and q["latest"][0]["hit"] is None
    assert not q["running"]


def test_predictor_buttons_only_run_the_predictor_script(tmp_path):
    calls = []
    runner = lambda root, *args, script="run_mac.sh": calls.append((script, args)) or (True, "")
    assert control(tmp_path, "predictor_start", runner=runner)[0] == 200
    (tmp_path / ".run").mkdir()
    (tmp_path / ".run/predictor.pid").write_text(str(__import__("os").getpid()))
    assert control(tmp_path, "predictor_stop", runner=runner)[0] == 200
    assert calls == [("run_predictor.sh", ("start",)), ("run_predictor.sh", ("stop",))]


def test_predictor_never_imports_the_bot_config():
    import subprocess
    import sys
    code = "import sys, predictor.live; print('config' in sys.modules or 'data_handler' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=live.ROOT)
    assert out.stdout.strip() == "False", out.stderr
