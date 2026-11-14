"""The live direction predictor: on-time predictions only, scored when the horizon ends, and no orders anywhere."""
import json
import math
import os

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
        self.bars, self.funding, self.symbols = {}, {}, list(SYMBOLS)
        opens = last_close - BAR_MS * np.arange(n, 0, -1)
        for s in SYMBOLS:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            self.bars[s] = np.column_stack([opens, close, close * 1.01, close * 0.99, close, rng.uniform(1, 2, n)])
            self.funding[s] = np.array([[t, 0.0001] for t in range(int(opens[0]), last_close + 1, 8 * 3_600_000)], dtype=float)
        self.updated = []

    def update(self, now_ms, bars=True):
        self.updated.append(now_ms)

    frames = live.Market.frames


def run_cycle(tmp_path, market, now_ms, made_ms):
    return live.cycle(market, [live.Model(h) for h in live.HORIZONS], tmp_path, now_ms, clock=lambda: made_ms / 1000)


def test_predicts_every_coin_once_on_time(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    market = FakeMarket(close)
    assert run_cycle(tmp_path, market, close + 60_000, close + 90_000) == f"predicted {len(SYMBOLS)} coins"
    rows = live.read_jsonl(tmp_path / "predictions.jsonl")
    assert sorted((r["symbol"], r["horizon"]) for r in rows) == sorted((s, h) for s in SYMBOLS for h in (1, 2))
    assert {r["close_ms"] for r in rows} == {close}
    assert all(0 < r["p_up"] < 1 and r["signal"] == (r["expected_gain"] > 0.005) for r in rows)
    assert all(isinstance(r["confident"], bool) and (r["confident"] or not r["top5"]) for r in rows)   # top 5% is inside top 10%
    btc = next(r for r in rows if r["symbol"] == "BTCUSDT")
    old = [{k: v for k, v in r.items() if k != "horizon"} for r in rows if r["horizon"] == 1]      # a file from before 8h predictions
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
    assert run_cycle(tmp_path, market, close + 100_000, close + 110_000) == f"predicted {len(SYMBOLS)} coins"
    assert sum(r.get("horizon") == 2 for r in live.read_jsonl(tmp_path / "predictions.jsonl")) == len(SYMBOLS)
    assert math.isclose(btc["price"], market.bars["BTCUSDT"][-1, 4])
    assert run_cycle(tmp_path, market, close + 120_000, close + 130_000) == "already predicted"


def test_a_late_close_is_skipped_not_predicted_after_the_fact(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    out = run_cycle(tmp_path, FakeMarket(close), close + 60_000, close + live.MAX_LATE_MS + 1)
    assert out.startswith("too late") and not (tmp_path / "predictions.jsonl").exists()


def test_outcome_is_the_close_at_the_end_of_the_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    later = FakeMarket(close)
    run_cycle(tmp_path, later, close + 60_000, close + 60_000)
    def one_more_bar():                              # closes 2% above the last
        for s in SYMBOLS:
            last = later.bars[s][-1].copy()
            last[0] += BAR_MS
            last[1:5] *= 1.02
            later.bars[s] = np.vstack([later.bars[s], last])

    one_more_bar()
    run_cycle(tmp_path, later, close + BAR_MS + 60_000, close + BAR_MS + 60_000)
    first = [o for o in live.read_jsonl(tmp_path / "outcomes.jsonl") if o["close_ms"] == close]
    assert {(o["symbol"], o["horizon"]) for o in first} == {(s, 1) for s in SYMBOLS}       # the 8h answer is not known yet
    b = later.bars["ETHUSDT"]
    eth = next(o for o in first if o["symbol"] == "ETHUSDT")
    assert math.isclose(eth["log_return"], math.log(b[-1, 4] / b[-2, 4]))
    one_more_bar()
    run_cycle(tmp_path, later, close + 2 * BAR_MS + 60_000, close + 2 * BAR_MS + 60_000)
    eth8 = next(o for o in live.read_jsonl(tmp_path / "outcomes.jsonl")
                if o["symbol"] == "ETHUSDT" and o["close_ms"] == close and o["horizon"] == 2)
    assert math.isclose(eth8["log_return"], 2 * math.log(1.02))


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
    assert 0 < model.thr <= model.thr95 < 0.5
    eight = live.Model(2)                            # an 8h answer is known one bar later, so one bar fewer per coin is usable
    data8 = live.with_closes(market.frames(2), market.bars)
    eight.refit_if_needed(data8, now)
    usable8 = data8.dropna(subset=live.FEATURES + ["fwd", "sd42"])
    assert eight.train_rows == int((usable8["close_ms"] + 2 * BAR_MS <= month).sum()) == model.train_rows - len(SYMBOLS)
    extra = data[data["symbol"] == "BTCUSDT"].assign(symbol="NEWCOIN")      # every coin it predicts is learned from (v7 H2)
    wider = live.Model()
    wider.refit_if_needed(pd.concat([data, extra], ignore_index=True), now)
    assert wider.train_rows == model.train_rows + int((extra.dropna(subset=live.FEATURES + ["fwd", "sd42"])["close_ms"] + BAR_MS <= month).sum())


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dashboard_scores_predictions(tmp_path):
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    write(folder / "predictions.jsonl", [
        {"close_ms": 0, "symbol": "BTCUSDT", "price": 100.0, "p_up": 0.6, "expected_gain": 0.006, "signal": True, "confident": True},
        {"close_ms": 0, "symbol": "ETHUSDT", "price": 10.0, "p_up": 0.45, "expected_gain": 0.001, "signal": False},
        {"close_ms": BAR_MS, "symbol": "BTCUSDT", "price": 101.0, "p_up": 0.55, "expected_gain": 0.002, "signal": False},
        {"close_ms": 0, "horizon": 2, "symbol": "BTCUSDT", "price": 100.0, "p_up": 0.3, "confident": True, "top5": True},
        {"close_ms": 0, "horizon": 2, "symbol": "ETHUSDT", "price": 10.0, "p_up": 0.6, "confident": True, "top5": False},
        {"close_ms": BAR_MS, "horizon": 2, "symbol": "BTCUSDT", "price": 101.0, "p_up": 0.7, "confident": True, "top5": True}])
    write(folder / "outcomes.jsonl", [
        {"close_ms": 0, "symbol": "BTCUSDT", "next_price": 101.0, "log_return": math.log(1.01)},
        {"close_ms": 0, "symbol": "ETHUSDT", "next_price": 10.5, "log_return": math.log(1.05)},
        {"close_ms": 0, "horizon": 2, "symbol": "BTCUSDT", "next_price": 99.0, "log_return": math.log(0.99)},
        {"close_ms": 0, "horizon": 2, "symbol": "ETHUSDT", "next_price": 9.0, "log_return": math.log(0.9)}])
    q = prediction_summary(tmp_path, now_ms=BAR_MS)
    assert (q["scored"], q["hits"], q["ups"], q["signals"], q["signal_hits"]) == (2, 1, 2, 1, 1)
    assert math.isclose(q["signal_net"], 0.01 - 0.002)
    assert (q["confident"], q["confident_hits"]) == (1, 1) and math.isclose(q["confident_net"], 0.01 - 0.002)
    assert [r["symbol"] for r in q["latest"]] == ["BTCUSDT"] and q["latest"][0]["hit"] is None
    assert q["latest"][0]["p_up_8h"] == 0.7              # 8h rows are scored apart and never mixed into the 4h numbers
    assert q["eight"] == {"scored": 2, "hits": 1, "confident": 2, "confident_hits": 1, "top5": 1, "top5_hits": 1}
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


def test_collector_saves_each_completed_hour_once(tmp_path):
    from predictor.collect import HOUR_MS, Collector, last_times
    now = 1_000 * HOUR_MS

    class Client:
        calls = 0

        def _rows(self, q, **fields):
            Client.calls += 1
            return [{"timestamp": t, **fields} for t in range(max(q["startTime"] // HOUR_MS + 1, 990) * HOUR_MS, now + 1, HOUR_MS)
                    if q["startTime"] <= t <= q["endTime"]][-500:]

        def fapiDataGetOpenInterestHist(self, q):
            return self._rows(q, sumOpenInterest="5", sumOpenInterestValue="500")

        def fapiDataGetGlobalLongShortAccountRatio(self, q):
            return self._rows(q, longAccount="0.6")

        def fapiDataGetTopLongShortPositionRatio(self, q):
            return self._rows(q, longAccount="0.7")

    c = Collector(tmp_path, Client(), ["BTCUSDT"], pause=0)
    assert c.positioning(now) == 10          # hours 990..999; the hour still forming (1000) is left out
    assert c.positioning(now) == 0
    rows = live.read_jsonl(tmp_path / "positioning.jsonl")
    assert rows[-1] == {"ts": 999 * HOUR_MS, "symbol": "BTCUSDT", "oi": 5.0, "oi_value": 500.0, "long_accounts": 0.6,
                        "top_long_positions": 0.7}
    assert last_times(tmp_path / "positioning.jsonl") == {"BTCUSDT": 999 * HOUR_MS}
    assert Collector(tmp_path, Client(), ["BTCUSDT"], pause=0).positioning(now + HOUR_MS) == 1     # a restart continues


def test_order_book_imbalance_and_liquidation_rows():
    from predictor.collect import book_row, liquidation_row
    row = book_row("X", 1, [["100", "3"], ["99", "1"]], [["101", "1"], ["102", "1"]])
    assert math.isclose(row["mid"], 100.5) and math.isclose(row["spread"], 1 / 100.5)
    assert math.isclose(row["imbalance_20"], (399 - 203) / (399 + 203))
    assert book_row("X", 1, [], [["1", "1"]]) is None
    message = {"e": "forceOrder", "o": {"s": "BTCUSDT", "S": "SELL", "ap": "50000", "z": "0.2", "T": 123}}
    assert liquidation_row(message, {"BTCUSDT"}) == {"ts": 123, "symbol": "BTCUSDT", "side": "SELL", "value": 10000.0}
    assert liquidation_row({"stream": "!forceOrder@arr", "data": message}, {"BTCUSDT"})["value"] == 10000.0
    assert liquidation_row(message, {"ETHUSDT"}) is None and liquidation_row({"o": {"s": "BTCUSDT"}}, {"BTCUSDT"}) is None


def test_relative_prediction_is_scored_against_the_median_coin(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "DEV_START_MS", 0)
    close = T0 + 400 * BAR_MS
    market = FakeMarket(close)
    market.symbols.append("BTCDOMUSDT")              # the index product is never part of the relative prediction
    market.bars["BTCDOMUSDT"], market.funding["BTCDOMUSDT"] = market.bars["ETHUSDT"] * [1, 3, 3, 3, 3, 1], market.funding["ETHUSDT"]
    models = [live.Model(2, relative=True)]
    cycle = lambda now: live.cycle(market, models, tmp_path, now, clock=lambda: now / 1000)
    assert cycle(close + 60_000) == f"predicted {len(SYMBOLS)} coins"
    rows = live.read_jsonl(tmp_path / "predictions.jsonl")
    assert {r["symbol"] for r in rows} == set(SYMBOLS) and {(r["kind"], r["horizon"]) for r in rows} == {("relative", 2)}
    for step, eth in ((1, 1.02), (2, 1.05)):         # two more bars: every coin +2% each, ETH +5% on the second
        for s in market.symbols:
            last = market.bars[s][-1].copy()
            last[0] += BAR_MS
            last[1:5] *= eth if s == "ETHUSDT" else 1.02
            market.bars[s] = np.vstack([market.bars[s], last])
        cycle(close + step * BAR_MS + 60_000)
    outcomes = {o["symbol"]: o for o in live.read_jsonl(tmp_path / "outcomes.jsonl") if o["close_ms"] == close}
    assert set(outcomes) == set(SYMBOLS) and all(o["kind"] == "relative" for o in outcomes.values())
    assert math.isclose(outcomes["ETHUSDT"]["relative_return"], math.log(1.05) - math.log(1.02))     # the median coin made 2 x 2%
    assert math.isclose(outcomes["BTCUSDT"]["relative_return"], 0.0, abs_tol=1e-12)
    assert math.isclose(outcomes["BTCUSDT"]["log_return"], 2 * math.log(1.02))


def test_dashboard_scores_relative_predictions_apart(tmp_path):
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    base = {"close_ms": 0, "horizon": 2, "kind": "relative", "price": 1.0}
    write(folder / "predictions.jsonl", [
        {"close_ms": 0, "symbol": "BTCUSDT", "price": 100.0, "p_up": 0.6, "expected_gain": 0.001, "signal": False},
        {**base, "symbol": "BTCUSDT", "p_up": 0.7, "confident": True, "top5": True},
        {**base, "symbol": "ETHUSDT", "p_up": 0.4, "confident": True, "top5": False},
        {**base, "symbol": "SOLUSDT", "p_up": 0.55, "confident": False, "top5": False, "horizon": 1}])
    write(folder / "outcomes.jsonl", [
        {**base, "symbol": "BTCUSDT", "log_return": -0.01, "relative_return": 0.02},     # fell, but less than the median coin: a hit
        {**base, "symbol": "ETHUSDT", "log_return": -0.03, "relative_return": 0.01},
        {**base, "symbol": "SOLUSDT", "horizon": 1, "log_return": 0.01, "relative_return": 0.001}])
    q = prediction_summary(tmp_path, now_ms=BAR_MS)
    assert q["relative"]["8h"] == {"scored": 2, "hits": 1, "confident": 2, "confident_hits": 1, "top5": 1, "top5_hits": 1}
    assert q["relative"]["4h"]["scored"] == 1 and q["relative"]["4h"]["hits"] == 1
    assert q["scored"] == 0 and q["eight"]["scored"] == 0 and q["latest"][0]["p_above_8h"] == 0.7


def test_dashboard_lists_every_coin_with_its_confidence_rank(tmp_path):
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    coin = lambda symbol, p4, p8, **flags8: [
        {"close_ms": BAR_MS, "symbol": symbol, "price": 1.0, "p_up": p4, "expected_gain": 0.0, "signal": False, "confident": p4 == 0.4},
        {"close_ms": BAR_MS, "horizon": 2, "symbol": symbol, "price": 1.0, "p_up": p8, **flags8},
        {"close_ms": BAR_MS, "horizon": 2, "kind": "relative", "symbol": symbol, "price": 1.0, "p_up": 0.99}]
    write(folder / "predictions.jsonl", [{"close_ms": 0, "symbol": "OLDUSDT", "price": 1.0, "p_up": 0.9, "expected_gain": 0.0, "signal": False}]
          + coin("CCCUSDT", 0.52, 0.30, confident=True, top5=True) + coin("AAAUSDT", 0.40, 0.55, confident=True) + coin("BBBUSDT", 0.51, 0.49))
    write(folder / "outcomes.jsonl", [])
    coins = prediction_summary(tmp_path, now_ms=BAR_MS)["coins"]
    assert [c["symbol"] for c in coins] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]        # the latest close only, by name
    a, b, c = coins
    assert (a["p_up"], a["rank"], a["confident"], a["p_up_8h"], a["rank_8h"], a["confident_8h"], a["top5_8h"]) == (0.40, 1, True, 0.55, 2, True, False)
    assert (b["rank"], b["rank_8h"], b["confident"], b["confident_8h"]) == (3, 3, False, False)
    assert (c["rank"], c["rank_8h"], c["top5_8h"]) == (2, 1, True)                  # the relative rows never count here


CANDLE_MS = live.NOW_CANDLE_MS


class CandleClient:
    """15-minute random-walk candles for every coin, newest one still forming; answers the paged klines query."""

    def __init__(self, newest_open: int, n: int = 3200, skip: tuple = ()):
        rng = np.random.default_rng(5)
        self.calls, self.candles = [], {}
        opens = newest_open - CANDLE_MS * np.arange(n - 1, -1, -1)
        for s in SYMBOLS:
            if s in skip:
                continue
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
            self.candles[s] = np.column_stack([opens, close, close * 1.001, close * 0.999, close, rng.uniform(1, 2, n)])

    def fapiPublicGetKlines(self, query):
        self.calls.append(query)
        rows = self.candles.get(query["symbol"], np.empty((0, 6)))
        rows = rows[rows[:, 0] <= query.get("endTime", math.inf)][-query["limit"]:]
        return [[str(int(r[0]))] + [str(v) for v in r[1:]] for r in rows]


def fitted_models(market, now_ms):
    models = [live.Model(h, relative) for relative in (False, True) for h in live.HORIZONS]
    for m in models:
        m.refit_if_needed(m.prepare(market.frames(m.horizon)), now_ms)
    return models


def test_bars_ending_now_count_back_from_the_newest_candle():
    t = T0 + 5 * CANDLE_MS + CANDLE_MS * np.arange(16 * 3 + 5)       # 3 whole bars and 5 older candles
    candles = np.column_stack([t, np.arange(len(t)), np.arange(len(t)) + 0.5, np.arange(len(t)) - 0.5, np.arange(len(t)) + 0.1, np.ones(len(t))])
    bars = live.bars_ending_now(candles)
    assert len(bars) == 3 and (np.diff(bars[:, 0]) == BAR_MS).all()
    assert bars[-1, 0] + BAR_MS == t[-1] + CANDLE_MS                 # the last bar ends when the forming candle will
    assert list(bars[-1, 1:]) == [candles[-16, 1], candles[-1, 2], candles[-16, 3], candles[-1, 4], 16.0]
    holed = np.delete(candles, np.s_[-20:-18], axis=0)               # two candles missing in the middle bar
    assert len(live.bars_ending_now(holed)) == 3 and live.bars_ending_now(holed)[1, 5] == 14.0


def test_as_of_now_prediction_ranks_every_coin_on_bars_that_end_now(tmp_path):
    now = T0 + 7 * CANDLE_MS + 123_000                               # in the middle of a 4h bar and of a 15m candle
    market = FakeMarket(T0, n=1200)
    models = fitted_models(market, now)
    market.client = CandleClient(now // CANDLE_MS * CANDLE_MS, skip=("ETHUSDT",))
    result = live.predict_now(market, models, now, clock=lambda: now / 1000 + 20)
    coins = {c["symbol"]: c for c in result["coins"]}
    assert result["ok"] and result["made_ms"] == now + 20_000
    assert result["end_ms"] == now // CANDLE_MS * CANDLE_MS + CANDLE_MS and result["end_ms"] % BAR_MS != 0
    assert set(coins) == set(SYMBOLS) - {"ETHUSDT"}                  # a coin without candles is left out, not guessed
    assert sorted(c["rank"] for c in coins.values()) == list(range(1, len(coins) + 1))
    assert sorted(c["rank_8h"] for c in coins.values()) == list(range(1, len(coins) + 1))
    first = next(c for c in coins.values() if c["rank"] == 1)
    assert all(abs(first["p_up"] - 0.5) >= abs(c["p_up"] - 0.5) for c in coins.values())
    btc = coins["BTCUSDT"]
    assert btc["price"] == market.client.candles["BTCUSDT"][-1, 4]   # the forming candle's price
    assert 0 < btc["p_up"] < 1 and 0 < btc["p_up_8h"] < 1 and isinstance(btc["confident"], bool)
    assert all(q["interval"] == "15m" for q in market.client.calls) and market.updated == [now]


def test_as_of_now_request_is_answered_once_and_never_scored(tmp_path):
    now = T0 + 7 * CANDLE_MS
    market = FakeMarket(T0, n=1200)
    models = fitted_models(market, now)
    market.client = CandleClient(now // CANDLE_MS * CANDLE_MS)
    clock = lambda: now / 1000
    assert not live.serve_now(market, models, tmp_path, clock)       # nothing was asked
    write(tmp_path / "now_request.json", [{"requested_ms": now - 1000}])
    assert live.serve_now(market, models, tmp_path, clock)
    answer = json.loads((tmp_path / "now.json").read_text())
    assert answer["ok"] and answer["requested_ms"] == now - 1000 and len(answer["coins"]) == len(SYMBOLS)
    assert not live.serve_now(market, models, tmp_path, clock)       # the same request is not answered twice
    write(tmp_path / "now_request.json", [{"requested_ms": now}])
    assert not live.serve_now(market, models, tmp_path, clock)       # a new one within a minute waits
    assert live.serve_now(market, models, tmp_path, lambda: now / 1000 + 61)
    assert len(live.read_jsonl(tmp_path / "now_predictions.jsonl")) == 2 * len(SYMBOLS)
    assert not (tmp_path / "predictions.jsonl").exists()             # kept apart from the scored record
    market.client = None                                             # the exchange cannot be reached
    write(tmp_path / "now_request.json", [{"requested_ms": now + 200_000}])
    assert live.serve_now(market, models, tmp_path, lambda: now / 1000 + 300)
    assert json.loads((tmp_path / "now.json").read_text())["ok"] is False


def test_dashboard_asks_for_an_as_of_now_prediction_only_while_the_predictor_runs(tmp_path):
    from dashboard.server import now_view
    status, out = control(tmp_path, "predict_now", runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("runs no script")))
    assert status == 409 and not out["ok"] and not (tmp_path / "runtime/predictor/now_request.json").exists()
    (tmp_path / ".run").mkdir()
    (tmp_path / ".run/predictor.pid").write_text(str(os.getpid()))
    status, out = control(tmp_path, "predict_now", runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("runs no script")))
    folder = tmp_path / "runtime/predictor"
    asked = json.loads((folder / "now_request.json").read_text())["requested_ms"]
    assert status == 200 and out["ok"] and now_view(folder)["pending"]
    (folder / "now.json").write_text(json.dumps({"requested_ms": asked, "ok": True, "made_ms": asked + 5, "coins": [{"symbol": "BTCUSDT", "p_up": 0.6}]}))
    view = now_view(folder)
    assert not view["pending"] and view["ok"] and view["coins"][0]["symbol"] == "BTCUSDT"


def test_relative_model_ranks_the_coins_and_stamps_its_version(tmp_path):
    from research.direction import RANK_FEATURES
    market = FakeMarket(T0)
    relative, direction = live.Model(2, relative=True), live.Model(2)
    assert set(RANK_FEATURES) <= set(relative.features) and not set(RANK_FEATURES) & set(direction.features)
    data = relative.prepare(market.frames(2))
    last = data[data["close_ms"] == T0]
    assert np.allclose(sorted(last["r1_rank"]), np.linspace(-0.5, 0.5, len(SYMBOLS)))     # one coin at every step from lowest to highest
    live.cycle(market, [relative, direction], tmp_path, T0 + 60_000, clock=lambda: (T0 + 60_000) / 1000)
    rows = live.read_jsonl(tmp_path / "predictions.jsonl")
    assert {(r["kind"], r["version"]) for r in rows} == {("relative", 11), ("direction", 7)}


def test_dashboard_scores_only_the_newest_relative_model(tmp_path):
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    row = lambda t, symbol, **more: {"close_ms": t, "horizon": 2, "kind": "relative", "symbol": symbol, "price": 1.0, "p_up": 0.6, **more}
    write(folder / "predictions.jsonl", [row(0, "BTCUSDT"), row(0, "ETHUSDT", version=9), row(BAR_MS, "BTCUSDT", version=11, confident=True)])
    write(folder / "outcomes.jsonl", [{**row(0, "BTCUSDT"), "log_return": 0.0, "relative_return": 0.01},
                                      {**row(0, "ETHUSDT"), "log_return": 0.0, "relative_return": 0.01},
                                      {**row(BAR_MS, "BTCUSDT"), "log_return": 0.0, "relative_return": -0.01}])
    q = prediction_summary(tmp_path, now_ms=4 * BAR_MS)
    assert q["relative_version"] == 11
    assert q["relative"]["8h"] == {"scored": 1, "hits": 0, "confident": 1, "confident_hits": 0, "top5": 0, "top5_hits": 0}
