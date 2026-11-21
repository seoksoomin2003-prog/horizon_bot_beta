"""v18 live top-coin book (predictor/topcoin.py): the research alphas, buying the top S, sizing, costs, funding,
the sell and switch rules, staying flat, a late close, and the dashboard summary."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from predictor import shadow, topcoin
from research import alpha
from research.h4_data import BAR_MS, SYMBOLS
from tests.test_shadow import add_bar, first_close, market_at


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(shadow, "TRAIN_START", 0)
    monkeypatch.setattr(shadow, "COUNT", 8)
    monkeypatch.setattr(alpha, "MIN_COINS", 5)


class FakeModel:
    """Alphas chosen by the test; fitting only marks the month."""
    columns = [f"z_{n}" for n in shadow.NAMES]

    def __init__(self, alphas: dict) -> None:
        self.alphas, self.month = alphas, None

    def fit(self, data, month_ms):
        self.month = month_ms

    def predict(self, rows):
        return np.array([self.alphas.get(s, -0.01) for s in rows["symbol"]])


def run(book, market, close, late=False):
    made = close + (shadow.MAX_LATE_MS + 1 if late else 60_000)
    return book.cycle(market, close + 60_000, clock=lambda: made / 1000)


def members(market, close):
    return alpha.universe({s: market.bars[s] for s in SYMBOLS}, shadow.month_start(close), 8)


def scores(tmp_path, close):
    return {r["symbol"]: r for r in shadow.read_jsonl(tmp_path / "topcoin_scores.jsonl") if r["close_ms"] == close}


def test_scores_use_the_research_walk_forward_alphas(tmp_path, small):
    close = first_close()
    market = market_at(close)
    book = topcoin.TopCoinBook(tmp_path)
    run(book, market, close)
    got = scores(tmp_path, close)
    data, *_ = shadow.inputs(market, close)
    research = alpha.walk_forward(data, shadow.NAMES, shadow.HORIZON, 0, shadow.month_start(close), close + 1)
    research = research[research["close_ms"] == close].set_index("symbol")["alpha"]
    assert set(got) == set(members(market, close)) and len(got) == 8
    assert np.allclose([got[s]["alpha"] for s in research.index], research.to_numpy())
    for r in got.values():
        assert math.isclose(r["score"], (r["alpha"] - topcoin.ROUND_TRIP) / r["sigma"])
    assert [r["rank"] for r in sorted(got.values(), key=lambda r: -r["score"])] == list(range(1, 9))


def test_buys_the_top_score_sized_by_volatility_then_marks_funding_and_sells_when_alpha_turns_negative(tmp_path, small):
    close = first_close()
    market = market_at(close)
    coins = members(market, close)
    a, b = coins[0], coins[1]
    model = FakeModel({a: 0.02, b: 0.01})
    book = topcoin.TopCoinBook(tmp_path)
    book.model = model
    assert "buy " + a in run(book, market, close)
    row = scores(tmp_path, close)[a]
    assert row["status"] == "held" and scores(tmp_path, close)[b]["status"] == "positive"
    w = min(1.0, topcoin.TARGET_BAR_VOL / row["sigma"])
    price = float(market.bars[a][-1, 4])
    state = json.loads((tmp_path / "topcoin_state.json").read_text())
    assert math.isclose(state["equity"], 1000 * (1 - w * topcoin.SIDE_COST)) and state["position"]["symbol"] == a
    assert math.isclose(state["position"]["q"], 1000 * w / price)

    add_bar(market, {a: 1.05}, 1.0)
    market.funding[a] = np.vstack([market.funding[a], [close + BAR_MS, 0.001]])
    nxt = close + BAR_MS
    model.alphas[a] = -0.001
    out = run(book, market, nxt)
    assert "sell " + a in out
    q = 1000 * w / price
    expected = 1000 * (1 - w * topcoin.SIDE_COST) - 0.001 * q * price + q * price * 0.05 - q * price * 1.05 * topcoin.SIDE_COST
    trade = [t for t in shadow.read_jsonl(tmp_path / "topcoin_trades.jsonl") if t["action"] == "sell"][0]
    assert trade["why"] == "signal" and math.isclose(trade["equity"], expected) and math.isclose(trade["net"], expected / 1000 - 1)
    book_row = shadow.read_jsonl(tmp_path / "topcoin_book.jsonl")[-1]
    assert math.isclose(book_row["exposure_before"], q * price / (1000 * (1 - w * topcoin.SIDE_COST)))
    assert math.isclose(book_row["funding"], -0.001 * q * price)
    # it sold a and bought b, the new top, at the same close
    assert json.loads((tmp_path / "topcoin_state.json").read_text())["position"]["symbol"] == b
    assert run(book, market, nxt) == "top-coin book already at this close"


def test_switches_only_when_the_alpha_gap_beats_twice_the_round_trip(tmp_path, small):
    close = first_close()
    market = market_at(close)
    coins = members(market, close)
    a, b = coins[0], coins[1]
    model = FakeModel({a: 0.02, b: 0.001})
    book = topcoin.TopCoinBook(tmp_path)
    book.model = model
    run(book, market, close)
    for step, (gap, switched) in enumerate([(topcoin.SWITCH_GAP * 0.9, False), (topcoin.SWITCH_GAP * 1.1, True)], 1):
        add_bar(market, {}, 1.0)
        t = close + step * BAR_MS
        model.alphas[b] = 0.02 + gap
        run(book, market, t)
        held = json.loads((tmp_path / "topcoin_state.json").read_text())["position"]["symbol"]
        assert held == (b if switched else a)
        if not switched:
            assert scores(tmp_path, t)[b]["status"] == "top" and scores(tmp_path, t)[b]["switch_gap"] < 0
    assert [t["why"] for t in shadow.read_jsonl(tmp_path / "topcoin_trades.jsonl") if t["action"] == "sell"] == ["switch"]


def test_stays_flat_when_no_coin_beats_the_cost_and_a_late_close_only_marks(tmp_path, small):
    close = first_close()
    market = market_at(close)
    model = FakeModel({s: topcoin.ROUND_TRIP * 0.5 for s in SYMBOLS})
    book = topcoin.TopCoinBook(tmp_path)
    book.model = model
    assert run(book, market, close) == "top-coin book: equity 1000.00, flat"
    assert {r["status"] for r in scores(tmp_path, close).values()} == {"wait"}
    model.alphas[members(market, close)[0]] = 0.05
    add_bar(market, {}, 1.0)
    nxt = close + BAR_MS
    out = run(book, market, nxt, late=True)
    assert out.endswith("flat (late)") and not (tmp_path / "topcoin_trades.jsonl").exists()
    assert shadow.read_jsonl(tmp_path / "topcoin_book.jsonl")[-1]["note"] == "late" and not scores(tmp_path, nxt)


def test_dashboard_summarises_the_top_coin_book(tmp_path):
    from dashboard.server import prediction_summary
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    t0 = first_close()
    rows = [{"close_ms": t0 + i * BAR_MS, "equity": e, "funding": 0.0, "exposure_before": x, "market": m, "bench_ret": x * (m or 0.0),
             "symbol": "BNBUSDT", "exposure": 0.5, "btc": btc, "trades": 0, "note": ""}
            for i, (e, x, m, btc) in enumerate([(999.5, 0.0, None, 100.0), (1019.5, 0.5, 0.04, 102.0), (1009.5, 0.5, -0.02, 101.0)])]
    (folder / "topcoin_book.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "topcoin_state.json").write_text(json.dumps({"position": {"symbol": "BNBUSDT", "entry": 600.0, "last": 612.0,
                                                                        "entry_ms": t0, "q": 0.8}, "equity": 1009.5}))
    (folder / "topcoin_trades.jsonl").write_text(json.dumps({"close_ms": t0, "action": "buy", "symbol": "BNBUSDT", "price": 600.0,
                                                             "size": 0.5, "equity": 999.5}) + "\n")
    (folder / "topcoin_scores.jsonl").write_text("".join(json.dumps({"close_ms": t0 + 2 * BAR_MS, "symbol": s, "rank": k, "alpha": al,
                                                                     "sigma": 0.02, "score": (al - 0.002) / 0.02, "size": 0.53,
                                                                     "status": st, "switch_gap": g}) + "\n"
                                                         for k, (s, al, st, g) in enumerate([("BNBUSDT", 0.01, "held", None),
                                                                                             ("TRXUSDT", 0.008, "positive", -0.006),
                                                                                             ("SOLUSDT", 0.001, "wait", -0.013)], 1)))
    s = prediction_summary(tmp_path, now_ms=t0)["topcoin"]
    assert s["rows"] == 3 and math.isclose(s["total"], 0.0095) and math.isclose(s["max_drawdown"], 1 - 1009.5 / 1019.5)
    assert math.isclose(s["benchmark"], 1.02 * 0.99 - 1) and math.isclose(s["btc"], 0.01)
    assert s["position"]["symbol"] == "BNBUSDT" and math.isclose(s["position"]["gain"], 0.02)
    assert [c["symbol"] for c in s["scores"]] == ["BNBUSDT", "TRXUSDT", "SOLUSDT"] and s["trades"] == 0 and len(s["recent"]) == 1
