"""v15 shadow book (predictor/shadow.py): the same alphas as the research walk-forward, the book's arithmetic, no trading
when late, and the IC scored once the horizon has passed."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from predictor import shadow
from research import alpha
from research.h4_data import BAR_MS, SYMBOLS
from tests.test_predictor import FakeMarket

N = 1500


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(shadow, "TRAIN_START", 0)
    monkeypatch.setattr(shadow, "COUNT", 8)
    monkeypatch.setattr(alpha, "MIN_COINS", 5)


def market_at(close: int) -> FakeMarket:
    m = FakeMarket(close, n=N)
    for k, s in enumerate(SYMBOLS):                  # different funding per coin, so the carry signal has something to rank
        m.funding[s][:, 1] = 0.0001 * (k - 4)
    return m


def add_bar(market: FakeMarket, factor: dict, default: float = 1.0) -> None:
    for s in market.symbols:
        last = market.bars[s][-1].copy()
        last[0] += BAR_MS
        last[1:5] *= factor.get(s, default)
        market.bars[s] = np.vstack([market.bars[s], last])


def first_close() -> int:
    t = int(pd.Timestamp("2024-06-20").value // 1_000_000)
    return t // BAR_MS * BAR_MS


def test_shadow_alphas_are_the_research_walk_forward_alphas(tmp_path, small):
    close = first_close()
    market = market_at(close)
    book = shadow.ShadowBook(tmp_path)
    out = book.cycle(market, close + 60_000, clock=lambda: (close + 60_000) / 1000)
    assert out.startswith("shadow book rebalanced"), out
    live = pd.DataFrame(shadow.read_jsonl(tmp_path / "shadow_alpha.jsonl")).set_index("symbol")["alpha"]
    data, *_ = shadow.inputs(market, close)
    research = alpha.walk_forward(data, shadow.NAMES, shadow.HORIZON, 0, shadow.month_start(close), close + 1)
    research = research[research["close_ms"] == close].set_index("symbol")["alpha"]
    assert len(live) == 8 and np.allclose(live.sort_index(), research.sort_index())
    members = alpha.universe({s: market.bars[s] for s in SYMBOLS}, shadow.month_start(close), 8)
    assert set(live.index) == set(members)
    state = json.loads((tmp_path / "shadow_state.json").read_text())
    assert state["equity"] < shadow.START_EQUITY and state["last_close_ms"] == close     # it paid for its first trades
    row = shadow.read_jsonl(tmp_path / "shadow_book.jsonl")[0]
    assert math.isclose(state["equity"], shadow.START_EQUITY * (1 - row["traded"] * shadow.SIDE_COST))
    assert set(state["weights"]) <= set(members) and row["gross"] <= 1.0 + 1e-9 and abs(row["beta"]) < 0.05


def test_book_marks_moves_and_funding_then_pays_for_the_new_trades(tmp_path, small):
    close = first_close()
    market = market_at(close)
    book = shadow.ShadowBook(tmp_path)
    book.cycle(market, close + 60_000, clock=lambda: (close + 60_000) / 1000)
    before = json.loads((tmp_path / "shadow_state.json").read_text())
    w = before["weights"]
    eth_up = "ETHUSDT" if "ETHUSDT" in w else next(iter(w))
    add_bar(market, {eth_up: 1.05}, 1.02)
    market.funding[eth_up] = np.vstack([market.funding[eth_up], [close + BAR_MS, 0.001]])
    nxt = close + BAR_MS
    assert book.cycle(market, nxt + 60_000, clock=lambda: (nxt + 60_000) / 1000).startswith("shadow book rebalanced")
    row = shadow.read_jsonl(tmp_path / "shadow_book.jsonl")[-1]
    move = {s: (0.05 if s == eth_up else 0.02) for s in w}
    gain = sum(v * move[s] for s, v in w.items())
    paid = w[eth_up] * 0.001
    assert math.isclose(row["long"] + row["short"], gain) and math.isclose(row["funding"], -paid)
    universe_now = [s for s in alpha.universe({s: market.bars[s] for s in SYMBOLS}, shadow.month_start(nxt), 8)]
    assert math.isclose(row["market"], np.mean([0.05 if s == eth_up else 0.02 for s in universe_now]))
    expected = before["equity"] * (1 + gain - paid) * (1 - row["traded"] * shadow.SIDE_COST)
    assert math.isclose(row["equity"], expected) and math.isclose(json.loads((tmp_path / "shadow_state.json").read_text())["equity"], expected)
    assert book.cycle(market, nxt + 120_000, clock=lambda: (nxt + 120_000) / 1000) == "shadow book already at this close"
    assert len(shadow.read_jsonl(tmp_path / "shadow_book.jsonl")) == 2


def test_a_late_close_is_marked_but_not_traded(tmp_path, small):
    close = first_close()
    market = market_at(close)
    book = shadow.ShadowBook(tmp_path)
    book.cycle(market, close + 60_000, clock=lambda: (close + 60_000) / 1000)
    weights = json.loads((tmp_path / "shadow_state.json").read_text())["weights"]
    add_bar(market, {}, 1.01)
    nxt = close + BAR_MS
    out = book.cycle(market, nxt + 60_000, clock=lambda: (nxt + shadow.MAX_LATE_MS + 1) / 1000)
    assert "marked" in out and "late" in out
    row = shadow.read_jsonl(tmp_path / "shadow_book.jsonl")[-1]
    assert row["traded"] == 0 and not row["rebalanced"] and row["note"] == "late"
    after = json.loads((tmp_path / "shadow_state.json").read_text())["weights"]
    assert set(after) == set(weights)                 # the same coins, only drifted with the prices
    assert len(shadow.read_jsonl(tmp_path / "shadow_alpha.jsonl")) == 8


def test_alphas_are_scored_once_their_horizon_has_passed(tmp_path, small):
    close = first_close()
    market = market_at(close)
    book = shadow.ShadowBook(tmp_path)
    book.cycle(market, close + 60_000, clock=lambda: (close + 60_000) / 1000)
    rng = np.random.default_rng(5)
    for step in range(1, shadow.HORIZON + 1):
        add_bar(market, {s: float(np.exp(rng.normal(0, 0.01))) for s in SYMBOLS})
        t = close + step * BAR_MS
        book.cycle(market, t + 60_000, clock=lambda: (t + shadow.MAX_LATE_MS + 1) / 1000)      # late: marks only, keeps it fast
        if step < shadow.HORIZON:
            assert not (tmp_path / "shadow_ic.jsonl").exists()
    scored = shadow.read_jsonl(tmp_path / "shadow_ic.jsonl")
    assert [r["close_ms"] for r in scored] == [close] and scored[0]["coins"] == 8 and -1 <= scored[0]["ic"] <= 1
    data, *_ = shadow.inputs(market, close + shadow.HORIZON * BAR_MS)
    target = data[data["close_ms"] == close].set_index("symbol")["target"]
    told = pd.DataFrame(shadow.read_jsonl(tmp_path / "shadow_alpha.jsonl")).set_index("symbol")["alpha"]
    assert math.isclose(scored[0]["ic"], told.rank().corr(target[told.index].rank()))


def test_dashboard_summarises_the_shadow_book(tmp_path):
    from dashboard.server import prediction_summary
    folder = tmp_path / "runtime/predictor"
    folder.mkdir(parents=True)
    t0 = first_close()
    rows = [{"close_ms": t0 + i * BAR_MS, "equity": e, "funding": f, "cost": c, "traded": tr, "gross": 0.9, "net": 0.3, "beta": 0.0,
             "positions": 3, "rebalanced": True, "note": ""} for i, (e, f, c, tr) in
            enumerate([(999.0, 0.0, -0.001, 1.0), (1009.0, 0.001, 0.0, 0.0), (989.0, 0.0, -0.0002, 0.2)])]
    (folder / "shadow_book.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "shadow_state.json").write_text(json.dumps({"weights": {"AUSDT": 0.4, "BUSDT": -0.2, "CUSDT": 0.3}}))
    (folder / "shadow_ic.jsonl").write_text("".join(json.dumps({"close_ms": t0 + i * BAR_MS, "ic": v}) + "\n" for i, v in enumerate([0.1, 0.3])))
    s = prediction_summary(tmp_path, now_ms=t0)["shadow"]
    assert s["rows"] == 3 and math.isclose(s["total"], -0.011) and math.isclose(s["max_drawdown"], 1 - 989 / 1009)
    assert s["turnover_per_day"] is None and math.isclose(s["cost"], -0.0012)      # less than a day: no rate yet
    later = rows + [{**rows[-1], "close_ms": t0 + 12 * BAR_MS, "traded": 0.4}]
    (folder / "shadow_book.jsonl").write_text("".join(json.dumps(r) + "\n" for r in later))
    assert math.isclose(prediction_summary(tmp_path, now_ms=t0)["shadow"]["turnover_per_day"], 0.6 / 2)   # the first build left out
    assert [c["symbol"] for c in s["longs"]] == ["AUSDT", "CUSDT"] and [c["symbol"] for c in s["shorts"]] == ["BUSDT"]
    assert math.isclose(s["ic_mean"], 0.2) and s["ic_t"] is None
    assert prediction_summary(tmp_path / "none", now_ms=t0)["shadow"] == {"rows": 0}
