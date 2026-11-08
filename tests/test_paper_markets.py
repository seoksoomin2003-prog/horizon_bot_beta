"""Paper settings from the screen: spot or futures, leverage, liquidation and exchange minimum order sizes."""
import json

import ccxt
import pytest

import config
from paper import settings as paper_settings
from paper.exchange import MINUTE_MS, PaperBroker, fit_order
from tests.test_paper_mode import FakePublic, bar


def make(tmp_path, **kw):
    args = dict(initial=100.0, fee=0.0, slippage_bps=0.0, leverage=3.0, match_interval_s=0.0)
    args.update(kw)
    b = PaperBroker(str(tmp_path / "broker.json"), **args)
    b.public = FakePublic(last=100.0, bid=100.0, ask=100.0)
    return b


SYM = "BTC/USDT:USDT"


def test_settings_are_validated():
    ok = paper_settings.validate({"market": "futures", "leverage": 10, "initial": 50, "min_order": True})
    assert ok == {"market": "futures", "leverage": 10, "initial": 50.0, "min_order": True}
    assert paper_settings.validate({"market": "spot", "leverage": 10, "initial": 50, "min_order": False})["leverage"] == 1
    for bad in ({"market": "options", "leverage": 1, "initial": 50, "min_order": True},
                {"market": "futures", "leverage": 21, "initial": 50, "min_order": True},
                {"market": "futures", "leverage": 0, "initial": 50, "min_order": True},
                {"market": "futures", "leverage": 2.5, "initial": 50, "min_order": True},
                {"market": "futures", "leverage": "x", "initial": 50, "min_order": True},
                {"market": "futures", "leverage": 3, "initial": 1, "min_order": True},
                {"market": "futures", "leverage": 3, "initial": 50, "min_order": "yes"},
                None):
        with pytest.raises(ValueError):
            paper_settings.validate(bad)


def test_saved_settings_reach_the_paper_config_but_never_live(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_RUNTIME_DIR", str(tmp_path))
    paper_settings.save(tmp_path, {"market": "spot", "leverage": 5, "initial": 70, "min_order": False})
    got = config._paper_settings()
    assert (got["market"], got["leverage"], got["initial"], got["min_order"], got["saved"]) == ("spot", 1, 70.0, False, True)
    monkeypatch.setattr(config, "IS_PAPER", False)
    assert config._paper_settings()["saved"] is False and config._paper_settings()["market"] == "futures"


def test_broken_settings_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_RUNTIME_DIR", str(tmp_path))
    paper_settings.path(tmp_path).write_text('{"market": "futures", "leverage": 99')
    assert config._paper_settings()["saved"] is False


def test_new_account_records_its_settings(tmp_path):
    b = make(tmp_path, market="spot", fee=0.001, min_order=True)
    assert b.state["settings"] == {"market": "spot", "leverage": 1.0, "fee": 0.001, "min_order": True}


def test_spot_refuses_short_selling_but_closes_a_long(tmp_path):
    b = make(tmp_path, market="spot")
    with pytest.raises(ccxt.InvalidOrder):
        b.create_order(SYM, "market", "sell", 0.1)
    b.create_order(SYM, "market", "buy", 0.1)
    b.create_order(SYM, "market", "sell", 0.1)
    assert b.state["positions"] == {}


def test_spot_has_no_leverage(tmp_path):
    b = make(tmp_path, market="spot", leverage=10.0)
    with pytest.raises(ccxt.InsufficientFunds):
        b.create_order(SYM, "market", "buy", 1.5)   # 150 USDT with 100 cash
    b.create_order(SYM, "market", "buy", 0.9)


def test_higher_leverage_allows_a_larger_position(tmp_path):
    b = make(tmp_path, leverage=10.0)
    b.create_order(SYM, "market", "buy", 9.0)   # 900 USDT on 100 at 10x
    assert b.state["positions"][SYM]["amount"] == 9.0


def test_liquidation_when_a_bar_opens_past_the_level(tmp_path):
    b = make(tmp_path, leverage=10.0)
    b.create_order(SYM, "market", "buy", 5.0)
    b.create_order(SYM, "STOP_MARKET", "sell", 5.0, params={"stopPrice": 95.0, "reduceOnly": True})
    assert b.liquidation_price(b.state["positions"][SYM]) == pytest.approx(90.5)
    b.public.bars = [bar(1, 89.0, 89.5, 88.0, 89.0)]
    b.match(force=True)
    assert b.state["positions"] == {} and b.state["orders"] == {}
    last = b.state["trades"][-1]
    assert last["reason"] == "liquidation" and last["price"] == pytest.approx(90.0)
    assert b.state["cash"] == pytest.approx(100.0 - 50.0)   # the whole margin of 500 / 10x


def test_stop_in_front_of_the_liquidation_level_closes_first(tmp_path):
    b = make(tmp_path, leverage=10.0)
    b.create_order(SYM, "market", "buy", 5.0)
    b.create_order(SYM, "STOP_MARKET", "sell", 5.0, params={"stopPrice": 95.0, "reduceOnly": True})
    b.public.bars = [bar(1, 99.0, 99.0, 85.0, 86.0)]
    b.match(force=True)
    assert b.state["trades"][-1]["reason"] == "stop" and b.state["trades"][-1]["price"] == pytest.approx(95.0)


def test_unprotected_position_is_liquidated_when_crossed(tmp_path):
    b = make(tmp_path, leverage=5.0)
    b.create_order(SYM, "market", "sell", 2.0)
    b.public.bars = [bar(1, 101.0, 125.0, 100.0, 120.0)]
    b.match(force=True)
    t = b.state["trades"][-1]
    assert t["reason"] == "liquidation" and t["price"] == pytest.approx(120.0) and b.state["positions"] == {}


def test_spot_is_never_liquidated(tmp_path):
    b = make(tmp_path, market="spot")
    b.create_order(SYM, "market", "buy", 0.5)
    b.public.bars = [bar(1, 10.0, 10.0, 5.0, 6.0)]
    b.match(force=True)
    assert SYM in b.state["positions"]


def test_old_account_without_a_bar_after_entry_is_left_alone(tmp_path):
    b = make(tmp_path, leverage=10.0)
    b.create_order(SYM, "market", "buy", 5.0)
    b.public.bars = [bar(0, 50.0, 50.0, 50.0, 50.0)]   # the bar holding the entry
    b.match(force=True)
    assert SYM in b.state["positions"]


FUT = dict(market="futures", leverage=3, raise_to_minimum=True)


def test_small_order_is_raised_to_the_minimum_on_the_step():
    amount, note = fit_order(0.0002, 76000.0, 75500.0, "LONG", 50.0, 0.0, minimums=(0.001, 50.0, 0.001), **FUT)
    assert amount == 0.001 and "raised" in note and "risk" in note


def test_minimum_notional_is_met_with_a_margin():
    amount, _ = fit_order(1.0, 0.2, 0.19, "LONG", 50.0, 0.0, minimums=(1.0, 5.0, 1.0), **FUT)
    assert amount == 26.0 and amount * 0.2 >= 5.0 * 1.01


def test_large_enough_order_is_unchanged():
    assert fit_order(30.0, 0.2, 0.19, "LONG", 50.0, 0.0, minimums=(1.0, 5.0, 1.0), **FUT) == (30.0, "")


def test_raising_is_off_means_skip():
    amount, note = fit_order(0.0002, 76000.0, 75500.0, "LONG", 50.0, 0.0, minimums=(0.001, 50.0, 0.001),
                             market="futures", leverage=3, raise_to_minimum=False)
    assert amount is None and "below" in note


def test_minimum_beyond_leverage_or_free_margin_is_skipped():
    assert fit_order(0.0001, 76000.0, 75500.0, "LONG", 20.0, 0.0, minimums=(0.001, 50.0, 0.001), **FUT)[0] is None
    assert fit_order(0.0001, 76000.0, 75500.0, "LONG", 50.0, 30.0, minimums=(0.001, 50.0, 0.001), **FUT)[0] is None


def test_spot_skips_shorts_and_caps_at_cash():
    spot = dict(market="spot", leverage=1, raise_to_minimum=True)
    assert fit_order(1.0, 10.0, 11.0, "SHORT", 50.0, 0.0, minimums=(0.1, 5.0, 0.1), **spot)[0] is None
    assert fit_order(6.0, 10.0, 9.0, "LONG", 50.0, 0.0, minimums=(0.1, 5.0, 0.1), **spot)[0] is None
    assert fit_order(0.2, 10.0, 9.0, "LONG", 50.0, 0.0, minimums=(0.1, 5.0, 0.1), **spot)[0] == pytest.approx(0.6)


def test_binance_symbols_name_the_futures_market_not_spot():
    from adapters.binance import BinanceAdapter
    assert BinanceAdapter().format_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert BinanceAdapter().format_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
    from utils.validator import normalize_symbol
    assert normalize_symbol("BTC/USDT:USDT") == "BTCUSDT"
