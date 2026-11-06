"""Paper mode: no path to a real order, no credential in memory, honest fills, separate state."""
import json
import os
import subprocess
import sys
import time

import ccxt
import pytest

import config
from paper.exchange import MINUTE_MS, PaperBlocked, PaperBroker, PaperExchange
from utils.persistence import get_absolute_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYM = "BTC/USDT:USDT"


class FakePublic:
    """Keyless market data. Any order or account call through it fails the test."""
    apiKey = ""
    secret = ""
    markets = {SYM: {"active": True}}

    def __init__(self, last=100.0, bid=99.9, ask=100.1):
        self.last, self.bid, self.ask, self.bars = last, bid, ask, []

    def fetch_ticker(self, symbol):
        return {"last": self.last, "bid": self.bid, "ask": self.ask}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [b for b in self.bars if since is None or b[0] >= since]

    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0.0}, "cost": {"min": 0.0}}, "precision": {"amount": 3}}

    def amount_to_precision(self, symbol, amount):
        return str(float(amount))

    def price_to_precision(self, symbol, price):
        return str(float(price))

    def __getattr__(self, name):
        raise AssertionError(f"public client must not receive {name}")


@pytest.fixture
def broker(tmp_path):
    b = PaperBroker(str(tmp_path / "broker.json"), initial=1000.0, fee=0.001, slippage_bps=10.0, leverage=3.0, match_interval_s=0.0)
    b.public = FakePublic()
    return b


def now_bar(offset=0):
    return (int(time.time() * 1000) // MINUTE_MS + offset) * MINUTE_MS


def bar(offset, o, h, l, c):
    return [now_bar(offset), o, h, l, c, 1.0]


def test_paper_is_the_default_and_secrets_are_not_loaded():
    code = ("import config;print(config.IS_PAPER, any([config.BINANCE_API_KEY, config.BINANCE_SECRET, config.BITGET_API_KEY,"
            "config.BITGET_API_SECRET, config.BITGET_PASSPHRASE, config.TELEGRAM_BOT_TOKEN]))")
    env = dict(os.environ, TRADING_MODE="", BINANCE_API_KEY="k" * 30, BINANCE_SECRET="s" * 30, TELEGRAM_BOT_TOKEN="t" * 30, PYTHONPATH=REPO)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip() == "True False", out.stderr


@pytest.mark.parametrize("value", ["paper", "PAPER", "demo", "true", "livee", ""])
def test_anything_but_exact_live_stays_paper(value):
    env = dict(os.environ, TRADING_MODE=value, PYTHONPATH=REPO)
    out = subprocess.run([sys.executable, "-c", "import config;print(config.IS_PAPER)"], env=env, capture_output=True, text=True, cwd=REPO)
    assert out.stdout.strip() == "True", out.stderr


def run_mode_check(**extra):
    env = dict(os.environ, PYTHONPATH=REPO, TRADING_MODE="live", **extra)
    return subprocess.run([sys.executable, "-c", "import main;main.check_trading_mode();print('ALLOWED')"], env=env,
                          capture_output=True, text=True, cwd=REPO)


def test_live_refused_without_confirmation():
    out = run_mode_check(LIVE_TRADING_CONFIRM="yes", DEMO_MODE="false", EXCHANGE="binance")
    assert out.returncode == 1 and "ALLOWED" not in out.stdout


def test_live_refused_when_demo_mode_would_trade_for_real_on_binance():
    out = run_mode_check(LIVE_TRADING_CONFIRM=config.LIVE_CONFIRM_PHRASE, DEMO_MODE="true", EXCHANGE="binance")
    assert out.returncode == 1 and "ALLOWED" not in out.stdout


def test_live_mode_check_passes_only_with_the_exact_phrase():
    out = run_mode_check(LIVE_TRADING_CONFIRM=config.LIVE_CONFIRM_PHRASE, DEMO_MODE="false", EXCHANGE="binance")
    assert out.returncode == 0 and "ALLOWED" in out.stdout


def test_real_data_handler_uses_paper_exchange_and_no_keys():
    from data_handler import DataHandler
    dh = DataHandler()
    assert isinstance(dh.exchange, PaperExchange)
    assert not dh.exchange._public.apiKey and not dh.exchange._public.secret
    assert not dh._exchange_config.get("apiKey") and not dh._exchange_config.get("secret")


@pytest.mark.parametrize("name", ["withdraw", "transfer", "fapiPrivatePostOrder", "fapiPrivateDeleteAllOpenOrders", "set_leverage",
                                  "private_mix_get_v2_mix_account_account", "post", "get", "request", "apiKey", "secret", "headers"])
def test_private_or_unknown_attributes_are_blocked(broker, name):
    ex = PaperExchange("binance", broker, public_factory=FakePublic)
    with pytest.raises(PaperBlocked):
        getattr(ex, name)


def test_client_with_credentials_is_refused(broker):
    class Keyed(FakePublic):
        apiKey = "real"
    with pytest.raises(PaperBlocked):
        PaperExchange("binance", broker, public_factory=Keyed)


def test_attributes_cannot_be_swapped(broker):
    ex = PaperExchange("binance", broker, public_factory=FakePublic)
    with pytest.raises(PaperBlocked):
        ex.create_order = lambda *a, **k: None


def test_orders_go_to_the_broker_never_the_public_client(broker):
    ex = PaperExchange("binance", broker, public_factory=FakePublic)
    order = ex.create_order(SYM, "market", "buy", 1.0)
    assert order["info"]["paper"] is True and order["status"] == "closed"


def test_market_buy_pays_ask_slippage_and_fee(broker):
    o = broker.create_order(SYM, "market", "buy", 2.0)
    px = 100.1 * 1.001
    assert o["average"] == pytest.approx(px) and o["filled"] == 2.0
    assert broker.state["cash"] == pytest.approx(1000 - 2 * px * 0.001)


def test_round_trip_pnl_and_ledger(broker):
    broker.create_order(SYM, "market", "buy", 1.0)
    broker.public.bid = 110.0
    broker.create_order(SYM, "market", "sell", 1.0, None, {"reduceOnly": True})
    entry, exit_ = 100.1 * 1.001, 110.0 * 0.999
    assert broker.state["realized"] == pytest.approx(exit_ - entry)
    assert broker.state["cash"] == pytest.approx(1000 + exit_ - entry - (entry + exit_) * 0.001)
    assert broker.state["positions"] == {}
    rows = [json.loads(x) for x in open(broker.ledger_path)]
    assert len(rows) == 2 and rows[1]["realized"] == pytest.approx(exit_ - entry)


def test_short_profits_when_price_falls(broker):
    broker.create_order(SYM, "market", "sell", 1.0)
    broker.public.ask = 90.0
    broker.create_order(SYM, "market", "buy", 1.0, None, {"reduceOnly": True})
    assert broker.state["realized"] > 0


def test_reduce_only_without_position_is_rejected(broker):
    with pytest.raises(ccxt.InvalidOrder):
        broker.create_order(SYM, "market", "sell", 1.0, None, {"reduceOnly": True})


def test_reduce_only_cannot_flip_the_position(broker):
    broker.create_order(SYM, "market", "buy", 1.0)
    broker.create_order(SYM, "market", "sell", 5.0, None, {"reduceOnly": True})
    assert broker.state["positions"] == {}


def test_margin_limit_rejects_oversized_entry(broker):
    with pytest.raises(ccxt.InsufficientFunds):
        broker.create_order(SYM, "market", "buy", 100.0)


def test_unknown_order_type_is_not_silently_filled(broker):
    with pytest.raises(ccxt.NotSupported):
        broker.create_order(SYM, "limit", "buy", 1.0, 99.0)


def long_with_stop(broker, stop=95.0, tp=None):
    broker.create_order(SYM, "market", "buy", 1.0)
    s = broker.create_order(SYM, "STOP_MARKET", "sell", 1.0, None, {"stopPrice": stop})
    t = broker.create_order(SYM, "TAKE_PROFIT_MARKET", "sell", 1.0, None, {"stopPrice": tp}) if tp else None
    return s, t


def test_stop_fills_at_trigger_with_slippage(broker):
    s, _ = long_with_stop(broker)
    broker.public.bars = [bar(1, 100, 100, 94, 96)]
    assert broker.fetch_open_orders(SYM) == []
    assert broker.state["positions"] == {} and broker.fetch_order(s["id"])["average"] == pytest.approx(95 * 0.999)


def test_stop_gap_fills_at_the_worse_open(broker):
    s, _ = long_with_stop(broker)
    broker.public.bars = [bar(1, 80, 81, 79, 80)]
    broker.match(force=True)
    assert broker.fetch_order(s["id"])["average"] == pytest.approx(80 * 0.999)


def test_stop_wins_when_stop_and_take_profit_share_a_bar(broker):
    s, t = long_with_stop(broker, 95.0, 110.0)
    broker.public.bars = [bar(1, 100, 111, 94, 100)]
    broker.match(force=True)
    assert broker.fetch_order(s["id"])["status"] == "closed" and broker.fetch_order(t["id"])["status"] == "expired"


def test_short_stop_triggers_upward(broker):
    broker.create_order(SYM, "market", "sell", 1.0)
    s = broker.create_order(SYM, "STOP_MARKET", "buy", 1.0, None, {"stopPrice": 105.0})
    broker.public.bars = [bar(1, 100, 106, 100, 105)]
    broker.match(force=True)
    assert broker.fetch_order(s["id"])["average"] == pytest.approx(105 * 1.001) and not broker.state["positions"]


def test_prices_from_before_the_order_cannot_trigger_it(broker):
    long_with_stop(broker)
    broker.public.bars = [bar(0, 100, 100, 90, 100), bar(-1, 100, 100, 80, 100)]
    broker.match(force=True)
    assert broker.state["positions"] and len(broker.state["orders"]) == 1


def test_stops_work_across_downtime_and_restart(broker):
    long_with_stop(broker)
    reborn = PaperBroker(broker.state_path, initial=1.0, fee=0.001, slippage_bps=10.0, match_interval_s=0.0)
    reborn.public = broker.public
    assert reborn.state["initial"] == 1000.0 and reborn.state["positions"]
    reborn.public.bars = [bar(i, 100, 100, 99, 99.5) for i in range(1, 5)] + [bar(5, 96, 96, 90, 92)]
    reborn.match(force=True)
    assert reborn.state["positions"] == {} and reborn.state["realized"] < 0


def test_price_feed_failure_leaves_the_account_unchanged(broker):
    long_with_stop(broker)
    before = json.dumps(broker.state, sort_keys=True)
    def fail(*a, **k):
        raise ccxt.NetworkError("down")
    broker.public.fetch_ohlcv = fail
    with pytest.raises(ccxt.NetworkError):
        broker.match(force=True)
    assert json.dumps(broker.state, sort_keys=True) == before


def test_balance_and_position_pass_the_bots_validators(broker):
    from utils.validator import Validator
    broker.create_order(SYM, "market", "sell", 1.0)
    broker.public.last = 90.0
    acc = Validator.balance(broker.fetch_balance())
    pos = Validator.position(broker.fetch_positions()[0])
    assert acc.equity > 1000 and acc.used_margin == pytest.approx(30.0)
    assert pos.symbol == "BTCUSDT" and pos.side == "SHORT" and pos.amount == 1.0 and pos.pnl > 0


def test_paper_and_live_state_never_share_a_folder():
    assert os.sep + "paper_" in get_absolute_path("x.json")
    assert os.environ["HORIZON_RUNTIME_DIR"] in get_absolute_path("x.json")
    code = "import config;from utils.persistence import get_absolute_path as g;print(g('x.json'))"
    env = dict(os.environ, TRADING_MODE="live", PYTHONPATH=REPO)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO).stdout
    assert "paper_" not in out and out.strip()


def test_full_entry_stop_and_exit_through_the_real_execution_engine(tmp_path, monkeypatch):
    """The bot's own order path, ending at the paper broker."""
    from data_handler import DataHandler
    from execution_engine import ExecutionEngine, is_protection_order
    from state_engine import StateEngine
    from utils.schemas import TradeSignal
    symbol = "PAPERFLOWUSDT"    # event journals are per symbol; keep other tests' journals clean
    b = PaperBroker(str(tmp_path / "b.json"), initial=1000.0, fee=0.001, slippage_bps=10.0, match_interval_s=0.0)
    public = FakePublic()
    b.public = public
    dh = DataHandler()
    dh._local.instance = PaperExchange("binance", b, public_factory=lambda: public)
    eng = ExecutionEngine(dh, simulation_mode=False)
    eng.state = StateEngine()
    sig = TradeSignal(symbol=symbol, trigger="LONG", confidence=1.0)
    pos = eng.execute_trade(symbol, sig, 1.0, 95.0, 110.0)
    assert pos is not None and pos.amount == 1.0 and pos.entry_price == pytest.approx(100.1 * 1.001)
    formatted = dh.format_symbol(symbol)
    orders = dh.exchange.fetch_open_orders(formatted)
    assert {o["type"] for o in orders} == {"STOP_MARKET", "TAKE_PROFIT_MARKET"}
    assert all(is_protection_order(o, pos) for o in orders)
    assert dh.fetch_active_positions()[0].symbol == symbol
    assert eng.close_full(symbol, pos)
    assert dh.fetch_active_positions() == [] and dh.exchange.fetch_open_orders(formatted) == []
    assert b.state["cash"] < 1000.0
    from utils.event_sourced_runtime import OwnershipRegistry
    OwnershipRegistry().get_owner(symbol).submit_event("POSITION_CLOSED", {})
    assert all(x.symbol != symbol for x in eng.state.active_positions)


def test_report_counts_closes_fees_and_drawdown(broker):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from paper_report import render, summarize
    broker.create_order(SYM, "market", "buy", 1.0)
    broker.public.bid = 90.0
    broker.create_order(SYM, "market", "sell", 1.0, None, {"reduceOnly": True})
    rows = [json.loads(x) for x in open(broker.ledger_path)]
    s = summarize(broker.state, rows)
    assert s["closes"] == 1 and s["win_rate"] == 0 and s["drawdown"] < 0 and s["return"] < 0
    assert "실제 돈이 아닙니다" in render(s)


def test_entry_atr_survives_event_projection_and_small_moves_do_not_close():
    """A position rebuilt from events had atr 0, so any price change above 0 read as a volatility spike."""
    from utils.event_sourced_runtime import OwnershipRegistry
    from utils.exit_engine import ExitAction, ExitEngine
    owner = OwnershipRegistry().get_owner("ATRKEEPUSDT")
    owner.submit_event("ENTRY_FILLED", {"side": "SHORT", "amount": 1.0, "price": 100.0})
    unknown = owner.get_state()
    unknown.stop_loss = 105.0
    assert unknown.atr == 0.0
    assert ExitEngine().compute_exit(unknown, 99.99, 1.0, "NEUTRAL", "RANGE")[2] != "VOL_SPIKE_PROTECTION"
    owner.submit_event("RECONCILE_CORRECTION", {"atr": 2.0})
    pos = owner.get_state()
    pos.stop_loss = 105.0
    assert pos.atr == 2.0
    assert ExitEngine().compute_exit(pos, 99.0, 2.0, "NEUTRAL", "RANGE")[2] != "VOL_SPIKE_PROTECTION"
    action, _, reason, _ = ExitEngine().compute_exit(pos, 106.5, 2.0, "NEUTRAL", "RANGE")
    assert action == ExitAction.FULL_CLOSE and reason == "VOL_SPIKE_PROTECTION"
    owner.submit_event("POSITION_CLOSED", {})


def test_a_normal_cycle_gap_is_not_a_system_wake():
    import config
    import pipeline
    assert pipeline.WAKE_GAP_SECONDS > config.RECONCILE_INTERVAL * 2


def test_firm_order_route_runs_end_to_end_in_paper_mode(tmp_path):
    """The path the live pipeline uses for every entry, including its log line."""
    from data_handler import DataHandler
    from execution_engine import ExecutionEngine
    from risk_engine import FirmOrderRequest
    from state_engine import StateEngine
    from dataclasses import fields
    from utils.event_sourced_runtime import OwnershipRegistry
    symbol = "PAPERFIRMUSDT"
    b = PaperBroker(str(tmp_path / "b.json"), initial=1000.0, fee=0.001, slippage_bps=10.0, match_interval_s=0.0)
    public = FakePublic()
    b.public = public
    dh = DataHandler()
    dh._local.instance = PaperExchange("binance", b, public_factory=lambda: public)
    eng = ExecutionEngine(dh, simulation_mode=False)
    eng.state = StateEngine()
    wanted = {"symbol": symbol, "side": "LONG", "amount": 1.0, "stop_loss": 95.0, "take_profit": 110.0, "approved": True,
              "conviction_at_entry": 1.0, "intent_id": "t1", "rejection_reason": ""}
    known = {f.name for f in fields(FirmOrderRequest)}
    pos = eng.execute_firm_order(FirmOrderRequest(**{k: v for k, v in wanted.items() if k in known}))
    assert pos is not None and b.state["positions"]
    OwnershipRegistry().get_owner(symbol).submit_event("POSITION_CLOSED", {})
