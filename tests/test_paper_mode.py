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


class SteppedPublic(FakePublic):
    """Rounds amounts down to 0.01 and rejects smaller ones, as the real client does for LINK."""

    def amount_to_precision(self, symbol, amount):
        rounded = int(float(amount) * 100 + 1e-9) / 100
        if rounded <= 0:
            raise ccxt.InvalidOrder(f"amount of {symbol} must be greater than minimum amount precision of 0.01")
        return f"{rounded:.2f}"


def test_partial_close_is_rounded_so_no_dust_outlives_the_stop(broker):
    public = SteppedPublic()
    broker.public = public
    ex = PaperExchange("binance", broker, public_factory=lambda: public)
    ex.create_order(SYM, "market", "buy", 8.04)
    ex.create_order(SYM, "market", "sell", 2.412, None, {"reduceOnly": True})
    assert broker.state["positions"][SYM]["amount"] == pytest.approx(5.63)
    ex.create_order(SYM, "STOP_MARKET", "sell", 5.63, None, {"stopPrice": 95.0})
    public.bars = [bar(1, 100, 100, 90, 92)]
    broker.match(force=True)
    assert SYM not in broker.state["positions"]


def test_amount_below_the_market_step_is_rejected(broker):
    public = SteppedPublic()
    broker.public = public
    ex = PaperExchange("binance", broker, public_factory=lambda: public)
    with pytest.raises(ccxt.InvalidOrder):
        ex.create_order(SYM, "market", "buy", 0.008)
    assert not broker.state["positions"]


def test_fill_in_the_current_bar_is_not_dated_in_the_future(broker):
    long_with_stop(broker)
    broker.public.bars = [bar(1, 100, 100, 90, 92)]
    broker.match(force=True)
    assert broker.state["trades"][-1]["timestamp"] <= time.time() * 1000


def test_fill_replayed_from_a_past_bar_keeps_that_bars_time(broker):
    s, _ = long_with_stop(broker)
    earlier = int(time.time() * 1000) - 10 * MINUTE_MS
    broker.state["orders"][s["id"]]["created_ms"] = earlier
    broker.state["last_checked"][SYM] = earlier
    past = bar(-5, 100, 100, 90, 92)
    broker.public.bars = [past]
    broker.match(force=True)
    assert broker.state["trades"][-1]["timestamp"] == past[0] + MINUTE_MS - 1


@pytest.fixture
def watched_position(tmp_path):
    from data_handler import DataHandler
    from execution_engine import ExecutionEngine
    from state_engine import StateEngine
    from utils.event_sourced_runtime import OwnershipRegistry
    from utils.schemas import TradeSignal
    symbol = "PAPERWATCHUSDT"
    b = PaperBroker(str(tmp_path / "b.json"), initial=1000.0, fee=0.001, slippage_bps=10.0, match_interval_s=0.0)
    public = FakePublic()
    b.public = public
    dh = DataHandler()
    dh._local.instance = PaperExchange("binance", b, public_factory=lambda: public)
    eng = ExecutionEngine(dh, simulation_mode=False)
    eng.state = StateEngine()
    eng.pending_sl.pop(symbol, None)    # the 2-second duplicate guard would otherwise see the previous test's stop
    pos = eng.execute_trade(symbol, TradeSignal(symbol=symbol, trigger="LONG", confidence=1.0), 1.0, 95.0, 0.0)
    assert pos.stop_loss == 95.0
    yield eng, pos, b, public, dh.format_symbol(symbol)
    eng.pending_sl.pop(symbol, None)
    OwnershipRegistry().get_owner(symbol).submit_event("POSITION_CLOSED", {})


def test_watchdog_places_no_stop_for_a_position_its_stop_just_closed(watched_position):
    eng, pos, b, public, formatted = watched_position
    eng.pending_sl[pos.symbol]["timestamp"] = 0     # the stop was placed long ago, past the duplicate guard
    public.bars = [bar(1, 100, 100, 90, 92)]
    eng.watchdog_verify_sl(pos)     # the position snapshot predates the fill found by this call
    assert not b.state["positions"]
    assert len([o for o in b.fetch_orders(formatted) if o["type"] == "STOP_MARKET"]) == 1


def test_watchdog_still_restores_a_missing_stop_on_an_open_position(watched_position):
    eng, pos, b, public, formatted = watched_position
    b.cancel_all_orders(formatted)
    eng.pending_sl.pop(pos.symbol, None)
    eng.watchdog_verify_sl(pos)
    assert [o["type"] for o in b.fetch_open_orders(formatted)] == ["STOP_MARKET"]


def test_watchdog_restores_the_stop_when_the_position_check_fails(watched_position, monkeypatch):
    eng, pos, b, public, formatted = watched_position
    b.cancel_all_orders(formatted)
    eng.pending_sl.pop(pos.symbol, None)
    real = eng.dh.fetch_active_positions
    calls = []

    def fail_first(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("exchange did not answer")
        return real(*a, **k)

    monkeypatch.setattr(eng.dh, "fetch_active_positions", fail_first)
    eng.watchdog_verify_sl(pos)
    assert [o["type"] for o in b.fetch_open_orders(formatted)] == ["STOP_MARKET"]


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


def _entry_pipeline(approved=True):
    """Runs the real entry worker with stubbed engines, down to the order call."""
    from unittest.mock import MagicMock
    import threading
    import pandas as pd
    from pipeline import Pipeline, WorkerLease
    from risk_engine import RiskEngine
    p = Pipeline.__new__(Pipeline)
    p._worker_lock = threading.RLock()
    p._entry_gate_lock = threading.Lock()
    p.active_entry_workers = {}
    p.signal_generated_count = p.execution_attempt_count = p.risk_rejection_count = 0
    p.dh, p.signal, p.conviction, p.state, p.execution = (MagicMock() for _ in range(5))
    p.state.lock = threading.RLock()
    p.state.last_entry_bar_ts = {}
    p.state.is_position_active.return_value = False
    p.state.active_positions = []
    p.dh.fetch_active_positions.return_value = []
    p.dh.fetch_ticker.return_value = MagicMock(last=100.0)
    p.dh.fetch_orderbook.return_value = {"bids": [[99.9, 10.0]], "asks": [[100.1, 10.0]]}
    p.signal.compute_all_signals.return_value = pd.DataFrame([{"timestamp": 1, "atr": 1.0}])
    p.signal.get_last_signal.return_value = MagicMock(trigger="LONG")
    p.conviction.evaluate_signal.return_value = MagicMock(side="LONG", composite_score=1.0)
    p.execution.execute_firm_order.return_value = None
    p.risk = RiskEngine.__new__(RiskEngine)
    p.risk.lock = threading.RLock()
    p.risk.trade_timestamps = []
    p.risk.hourly_trades = {}
    p.risk.approve_order = MagicMock(return_value=MagicMock(approved=approved, amount=1.0))
    def lease_for(symbol):
        lease = WorkerLease(symbol, "ENTRY")
        p.active_entry_workers[symbol] = lease
        return lease
    return p, lease_for


@pytest.fixture
def allow_entries(monkeypatch):
    from utils.degraded_manager import DegradedExchangeManager
    from utils.runtime_coordinator import RuntimeCoordinator
    monkeypatch.setattr(DegradedExchangeManager, "is_entry_allowed", lambda self: True)
    monkeypatch.setattr(RuntimeCoordinator, "get_state", lambda self, key, default=None: True)


def test_worker_flagged_as_hung_sends_no_order(allow_entries):
    p, lease_for = _entry_pipeline()
    lease = lease_for("AAAUSDT")
    lease.status = "STALLED"
    p._process_single_entry("AAAUSDT", MagicMockAccount(), lease)
    p.execution.execute_firm_order.assert_not_called()
    assert p.risk.trade_timestamps == []  # the unused slot is returned


def test_replaced_worker_sends_no_order(allow_entries):
    p, lease_for = _entry_pipeline()
    old = lease_for("AAAUSDT")
    lease_for("AAAUSDT")
    p._process_single_entry("AAAUSDT", MagicMockAccount(), old)
    p.execution.execute_firm_order.assert_not_called()


def test_healthy_worker_still_orders(allow_entries):
    p, lease_for = _entry_pipeline()
    p._process_single_entry("AAAUSDT", MagicMockAccount(), lease_for("AAAUSDT"))
    p.execution.execute_firm_order.assert_called_once()


def test_claim_marks_lease_as_ordering(allow_entries):
    from pipeline import CLAIMED, LEASE_ORDERING
    p, lease_for = _entry_pipeline()
    lease = lease_for("AAAUSDT")
    assert p._claim_entry_order("AAAUSDT", lease) == CLAIMED
    assert lease.status == LEASE_ORDERING


def test_parallel_entries_cannot_exceed_the_minute_limit(allow_entries):
    p, lease_for = _entry_pipeline()
    for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        p._process_single_entry(sym, MagicMockAccount(), lease_for(sym))
    assert p.execution.execute_firm_order.call_count == config.MAX_TRADES_PER_MINUTE
    assert len(p.risk.trade_timestamps) == config.MAX_TRADES_PER_MINUTE


def _held(*symbols):
    from unittest.mock import MagicMock
    return [MagicMock(symbol=s) for s in symbols]


def test_parallel_entries_cannot_exceed_the_position_limit(allow_entries):
    """Workers approved at the same moment each see the same count; only the last check before ordering can stop them."""
    import threading
    p, lease_for = _entry_pipeline()
    p.state.active_positions = _held("HELD1USDT", "HELD2USDT")
    both_approved = threading.Barrier(2, timeout=5)
    approve = p.risk.approve_order.return_value
    def approve_order(**kwargs):
        both_approved.wait()
        return approve
    p.risk.approve_order = approve_order
    order_may_finish = threading.Event()
    def slow_order(request):
        assert order_may_finish.wait(5)
    p.execution.execute_firm_order.side_effect = slow_order
    leases = {s: lease_for(s) for s in ("AAAUSDT", "BBBUSDT")}
    workers = [threading.Thread(target=p._process_single_entry, args=(s, MagicMockAccount(), l)) for s, l in leases.items()]
    for w in workers:
        w.start()
    deadline = time.time() + 5
    while p.risk_rejection_count == 0 and p.execution.execute_firm_order.call_count < 2 and time.time() < deadline:
        time.sleep(0.01)
    order_may_finish.set()
    for w in workers:
        w.join(5)
    assert p.execution.execute_firm_order.call_count == 1
    assert p.risk_rejection_count == 1
    assert len(p.risk.trade_timestamps) == 1  # the refused worker returned its slot


def test_order_still_being_sent_holds_a_position_slot(allow_entries):
    from pipeline import CLAIMED
    p, lease_for = _entry_pipeline()
    p.state.active_positions = _held("HELD1USDT", "HELD2USDT")
    assert p._claim_entry_order("AAAUSDT", lease_for("AAAUSDT")) == CLAIMED
    assert p._claim_entry_order("BBBUSDT", lease_for("BBBUSDT")) != CLAIMED


def test_failed_order_gives_its_slot_back(allow_entries):
    p, lease_for = _entry_pipeline()
    p.state.active_positions = _held("HELD1USDT", "HELD2USDT")
    p._process_single_entry("AAAUSDT", MagicMockAccount(), lease_for("AAAUSDT"))  # the stubbed order returns None
    p.risk.trade_timestamps = []
    p._process_single_entry("BBBUSDT", MagicMockAccount(), lease_for("BBBUSDT"))
    assert p.execution.execute_firm_order.call_count == 2


def test_order_already_in_position_list_is_not_counted_twice(allow_entries):
    from pipeline import CLAIMED, LEASE_ORDERING
    p, lease_for = _entry_pipeline()
    p.state.active_positions = _held("HELD1USDT", "AAAUSDT")
    lease_for("AAAUSDT").status = LEASE_ORDERING  # filled, worker not finished yet
    assert p._claim_entry_order("BBBUSDT", lease_for("BBBUSDT")) == CLAIMED


def test_entry_is_refused_when_already_over_the_limit(allow_entries):
    p, lease_for = _entry_pipeline()
    p.state.active_positions = _held("A1USDT", "A2USDT", "A3USDT", "A4USDT")
    p._process_single_entry("AAAUSDT", MagicMockAccount(), lease_for("AAAUSDT"))
    p.execution.execute_firm_order.assert_not_called()


def test_rate_limit_is_named_for_what_it_counts():
    import threading
    from risk_engine import RiskEngine
    r = RiskEngine.__new__(RiskEngine)
    r.lock = threading.RLock()
    r.trade_timestamps = [time.time()] * config.MAX_TRADES_PER_MINUTE
    ok, reason = r.check_system_health(1000.0)
    assert not ok and reason.startswith("ENTRY RATE LIMIT")


def test_markets_are_downloaded_once_across_threads(monkeypatch):
    import paper.exchange as px

    class Client:
        loads = 0
        def __init__(self, params):
            self.markets, self.currencies = None, None
        def load_markets(self):
            Client.loads += 1
            self.markets, self.currencies = {"A/USDT": {}}, {"USDT": {}}
        def set_markets(self, markets, currencies=None):
            self.markets, self.currencies = markets, currencies

    monkeypatch.setattr(px, "_shared_markets", None)
    monkeypatch.setattr(px.ccxt, "binance", Client)
    clients = [px._public_client("binance") for _ in range(4)]
    assert Client.loads == 1
    assert all(c.markets == {"A/USDT": {}} for c in clients)


def test_failed_market_download_is_retried_by_the_next_client(monkeypatch):
    import paper.exchange as px

    class Client:
        fail = True
        def __init__(self, params):
            self.markets, self.currencies = None, None
        def load_markets(self):
            if Client.fail:
                raise ccxt.NetworkError("down")
            self.markets, self.currencies = {"A/USDT": {}}, {}

    monkeypatch.setattr(px, "_shared_markets", None)
    monkeypatch.setattr(px.ccxt, "binance", Client)
    assert px._public_client("binance").markets is None
    Client.fail = False
    assert px._public_client("binance").markets == {"A/USDT": {}}


class MagicMockAccount:
    equity = 1000.0
    used_margin = 0.0
    is_available = True


def _bias_frame(*biases, start="2026-01-01 00:00"):
    import pandas as pd
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=len(biases), freq="15min"), "trend_bias": list(biases)})


def _seconds(when):
    import pandas as pd
    return (pd.Timestamp(when) - pd.Timestamp(0)).total_seconds()


def test_counter_trend_entry_is_not_a_flip():
    from pipeline import trend_was_with_position_since_entry as flipped
    df = _bias_frame("SHORT", "SHORT", "SHORT")
    assert not flipped(df, "LONG", _seconds("2026-01-01 00:16"))


def test_trend_turning_against_after_being_with_the_position_is_a_flip():
    from pipeline import trend_was_with_position_since_entry as flipped
    assert flipped(_bias_frame("LONG", "LONG", "SHORT"), "LONG", _seconds("2026-01-01 00:01"))  # aligned entry
    assert flipped(_bias_frame("SHORT", "LONG", "SHORT"), "LONG", _seconds("2026-01-01 00:01"))  # counter-trend, then with, then against


def test_trend_before_the_entry_bar_does_not_count():
    from pipeline import trend_was_with_position_since_entry as flipped
    assert not flipped(_bias_frame("LONG", "SHORT", "SHORT"), "LONG", _seconds("2026-01-01 00:16"))


def test_unknown_or_old_entry_time_keeps_the_plain_trend_exit():
    from pipeline import trend_was_with_position_since_entry as flipped
    df = _bias_frame("SHORT", "SHORT")
    assert flipped(df, "LONG", 0.0)
    assert flipped(df, "LONG", _seconds("2025-12-01 00:00"))


def test_exit_engine_applies_the_flip_only_when_told_it_is_one():
    from utils.exit_engine import ExitAction, ExitEngine
    from utils.schemas import PositionSchema
    pos = PositionSchema(symbol="AAAUSDT", side="LONG", amount=1.0, entry_price=100.0, stop_loss=95.0)
    assert ExitEngine().compute_exit(pos, 100.0, 1.0, "SHORT", "TREND", trend_flipped=False)[2] != "TREND_FLIP_EXIT"
    action, _, reason, _ = ExitEngine().compute_exit(pos, 100.0, 1.0, "SHORT", "TREND", trend_flipped=True)
    assert action == ExitAction.FULL_CLOSE and reason == "TREND_FLIP_EXIT"
    # The hard stop still works on a counter-trend position.
    assert ExitEngine().compute_exit(pos, 94.0, 1.0, "SHORT", "TREND", trend_flipped=False)[2] == "STOP_LOSS_HIT"


def test_entry_time_survives_event_projection():
    from utils.event_sourced_runtime import OwnershipRegistry
    owner = OwnershipRegistry().get_owner("ENTRYTIMEUSDT")
    owner.submit_event("ENTRY_FILLED", {"side": "LONG", "amount": 1.0, "price": 100.0})
    owner.submit_event("RECONCILE_CORRECTION", {"atr": 2.0, "timestamp": 1234.5})
    assert owner.get_state().timestamp == 1234.5
    owner.submit_event("POSITION_CLOSED", {})


def _engine_at(monkeypatch, clock):
    from risk_engine import RiskEngine
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    monkeypatch.setattr(RiskEngine, "_save_performance", lambda self: None)
    return RiskEngine()


def _lose(r, times=1):
    for _ in range(times):
        r.record_trade_result("AAAUSDT", -1.0, 10.0, 1000.0)


def test_cooldown_ends_by_itself_with_no_trades(monkeypatch):
    """A flat bot makes no trades, so a cooldown that only a win can clear would never end."""
    clock = {"now": 1_000_000.0}
    r = _engine_at(monkeypatch, clock)
    _lose(r, 3)
    assert r.cooldown_active
    clock["now"] += config.COOLDOWN_AFTER_LOSSES_SECONDS - 1
    assert r.cooldown_active
    clock["now"] += 1
    assert not r.cooldown_active


def test_loss_count_starts_over_after_a_cooldown(monkeypatch):
    clock = {"now": 1_000_000.0}
    r = _engine_at(monkeypatch, clock)
    _lose(r, 3)
    clock["now"] += config.COOLDOWN_AFTER_LOSSES_SECONDS
    _lose(r, 2)
    assert not r.cooldown_active
    _lose(r)
    assert r.cooldown_active


def test_win_still_ends_a_cooldown_early(monkeypatch):
    r = _engine_at(monkeypatch, {"now": 1_000_000.0})
    _lose(r, 3)
    r.record_trade_result("AAAUSDT", 1.0, 10.0, 1000.0)
    assert not r.cooldown_active and r.consecutive_losses == 0


def test_two_losses_do_not_start_a_cooldown(monkeypatch):
    r = _engine_at(monkeypatch, {"now": 1_000_000.0})
    _lose(r, 2)
    assert not r.cooldown_active


def test_partial_close_below_minimum_size_is_skipped_not_an_error():
    from utils.exit_engine import ExitAction, ExitEngine
    from utils.schemas import PositionSchema
    pos = PositionSchema(symbol="AAAUSDT", side="LONG", amount=0.001, entry_price=100.0, stop_loss=90.0, atr=5.0)
    action, _, _, _ = ExitEngine().compute_exit(pos, 111.0, 5.0, "LONG", "TREND", min_qty=0.01)
    assert action != ExitAction.PARTIAL_CLOSE_TP1


def test_float_noise_does_not_leave_an_unclosable_remainder(broker):
    """5.39 - 2.15 - 3.23 is 0.00999..., below the 0.01 minimum, which no order could close."""
    broker.create_order(SYM, "market", "buy", 5.39)
    broker.create_order(SYM, "market", "sell", 2.15, None, {"reduceOnly": True})
    broker.create_order(SYM, "market", "sell", 3.23, None, {"reduceOnly": True})
    assert broker.state["positions"][SYM]["amount"] == 0.01


def test_remainder_saved_with_float_noise_is_repaired_on_load(tmp_path):
    path = tmp_path / "broker.json"
    b = PaperBroker(str(path))
    b.state["positions"][SYM] = {"side": "SHORT", "amount": 0.009999999999999787, "entry": 10.0, "opened_ms": 1}
    b._save()
    assert PaperBroker(str(path)).state["positions"][SYM]["amount"] == 0.01


def test_stop_after_partial_close_covers_the_whole_remainder():
    """3.2399999... was cut to 3.23 by amount precision, so the stop left 0.01 behind."""
    from tests.test_execution_partial_close import _FakeDH, _FakeExchange
    from execution_engine import ExecutionEngine
    from utils.schemas import PositionSchema
    ex = _FakeExchange()
    eng = ExecutionEngine(_FakeDH(ex), simulation_mode=False)
    pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=5.39, entry_price=100.0, stop_loss=90.0)
    eng.close_partial("BTCUSDT", pos, 2.15, min_qty=0.0)
    assert pos.amount == 3.24
    assert ex._orders[-1]["amount"] == 3.24


@pytest.mark.parametrize("tp_action, filled_event", [("PARTIAL_CLOSE_TP1", "TP1_FILLED"), ("PARTIAL_CLOSE_TP2", "TP2_FILLED")])
def test_stop_replaced_by_a_partial_close_is_recorded(tmp_path, tp_action, filled_event):
    """Positions are rebuilt from events, so without this the next stop update cancels the old, gone stop."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import threading
    import pandas as pd
    from pipeline import Pipeline, WorkerLease
    from utils.event_sourced_runtime import OwnershipRegistry
    from utils.exit_engine import ExitAction
    symbol = f"STOPKEEP{filled_event[:3]}USDT"
    owner = OwnershipRegistry().get_owner(symbol)
    owner.submit_event("ENTRY_FILLED", {"side": "LONG", "amount": 1.0, "price": 100.0})
    owner.submit_event("SL_PLACED", {"price": 95.0, "order_id": "OLD"})
    if filled_event == "TP2_FILLED":
        owner.submit_event("TP1_FILLED", {"amount": 0.2})
    p = Pipeline.__new__(Pipeline)
    p._worker_lock = threading.Lock()
    p.active_exit_workers = {}
    p.state = SimpleNamespace(lock=threading.RLock(), persist=lambda: None)
    p.dh, p.signal, p.risk, p.execution, p.exit = (MagicMock() for _ in range(5))
    p.signal.compute_all_signals.return_value = pd.DataFrame([{"timestamp": 1, "atr": 1.0}])
    p.dh.fetch_ticker.return_value = MagicMock(last=110.0)
    p.exit.compute_exit.return_value = (ExitAction[tp_action], 0.2, "TP", 100.0)
    def close_partial(sym, pos, amount, min_qty=0.0, new_stop=None):
        pos.sl_plan_id = "NEW"
        pos.stop_loss = new_stop or pos.stop_loss
        return {"ok": True, "order_id": "X", "filled_amount": amount}
    p.execution.close_partial.side_effect = close_partial
    p._process_single_exit(owner.get_state(), WorkerLease(symbol, "EXIT"))
    state = owner.get_state()
    assert state.sl_plan_id == "NEW"
    assert state.stop_loss == (100.0 if filled_event == "TP1_FILLED" else 95.0)
    owner.submit_event("POSITION_CLOSED", {})
