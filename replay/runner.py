"""Runs the bot's real Pipeline over past minutes.

Real code: Pipeline, RiskEngine, ConvictionEngine, ExitEngine, ExecutionEngine, StateEngine, PaperBroker.
Replaced: the clock, the price feed, and SignalEngine's computation, whose results were stored by the bot's own code.
A worker pool runs inline, so results repeat exactly. Nothing here can see a bar after the current minute.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import Counter
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from replay.market_data import BAR_MINUTES, BAR_MS, DATA_DIR, MINUTE_MS, WINDOW_BARS, closed_bars, load_bars, window_at
from replay.signals import REGIME_NAMES, TRIGGER_NAMES, load_signals


class Clock:
    now_s = 0.0

    def time(self) -> float:
        return self.now_s


def _bare(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "")


class ReplayFeed:
    """Stands in for the keyless public client. Everything it returns ends at the current minute."""
    apiKey = ""
    secret = ""
    id = "replay"
    has = {}
    timeframes = {"1m": "1m", "15m": "15m"}

    def __init__(self, bars: dict, rules: dict):
        self.bars, self.rules = bars, rules
        self.t0 = {s: int(b[0, 0]) for s, b in bars.items()}
        self.closed = {s: closed_bars(b) for s, b in bars.items()}
        self.minute_ms = 0
        self.markets = {s: {"active": True} for s in bars}

    def set_minute(self, minute_ms: int) -> None:
        self.minute_ms = minute_ms

    def _index(self, symbol: str) -> int:
        return (self.minute_ms - self.t0[symbol]) // MINUTE_MS

    def load_markets(self, *a, **k) -> dict:
        return self.markets

    def market(self, symbol: str) -> dict:
        rule = self.rules[_bare(symbol)]
        return {"limits": {"amount": {"min": rule["min_amount"]}, "cost": {"min": 0.0}}, "precision": {"amount": rule["amount_step"]}}

    def amount_to_precision(self, symbol: str, amount) -> str:
        import ccxt
        step = self.rules[_bare(symbol)]["amount_step"]
        rounded = np.floor(float(amount) / step + 1e-9) * step
        if rounded <= 0:
            raise ccxt.InvalidOrder(f"amount of {symbol} must be greater than minimum amount precision of {step}")
        return f"{rounded:.10f}"

    def price_to_precision(self, symbol: str, price) -> str:
        tick = self.rules[_bare(symbol)]["price_tick"]
        return f"{round(float(price) / tick) * tick:.10f}"

    def fetch_ticker(self, symbol: str) -> dict:
        s = _bare(symbol)
        last = float(self.bars[s][self._index(s), 4])
        return {"symbol": symbol, "last": last, "bid": last, "ask": last, "timestamp": self.minute_ms + MINUTE_MS - 1,
                "high": last, "low": last, "baseVolume": 0.0, "quoteVolume": 0.0, "info": {}}

    def fetch_order_book(self, symbol: str, limit=None) -> dict:
        """No past order books exist. A balanced book leaves the imbalance score at zero and adds no spread penalty,
        and it is deep enough that the order-book slippage limit never refuses an entry."""
        last = self.fetch_ticker(symbol)["last"]
        return {"bids": [[last, 1e12]], "asks": [[last, 1e12]]}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", since=None, limit=None) -> list:
        s = _bare(symbol)
        index = self._index(s)
        bars = self.bars[s]
        if timeframe == "1m":
            lo = 0 if since is None else max(0, (int(since) - self.t0[s]) // MINUTE_MS)
            hi = index + 1
            if limit:
                hi = min(hi, lo + int(limit))
            return bars[lo:hi].tolist()
        closed, first = self.closed[s]
        return window_at(bars, closed, first, index).tolist()


def make_broker_class():
    from paper.exchange import PaperBroker

    class ReplayBroker(PaperBroker):
        """The paper broker's own fills and costs, on the replay clock, with funding from past rates."""

        def _save(self) -> None:
            pass

        def apply_funding(self, symbol: str, rate: float, price: float, when_ms: int) -> None:
            with self.lock:
                pos = self.state["positions"].get(symbol)
                if not pos:
                    return
                sign = 1.0 if pos["side"] == "LONG" else -1.0
                cost = sign * pos["amount"] * price * rate
                self.state["cash"] -= cost
                self.state["funding"] = self.state.get("funding", 0.0) + cost
                self.state["funding_events"] = self.state.get("funding_events", 0) + 1

    return ReplayBroker


class SerialExecutor:
    """Runs a worker at once on this thread. Keeps every Pipeline line, and makes entry order repeatable."""

    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)
        return future

    def shutdown(self, *a, **k) -> None:
        pass


class Frames:
    """The DataFrame the bot gets from compute_all_signals, built from stored rows only up to the current minute."""

    def __init__(self, signals: dict, start_ms: int):
        self.signals, self.start_ms = signals, start_ms
        self.minute_ms = start_ms

    def index(self, symbol: str) -> int:
        return (self.minute_ms - self.start_ms) // MINUTE_MS

    def trigger(self, symbol: str) -> int:
        return int(self.signals[symbol]["trigger"][self.index(symbol)])

    def frame(self, symbol: str):
        import pandas as pd
        s = self.signals[symbol]
        i = self.index(symbol)
        bar_start = self.minute_ms // BAR_MS * BAR_MS
        # Closed bars since the replay began, each with the trend it showed in its final minute.
        done = (bar_start - self.start_ms) // MINUTE_MS
        ends = np.arange(done - 1, -1, -BAR_MINUTES)[:WINDOW_BARS - 1][::-1]
        stamps = np.append((self.start_ms + (ends - (BAR_MINUTES - 1)) * MINUTE_MS), bar_start)
        bias = np.append(s["bias"][ends], s["bias"][i])
        return pd.DataFrame({
            "timestamp": pd.to_datetime(stamps, unit="ms"),
            "trend_bias": np.where(bias > 0, "LONG", "SHORT").astype(object),
            "trigger": TRIGGER_NAMES[int(s["trigger"][i])], "regime": REGIME_NAMES[int(s["regime"][i])],
            "atr": float(s["atr"][i]), "score": int(s["score"][i]), "confidence": float(s["confidence"][i]),
        })


class CountErrors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.counts = Counter()

    def emit(self, record):
        self.counts[record.getMessage().split(" ")[0][:40]] += 1


def run(name: str, start_ms: int, end_ms: int, symbols: list[str], *, initial: float = 1000.0, fee: float = 0.0005,
        slippage_bps: float = 5.0, data_dir: Path = DATA_DIR, runs_dir: Path | None = None) -> dict:
    """Needs the isolated environment set by scripts/replay.py before any bot module is imported."""
    run_dir = (runs_dir or data_dir / "runs") / name
    if Path(os.environ["HORIZON_RUNTIME_DIR"]).resolve() != (run_dir / "runtime").resolve():
        raise RuntimeError("replay state is not isolated from the running bot")

    bars = {s: load_bars(s, data_dir) for s in symbols}
    signals = {s: load_signals(s, start_ms, end_ms, data_dir) for s in symbols}
    funding = {s: np.load(data_dir / f"{s}.funding.npy") for s in symbols}
    feed = ReplayFeed(bars, json.loads((data_dir / "markets.json").read_text()))
    frames = Frames(signals, start_ms)
    clock = Clock()
    clock.now_s = start_ms / 1000

    time.time = clock.time      # this process only; it exists solely for the replay
    os.fsync = lambda fd: None

    import config
    if config.IS_LIVE or config.BINANCE_API_KEY:
        raise RuntimeError("replay must run in paper mode with no keys")
    # Earlier replays ran futures at 3x and skipped orders below the minimum; screen settings must not change that.
    config.PAPER_MARKET, config.PAPER_LEVERAGE, config.PAPER_MIN_ORDER = "futures", 3, False
    config.SYMBOLS = list(symbols)
    from logger import errors_logger, trades_logger
    for lg in (trades_logger, errors_logger):
        lg.handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
    trades_logger.setLevel(logging.ERROR)
    errors = CountErrors()
    errors_logger.addHandler(errors)

    import paper.exchange as paper_exchange
    paper_exchange._public_client = lambda exchange_name: feed
    Broker = make_broker_class()
    broker = Broker(str(run_dir / "broker.json"), initial=initial, fee=fee, slippage_bps=slippage_bps,
                    leverage=config.MAX_EFFECTIVE_LEVERAGE, match_interval_s=0.0)
    broker.public = feed
    paper_exchange._broker = broker

    from conviction_engine import ConvictionEngine
    from data_handler import DataHandler
    from execution_engine import ExecutionEngine
    from pipeline import Pipeline
    from risk_engine import RiskEngine
    from signal_engine import SignalEngine
    from state_engine import StateEngine
    from utils.exit_engine import ExitAction, ExitEngine
    from utils.journal import ExecutionJournal

    class ReplayPipeline(Pipeline):
        def _process_single_entry(self, symbol, account, lease):
            # Same outcome as the first check in the real worker, without building the frame.
            if frames.trigger(symbol) == 0:
                with self._worker_lock:
                    self.active_entry_workers.pop(symbol, None)
                return
            return super()._process_single_entry(symbol, account, lease)

    signal = SignalEngine()
    signal.compute_all_signals = lambda df, symbol: df
    dh = DataHandler()
    dh.fetch_ohlcv = lambda symbol, timeframe, limit=300: frames.frame(symbol)
    risk = RiskEngine()
    execution = ExecutionEngine(dh, simulation_mode=False)
    ExecutionJournal.verify_and_align_snapshot()
    state = StateEngine()
    execution.state = state
    exit_engine = ExitEngine()
    reasons = Counter()
    real_compute_exit = exit_engine.compute_exit

    def counted_exit(*a, **k):
        result = real_compute_exit(*a, **k)
        if result[0] != ExitAction.HOLD:
            reasons[result[2]] += 1
        return result

    exit_engine.compute_exit = counted_exit
    pipe = ReplayPipeline(dh=dh, signal_engine=signal, conviction_engine=ConvictionEngine(), risk_engine=risk,
                          execution_engine=execution, exit_engine=exit_engine, state_engine=state)
    pipe.exit_executor = SerialExecutor()
    pipe.entry_executor = SerialExecutor()

    def formatted(sym: str) -> str:
        return dh.format_symbol(sym)

    funding_next = {s: int(np.searchsorted(funding[s][:, 0], start_ms)) for s in symbols}
    curve = []
    began = time.perf_counter()
    summary_path = run_dir / "summary.json"
    cooldown_minutes = 0

    def write_summary(done: bool, minute_ms: int) -> dict:
        s = broker.state
        out = {"name": name, "done": done, "start_ms": start_ms, "end_ms": end_ms, "reached_ms": minute_ms, "symbols": symbols,
               "initial": initial, "cash": s["cash"], "realized": s["realized"], "fees": s["fees"], "funding": s.get("funding", 0.0),
               "funding_events": s.get("funding_events", 0), "fee_rate": fee, "slippage_bps": slippage_bps,
               "open_positions": {k: v for k, v in s["positions"].items()}, "exit_reasons": dict(reasons),
               "errors": dict(errors.counts), "signals": pipe.signal_generated_count, "attempts": pipe.execution_attempt_count,
               "risk_rejections": pipe.risk_rejection_count, "cooldown_minutes": cooldown_minutes, "curve": curve,
               "seconds": time.perf_counter() - began}
        summary_path.write_text(json.dumps(out))
        return out

    minute_ms = start_ms
    for minute_ms in range(start_ms, end_ms, MINUTE_MS):
        clock.now_s = (minute_ms + MINUTE_MS - 1) / 1000    # the end of this minute: its bar is complete, none after it exists
        feed.set_minute(minute_ms)
        frames.minute_ms = minute_ms
        for s in symbols:
            rows = funding[s]
            while funding_next[s] < len(rows) and rows[funding_next[s], 0] <= minute_ms:
                broker.apply_funding(formatted(s), float(rows[funding_next[s], 1]), feed.fetch_ticker(s)["last"], int(rows[funding_next[s], 0]))
                funding_next[s] += 1
        pipe.run_cycle()
        cooldown_minutes += bool(risk.cooldown_active)
        if (minute_ms - start_ms) % (6 * 3600_000) == 0:
            bal = broker.fetch_balance()
            curve.append([minute_ms, bal["USDT"]["total"], broker.state["cash"], len(broker.state["positions"])])
        if (minute_ms - start_ms) % (24 * 3600_000) == 0:
            write_summary(False, minute_ms)
            print(f"{time.strftime('%Y-%m-%d', time.gmtime(minute_ms / 1000))} cash {broker.state['cash']:.2f} "
                  f"open {len(broker.state['positions'])} elapsed {time.perf_counter() - began:.0f}s", flush=True)
    return write_summary(True, minute_ms)


def prepare_run_dir(name: str, data_dir: Path = DATA_DIR, runs_dir: Path | None = None) -> Path:
    base = (runs_dir or data_dir / "runs").resolve()
    run_dir = (base / name).resolve()
    if base not in run_dir.parents:
        raise ValueError("run name must stay inside the runs folder")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "runtime").mkdir(parents=True)
    return run_dir
