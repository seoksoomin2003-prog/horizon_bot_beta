"""Simulated exchange that stands in for the ccxt client when TRADING_MODE is not live.

Only an allowlist of public market-data attributes reaches a keyless ccxt client. Every account or order call is
answered by PaperBroker, and any other attribute raises PaperBlocked, so a new private call fails instead of leaking.

Fill model (assumptions, not measurements):
- Market orders fill at the live ask or bid, plus PAPER_SLIPPAGE_BPS.
- Stops trigger on 1-minute last-price bars, not the mark price. They fill at the trigger or a worse gap open, plus
  slippage. Take-profits fill at the trigger, minus slippage. If both trigger in one bar, the stop goes first.
- Resting orders keep working while the program is off, as exchange-side orders do: missed bars are replayed.
- Futures: a position is liquidated when a bar opens past its liquidation price, or crosses it with no stop in front.
  Isolated-margin approximation with a 0.5% maintenance rate; the whole margin is lost at the bankruptcy price.
- Spot: no leverage and no short selling. Prices still come from the futures feed the strategy reads.
- Not modelled: funding, partial fills, order book depth, rejections for exchange rate limits.
"""

from __future__ import annotations

import json
import os
import math
import threading
import time
from decimal import Decimal
from typing import Any, Callable, Optional

import ccxt

from paper.settings import liquidation_price
from utils.persistence import atomic_write_json, get_absolute_path

STOP_TYPES = {"STOP_MARKET", "STOP"}
TP_TYPES = {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}
MINUTE_MS = 60_000
MAX_REPLAY_BARS = 1000
RECENT_LIMIT = 500
AMOUNT_DECIMALS = 10


class PaperBlocked(AttributeError):
    """A private or unknown exchange attribute was requested in paper mode."""


class PaperBroker:
    """Account, position and order book of the simulation. All mutation happens under one lock and is persisted."""

    def __init__(self, state_path: str, *, initial: float = 1000.0, fee: float = 0.0005, slippage_bps: float = 5.0,
                 leverage: float = 3.0, match_interval_s: float = 3.0, market: str = "futures", min_order: bool = False):
        self.state_path = state_path
        self.ledger_path = os.path.join(os.path.dirname(state_path), "paper_trades.jsonl")
        self.fee = fee
        self.slip = slippage_bps / 10_000.0
        self.market = market
        self.leverage = 1.0 if market == "spot" else leverage
        self.min_order = min_order
        self.match_interval_s = match_interval_s
        self.lock = threading.RLock()
        self._net_lock = threading.Lock()   # ccxt clients are not shared across threads without it
        self.public: Any = None
        self._last_match: dict[str, float] = {}
        self._marks: dict[str, float] = {}
        self.state = self._load(initial)

    def _load(self, initial: float) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            for pos in state["positions"].values():
                pos["amount"] = round(pos["amount"], AMOUNT_DECIMALS)
            return state
        return {"initial": initial, "cash": initial, "realized": 0.0, "fees": 0.0, "seq": 0, "positions": {},
                "orders": {}, "done": {}, "trades": [], "last_checked": {},
                "settings": {"market": self.market, "leverage": self.leverage, "fee": self.fee, "min_order": self.min_order}}

    def _save(self) -> None:
        # Last seen prices of open positions, so the dashboard can value them without a network connection.
        self.state["marks"] = {s: self._marks[s] for s in self.state["positions"] if s in self._marks}
        self.state["marks_ms"] = self._now_ms()
        atomic_write_json(self.state_path, self.state)

    def _next_id(self, prefix: str) -> str:
        self.state["seq"] += 1
        return f"{prefix}{self.state['seq']}"

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _quote(self, symbol: str) -> tuple[float, float, float]:
        with self._net_lock:
            t = self.public.fetch_ticker(symbol)
        last = float(t.get("last") or 0)
        if last <= 0:
            raise ccxt.ExchangeError(f"paper: no price for {symbol}")
        self._marks[symbol] = last
        return float(t.get("bid") or last), float(t.get("ask") or last), last

    def _order_view(self, o: dict) -> dict:
        return {"id": o["id"], "clientOrderId": o.get("client_id"), "symbol": o["symbol"], "type": o["type"],
                "side": o["side"], "amount": o["amount"], "filled": o.get("filled", 0.0),
                "remaining": o["amount"] - o.get("filled", 0.0), "price": o.get("fill_px"),
                "average": o.get("fill_px"), "stopPrice": o.get("trigger"), "triggerPrice": o.get("trigger"),
                "status": o["status"], "reduceOnly": o.get("reduce_only", False), "timestamp": o["created_ms"],
                "info": {"paper": True}}

    def _finish(self, order: dict, status: str) -> None:
        order["status"] = status
        self.state["orders"].pop(order["id"], None)
        self.state["done"][order["id"]] = order
        if len(self.state["done"]) > RECENT_LIMIT:
            self.state["done"].pop(next(iter(self.state["done"])))

    def _fill(self, order: dict, price: float, now_ms: int, reason: str) -> float:
        """Applies a one-way net position fill. Returns the filled amount."""
        symbol, side = order["symbol"], order["side"]
        qty = float(order["amount"])
        pos = self.state["positions"].get(symbol)
        realized = 0.0
        filled = 0.0
        if pos and (pos["side"] == "LONG") != (side == "buy"):
            closed = min(qty, pos["amount"])
            direction = 1.0 if pos["side"] == "LONG" else -1.0
            realized = direction * (price - pos["entry"]) * closed
            pos["amount"] = round(pos["amount"] - closed, AMOUNT_DECIMALS)   # a real exchange keeps exact sizes
            filled += closed
            qty = round(qty - closed, AMOUNT_DECIMALS)
            if pos["amount"] <= 1e-12:
                self.state["positions"].pop(symbol)
                pos = None
        if qty > 0 and not order.get("reduce_only"):
            pos = self.state["positions"].get(symbol)
            if pos:
                total = round(pos["amount"] + qty, AMOUNT_DECIMALS)
                pos["entry"] = (pos["entry"] * pos["amount"] + price * qty) / total
                pos["amount"] = total
            else:
                self.state["positions"][symbol] = {"side": "LONG" if side == "buy" else "SHORT", "amount": qty,
                                                   "entry": price, "opened_ms": now_ms}
                self.state["last_checked"].setdefault(symbol, now_ms)
            filled += qty
        fee = filled * price * self.fee
        self.state["cash"] += realized - fee
        self.state["realized"] += realized
        self.state["fees"] += fee
        order["filled"], order["fill_px"] = filled, price
        trade = {"id": self._next_id("T"), "order": order["id"], "symbol": symbol, "side": side, "price": price,
                 "amount": filled, "cost": filled * price, "fee": {"cost": fee, "currency": "USDT"},
                 "timestamp": now_ms, "realized": realized, "reason": reason, "cash_after": self.state["cash"]}
        self.state["trades"].append(trade)
        del self.state["trades"][:-RECENT_LIMIT]
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(trade, separators=(",", ":")) + "\n")
        return filled

    def create_order(self, symbol: str, type: str, side: str, amount: Any, price: Any = None,
                     params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        kind, side = str(type).upper(), str(side).lower()
        qty = float(amount)
        if qty <= 0:
            raise ccxt.InvalidOrder(f"paper: non-positive amount {amount}")
        reduce_only = bool(params.get("reduceOnly")) or str(params.get("tradeSide", "")).lower() == "close"
        client_id = params.get("newClientOrderId") or params.get("clientOrderId") or params.get("clientOid")
        if kind == "MARKET":
            bid, ask, last = self._quote(symbol)
            px = (ask if side == "buy" else bid) * (1 + self.slip if side == "buy" else 1 - self.slip)
            with self.lock:
                pos = self.state["positions"].get(symbol)
                closing = bool(pos) and (pos["side"] == "LONG") != (side == "buy")
                if reduce_only and not closing:
                    raise ccxt.InvalidOrder(f"paper: reduce-only order with no position to reduce on {symbol}")
                if self.market == "spot" and side == "sell" and not closing:
                    raise ccxt.InvalidOrder(f"paper: spot has no short selling ({symbol})")
                if not closing and self._free_locked() < qty * px / self.leverage:
                    raise ccxt.InsufficientFunds(f"paper: margin too low for {qty} {symbol}")
                now = self._now_ms()
                order = {"id": self._next_id("O"), "client_id": client_id, "symbol": symbol, "type": "MARKET",
                         "side": side, "amount": qty, "reduce_only": reduce_only, "created_ms": now}
                self._fill(order, px, now, "market")
                self._finish(order, "closed")
                self._save()
                return self._order_view(order)
        if kind in STOP_TYPES | TP_TYPES:
            trigger = params.get("stopPrice", params.get("triggerPrice"))
            if trigger is None or float(trigger) <= 0:
                raise ccxt.InvalidOrder(f"paper: {kind} needs a trigger price")
            with self.lock:
                order = {"id": self._next_id("A"), "client_id": client_id, "symbol": symbol, "type": kind, "side": side,
                         "amount": qty, "trigger": float(trigger), "reduce_only": True, "status": "open",
                         "created_ms": self._now_ms()}
                self.state["orders"][order["id"]] = order
                self.state["last_checked"].setdefault(symbol, order["created_ms"])
                self._save()
                return self._order_view(order)
        raise ccxt.NotSupported(f"paper: order type {type} is not simulated")

    def _trigger_price(self, order: dict, bar_open: float, high: float, low: float) -> Optional[float]:
        """Returns the fill price if this bar triggers the order, else None."""
        t, sell = order["trigger"], order["side"] == "sell"
        if order["type"] in STOP_TYPES:
            if sell and low <= t:
                return min(t, bar_open) * (1 - self.slip)
            if not sell and high >= t:
                return max(t, bar_open) * (1 + self.slip)
        else:
            if sell and high >= t:
                return t * (1 - self.slip)
            if not sell and low <= t:
                return t * (1 + self.slip)
        return None

    def _apply_trigger(self, order: dict, px: float, now_ms: int) -> None:
        pos = self.state["positions"].get(order["symbol"])
        if not pos or (pos["side"] == "LONG") == (order["side"] == "buy"):
            self._finish(order, "expired")
            return
        self._fill(order, px, now_ms, "stop" if order["type"] in STOP_TYPES else "take_profit")
        self._finish(order, "closed")

    def liquidation_price(self, pos: dict) -> float:
        return liquidation_price(pos["side"], pos["entry"], self.leverage)

    def _liquidate(self, symbol: str, pos: dict, now_ms: int) -> None:
        bankrupt = pos["entry"] * (1 - 1 / self.leverage if pos["side"] == "LONG" else 1 + 1 / self.leverage)
        order = {"id": self._next_id("L"), "client_id": None, "symbol": symbol, "type": "LIQUIDATION",
                 "side": "sell" if pos["side"] == "LONG" else "buy", "amount": pos["amount"], "reduce_only": True,
                 "created_ms": now_ms}
        self._fill(order, bankrupt, now_ms, "liquidation")
        self._finish(order, "closed")
        for other in [o for o in self.state["orders"].values() if o["symbol"] == symbol]:
            self._finish(other, "expired")

    def _check_liquidation(self, sym: str, t: int, o: float, h: float, low: float, now_ms: int) -> bool:
        pos = self.state["positions"].get(sym)
        if self.market != "futures" or not pos or t < pos["opened_ms"] // MINUTE_MS * MINUTE_MS + MINUTE_MS:
            return False
        liq, long = self.liquidation_price(pos), pos["side"] == "LONG"
        gapped = o <= liq if long else o >= liq
        crossed = low <= liq if long else h >= liq
        # A stop between the price and the liquidation level is reached first, unless the bar opens past both.
        guarded = any(x["symbol"] == sym and x["type"] in STOP_TYPES and (x["trigger"] > liq if long else x["trigger"] < liq)
                      for x in self.state["orders"].values())
        if gapped or (crossed and not guarded):
            self._liquidate(sym, pos, now_ms)
            return True
        return False

    def match(self, symbol: Optional[str] = None, *, force: bool = False) -> None:
        """Replays 1-minute bars since the last check against resting orders. Bars are fetched outside the lock."""
        with self.lock:
            symbols = sorted({o["symbol"] for o in self.state["orders"].values()} |
                             (set(self.state["positions"]) if self.market == "futures" else set()))
        if symbol is not None:
            symbols = [s for s in symbols if s == symbol]
        for sym in symbols:
            if not force and time.time() - self._last_match.get(sym, 0.0) < self.match_interval_s:
                continue
            with self.lock:
                since = self.state["last_checked"].get(sym, self._now_ms())
            with self._net_lock:
                bars = self.public.fetch_ohlcv(sym, "1m", since=int(since // MINUTE_MS * MINUTE_MS), limit=MAX_REPLAY_BARS)
            self._last_match[sym] = time.time()
            with self.lock:
                self._replay(sym, bars or [], since)

    def _replay(self, sym: str, bars: list, since: float) -> None:
        now = self._now_ms()
        changed = False
        for bar in bars:
            t, o, h, l = int(bar[0]), float(bar[1]), float(bar[2]), float(bar[3])
            if self._check_liquidation(sym, t, o, h, l, min(now, t + MINUTE_MS - 1)):
                changed = True
            live = sorted((x for x in self.state["orders"].values() if x["symbol"] == sym),
                          key=lambda x: (x["type"] not in STOP_TYPES, x["created_ms"]))
            for order in live:
                # The bar holding the placement also holds prices from before it, so it cannot trigger the order.
                if t < order["created_ms"] // MINUTE_MS * MINUTE_MS + MINUTE_MS:
                    continue
                px = self._trigger_price(order, o, h, l)
                if px is not None and order["id"] in self.state["orders"]:
                    # The current bar ends in the future; a fill cannot be dated later than now.
                    self._apply_trigger(order, px, min(now, t + MINUTE_MS - 1))
                    changed = True
            self._marks[sym] = float(bar[4])
        if bars:
            newest = int(bars[-1][0])
            self.state["last_checked"][sym] = max(float(since), float(newest))
            changed = True
        if changed:
            self._save()

    def cancel_order(self, order_id: Any, symbol: Optional[str] = None, params: Optional[dict] = None) -> dict:
        with self.lock:
            order = self.state["orders"].get(str(order_id))
            if order is None:
                raise ccxt.OrderNotFound(f"paper: order {order_id} is not open")
            self._finish(order, "canceled")
            self._save()
            return self._order_view(order)

    def cancel_all_orders(self, symbol: Optional[str] = None, params: Optional[dict] = None) -> list:
        with self.lock:
            out = []
            for order in list(self.state["orders"].values()):
                if symbol is None or order["symbol"] == symbol:
                    self._finish(order, "canceled")
                    out.append(self._order_view(order))
            self._save()
            return out

    def fetch_open_orders(self, symbol: Optional[str] = None, since=None, limit=None, params=None) -> list:
        self.match(symbol)
        with self.lock:
            return [self._order_view(o) for o in self.state["orders"].values() if symbol is None or o["symbol"] == symbol]

    def fetch_order(self, order_id: Any, symbol: Optional[str] = None, params: Optional[dict] = None) -> dict:
        self.match(symbol)
        with self.lock:
            order = self.state["orders"].get(str(order_id)) or self.state["done"].get(str(order_id))
            if order is None:
                raise ccxt.OrderNotFound(f"paper: unknown order {order_id}")
            return self._order_view(order)

    def fetch_orders(self, symbol: Optional[str] = None, since=None, limit=None, params=None) -> list:
        with self.lock:
            rows = list(self.state["done"].values()) + list(self.state["orders"].values())
            return [self._order_view(o) for o in rows if symbol is None or o["symbol"] == symbol]

    def fetch_my_trades(self, symbol: Optional[str] = None, since=None, limit=None, params=None) -> list:
        self.match(symbol)
        with self.lock:
            rows = [dict(t) for t in self.state["trades"] if symbol is None or t["symbol"] == symbol]
        return rows[-limit:] if limit else rows

    def _refresh_marks(self) -> None:
        with self.lock:
            symbols = list(self.state["positions"])
        for s in symbols:
            self._quote(s)

    def _unrealized_locked(self) -> float:
        total = 0.0
        for s, p in self.state["positions"].items():
            mark = self._marks.get(s, p["entry"])
            total += (1.0 if p["side"] == "LONG" else -1.0) * (mark - p["entry"]) * p["amount"]
        return total

    def _used_locked(self) -> float:
        return sum(p["amount"] * self._marks.get(s, p["entry"]) / self.leverage for s, p in self.state["positions"].items())

    def _free_locked(self) -> float:
        return self.state["cash"] + self._unrealized_locked() - self._used_locked()

    def fetch_balance(self, params: Optional[dict] = None) -> dict:
        self.match()
        self._refresh_marks()
        with self.lock:
            total = self.state["cash"] + self._unrealized_locked()
            used = self._used_locked()
            usdt = {"free": total - used, "used": used, "total": total}
            return {"USDT": usdt, "free": {"USDT": usdt["free"]}, "used": {"USDT": used}, "total": {"USDT": total},
                    "info": {"paper": True}}

    def fetch_positions(self, symbols=None, params=None) -> list:
        self.match()
        self._refresh_marks()
        with self.lock:
            out = []
            for s, p in self.state["positions"].items():
                mark = self._marks.get(s, p["entry"])
                sign = 1.0 if p["side"] == "LONG" else -1.0
                out.append({"symbol": s, "side": p["side"].lower(), "contracts": p["amount"], "entryPrice": p["entry"],
                            "markPrice": mark, "unrealizedPnl": sign * (mark - p["entry"]) * p["amount"],
                            "leverage": self.leverage, "timestamp": p["opened_ms"], "info": {"paper": True}})
            return out


_PUBLIC = frozenset({"load_markets", "markets", "market", "amount_to_precision", "price_to_precision", "fetch_ohlcv",
                     "fetch_ticker", "fetch_order_book", "load_time_difference", "milliseconds_offset", "milliseconds",
                     "symbols", "id", "has", "timeframes", "fetch_time", "iso8601", "parse8601"})
_ACCOUNT = frozenset({"cancel_order", "cancel_all_orders", "fetch_open_orders", "fetch_order", "fetch_orders",
                      "fetch_my_trades", "fetch_balance", "fetch_positions"})


class PaperExchange:
    """Per-thread facade. Public attributes go to a keyless client, account calls to the shared broker."""

    def __init__(self, exchange_name: str, broker: PaperBroker, public_factory: Optional[Callable[[], Any]] = None):
        factory = public_factory or (lambda: _public_client(exchange_name))
        object.__setattr__(self, "_public", factory())
        object.__setattr__(self, "_broker", broker)
        with broker.lock:
            if broker.public is None:
                broker.public = factory()
        if getattr(self._public, "apiKey", "") or getattr(self._public, "secret", ""):
            raise PaperBlocked("paper mode received a client with credentials")

    def __getattr__(self, name: str) -> Any:
        if name in _PUBLIC:
            return getattr(self._public, name)
        if name in _ACCOUNT:
            return getattr(self._broker, name)
        raise PaperBlocked(f"paper mode blocks exchange attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise PaperBlocked(f"paper mode blocks setting exchange attribute {name!r}")

    def create_order(self, symbol: str, type: str, side: str, amount: Any, price: Any = None,
                     params: Optional[dict] = None) -> dict:
        # Real ccxt clients round the amount here, and reject one below the market's step.
        return self._broker.create_order(symbol, type, side, self._public.amount_to_precision(symbol, amount), price, params)

    def set_position_mode(self, *args: Any, **kwargs: Any) -> dict:
        return {"paper": True}

    def fapiPrivateGetPositionSideDual(self, *args: Any, **kwargs: Any) -> dict:
        return {"dualSidePosition": False}


_shared_markets: Optional[tuple] = None
_markets_lock = threading.Lock()


def _public_client(exchange_name: str) -> Any:
    """One market download per process. A failure leaves the client to load lazily, and the next client retries."""
    global _shared_markets
    options = {"defaultType": "future"} if exchange_name == "binance" else {"defaultType": "swap"}
    client = getattr(ccxt, exchange_name)({"enableRateLimit": True, "timeout": 10_000, "options": options})
    with _markets_lock:
        if _shared_markets is None:
            try:
                client.load_markets()
                _shared_markets = (client.markets, client.currencies)
            except Exception:
                return client
            return client
        client.set_markets(*_shared_markets)
    return client


_broker: Optional[PaperBroker] = None
_broker_lock = threading.Lock()


def get_broker() -> PaperBroker:
    global _broker
    with _broker_lock:
        if _broker is None:
            import config
            _broker = PaperBroker(get_absolute_path("paper_broker.json"), initial=config.PAPER_INITIAL_EQUITY, fee=config.PAPER_FEE,
                                  slippage_bps=config.PAPER_SLIPPAGE_BPS, leverage=config.PAPER_LEVERAGE,
                                  market=config.PAPER_MARKET, min_order=config.PAPER_MIN_ORDER)
        return _broker


_spot_markets: Optional[dict] = None


def _spot_market(symbol: str) -> dict:
    """Spot rules (minimum size and amount step) from a keyless spot client, loaded once."""
    global _spot_markets
    with _markets_lock:
        if _spot_markets is None:
            client = ccxt.binance({"enableRateLimit": True, "timeout": 10_000, "options": {"defaultType": "spot"}})
            _spot_markets = client.load_markets()
    return _spot_markets[symbol.split(":")[0]]


def order_minimums(public: Any, symbol: str, market: str) -> tuple[float, float, float]:
    """(minimum amount, minimum notional in USDT, amount step) for the paper market in use."""
    m = _spot_market(symbol) if market == "spot" else public.market(symbol)
    limits = m.get("limits", {})
    return (float(limits.get("amount", {}).get("min") or 0.0), float(limits.get("cost", {}).get("min") or 0.0),
            float(m.get("precision", {}).get("amount") or 0.0))


def _ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    units = math.ceil(value / step - 1e-9)
    return float(Decimal(units) * Decimal(str(step)))


def fit_order(amount: float, price: float, stop: float, side: str, equity: float, used_margin: float, *,
              minimums: tuple[float, float, float], market: str, leverage: float, raise_to_minimum: bool) -> tuple[Optional[float], str]:
    """Applies the paper market's rules to a sized entry. Returns (amount or None to skip, explanation)."""
    if market == "spot" and side == "SHORT":
        return None, "spot has no short selling"
    if price <= 0 or equity <= 0:
        return None, "no price or equity"
    min_qty, min_cost, step = minimums
    # 1% above the minimum notional, so a small price move before the fill does not drop it below.
    need = max(min_qty, min_cost * 1.01 / price)
    note = ""
    if amount < need:
        if not raise_to_minimum:
            return None, f"below exchange minimum ({amount * price:.2f} < {min_cost:g} USDT)"
        raised = _ceil_to_step(need, step)
        note = f"raised to exchange minimum {amount:.8g} -> {raised:.8g} ({raised * price:.2f} USDT)"
        amount = raised
    lev = 1.0 if market == "spot" else float(leverage)
    notional = amount * price
    if notional > lev * equity:
        return None, f"order {notional:.2f} USDT is more than {lev:g}x equity {equity:.2f}"
    if notional / lev > equity - used_margin:
        return None, f"order needs {notional / lev:.2f} USDT margin, free {equity - used_margin:.2f}"
    if note:
        note += f", risk to stop {amount * abs(price - stop) / equity:.2%} of equity"
    return amount, note
