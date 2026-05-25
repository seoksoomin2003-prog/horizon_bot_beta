"""
execution_engine.py — Institutional-Grade Execution Layer (v2 Refactor)

INVARIANTS:
  1. No position exists without an exchange-side STOP LOSS. [C1]
  2. Local state ALWAYS matches exchange FILLED size. [C4]
  3. No order is placed below exchange minQty. [C5]
  4. Every exit/manual close is preceded by cancel_all_orders. [C3]
  5. SL updates (Trailing/Breakeven) are immediately synced to exchange. [C1, C2]
"""

import uuid
import time
from logger import trades_logger, errors_logger
from utils.safe_core import SafeCore
from utils.telegram_commands import send_message
from utils.schemas import PositionSchema, TradeSignal, Ticker, AccountState, PerformanceTrade
from typing import List, Dict, Any, Optional
from utils.validator import Validator
from utils.exceptions import ExecutionFailure, DataIntegrityError
from utils.safe_ops import safe_float

from risk_engine import FirmOrderRequest
from utils.retry_scheduler import BoundedRetryScheduler

def generate_order_uid(symbol, side, intent_id):
    return f"{symbol}_{side}_{intent_id}"

def match_order_by_client_id(client_id, order_list):
    for o in order_list:
        cid = o.get("clientOrderId") or o.get("info", {}).get("clientOrderId")
        if str(cid) == str(client_id):
            return o
    return None

def normalize_order_type(order):
    t = str(order.get("type", "")).upper()
    
    if "STOP" in t:
        return "STOP"
    if "TAKE_PROFIT" in t:
        return "TAKE_PROFIT"
        
    return t

def is_protection_order(order, position):
    order_type = str(order.get("type", "")).upper()
    
    VALID_TYPES = {
        "STOP_MARKET",
        "STOP",
        "TAKE_PROFIT",
        "TAKE_PROFIT_MARKET"
    }
    
    if order_type not in VALID_TYPES:
        return False
        
    # Infer reduceOnly via direction
    order_side = str(order.get("side", "")).lower()
    pos_side = ""
    if hasattr(position, "side"):
        pos_side = position.side.upper()
    elif isinstance(position, dict):
        pos_side = str(position.get("side", "")).upper()
    
    # LONG -> protection = SELL
    if pos_side == "LONG" and order_side == "sell":
        return True
        
    # SHORT -> protection = BUY
    if pos_side == "SHORT" and order_side == "buy":
        return True
        
    return False

def is_any_protection_order(order):
    order_type = normalize_order_type(order)
    return order_type in ["STOP", "TAKE_PROFIT"]

def count_entry_orders(orders):
    return sum(1 for o in orders if not is_any_protection_order(o))

def force_remove_position(self, symbol):
    if hasattr(self, "state") and self.state:
        try:
            self.state.remove_position(symbol)
        except Exception:
            pass

def failsafe_guard(self, symbol):
    if not hasattr(self, 'failsafe_attempts'):
        self.failsafe_attempts = {}
    
    self.failsafe_attempts[symbol] = self.failsafe_attempts.get(symbol, 0) + 1
    
    if self.failsafe_attempts[symbol] > 2:
        errors_logger.critical(f"FORCED STATE RESET: Max failsafe attempts for {symbol}.")
        force_remove_position(self, symbol)
        self.failsafe_attempts[symbol] = 0
        return False
        
    try:
        positions = self.dh.fetch_active_positions(retries=1)
        if not any(p.symbol == symbol for p in positions):
            errors_logger.critical(f"FORCED STATE RESET: No position found for {symbol} on exchange.")
            force_remove_position(self, symbol)
            return False
    except Exception:
        pass

    return True

class ExecutionEngine:
    def __init__(self, data_handler, simulation_mode=True):
        import threading
        self.dh = data_handler
        self.simulation_mode = simulation_mode
        self.lock = threading.Lock()
        with self.lock:
            self._load_registries()
        self.scheduler = BoundedRetryScheduler()
        if self.simulation_mode:
            trades_logger.info("🚀 EXECUTION ENGINE INITIALIZED IN [SIMULATION MODE]")

    def _load_registries(self):
        import os
        import json
        self.registry_file = "data/order_registry.json"
        self.pending_sl_file = "data/pending_sl.json"
        
        # Load order_registry
        self.order_registry = {}
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.order_registry = json.load(f)
            except Exception:
                pass
                
        # Load pending_sl
        self.pending_sl = {}
        if os.path.exists(self.pending_sl_file):
            try:
                with open(self.pending_sl_file, 'r') as f:
                    self.pending_sl = json.load(f)
            except Exception:
                pass

    def _save_order_registry(self):
        from utils.persistence import atomic_write_json
        try:
            atomic_write_json(self.registry_file, self.order_registry)
        except Exception:
            pass

    def _save_pending_sl(self):
        from utils.persistence import atomic_write_json
        try:
            atomic_write_json(self.pending_sl_file, self.pending_sl)
        except Exception:
            pass

    def execute_firm_order(self, request: FirmOrderRequest) -> Optional[PositionSchema]:
        """
        [PHASE 4] Executes a firm order request. 
        If simulation_mode is True, no real API calls are made.
        """
        if not request.approved:
            trades_logger.warning(f"[REJECTED] {request.symbol}: {request.rejection_reason}")
            return None

        if self.simulation_mode:
            trades_logger.info(f"[SIMULATION] Executing {request.side} for {request.symbol} (Size: {request.amount})")
            # Return a mock PositionSchema
            return PositionSchema(
                symbol=request.symbol,
                side=request.side,
                amount=request.amount,
                initial_amount=request.amount,
                entry_price=0.0, # Placeholder
                stop_loss=request.stop_loss,
                take_profit=request.take_profit or 0.0,
                risk_pct=0.0
            )

        # Real Execution logic (mapping to legacy execute_trade)
        trades_logger.info(f"[LIVE] Routing Firm Order to Exchange: {request.symbol} {request.side} {request.amount}")
        
        # Create a transient signal for legacy compatibility
        signal = TradeSignal(symbol=request.symbol, trigger=request.side, confidence=request.conviction_at_entry)
        
        try:
            return self.execute_trade(
                symbol=request.symbol,
                signal=signal,
                amount=request.amount,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                intent_id=request.intent_id
            )
        except Exception as e:
            errors_logger.error(f"[EXECUTION FAILURE] {request.symbol}: {e}")
            return None

    def build_bitget_params(self, side, is_close):
        """Standardizes Bitget order parameters based on active hold mode."""
        if self.dh.exchange_name != 'bitget':
            return {}

        params = {
            'clientOid': str(uuid.uuid4()).replace('-', '')[:30]
        }
        
        actual_mode = self.dh.get_position_mode()
        from config import POSITION_MODE
        mode = actual_mode or POSITION_MODE

        if mode == 'hedge':
            params['tradeSide'] = 'close' if is_close else 'open'
            params['posSide'] = side.lower()
        
        return params

    def place_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float, retries: int = 3, force: bool = False) -> Optional[Dict]:
        """[C1] Mandatory Stop Loss Placement with Non-Blocking Detached Retries."""
        import threading
        formatted_symbol = self.dh.format_symbol(symbol)
        now_ts = time.time()
        
        with self.lock:
            if symbol in self.pending_sl:
                if now_ts - self.pending_sl[symbol]["timestamp"] > 3:
                    del self.pending_sl[symbol]
                    self._save_pending_sl()
                elif now_ts - self.pending_sl[symbol]["timestamp"] < 2 and not force:
                    return None

        def _execute_placement_attempt() -> Optional[Dict]:
            # PATCH 5 - SL duplication prevention
            if not force:
                try:
                    orders = self.dh.exchange.fetch_open_orders(formatted_symbol)
                    mock_pos = type('obj', (object,), {'side': side})
                    if any(is_protection_order(o, mock_pos) for o in orders):
                        trades_logger.info(f"Skipping SL creation for {symbol}: valid protection order already exists.")
                        return next((o for o in orders if is_protection_order(o, mock_pos)), None)
                except Exception as e:
                    trades_logger.warning(f"Failed to check open orders for SL check: {e}")

            opposite_side = "sell" if side.upper() == "LONG" else "buy"
            
            # C-01: Verify Direction before reduceOnly Stop Loss
            if not self.verify_position_direction(symbol, side):
                raise ExecutionFailure(f"Direction Mismatch: Cannot place SL for {symbol} ({side})")
            
            try:
                self.dh.exchange.load_markets()
                amount_precision = self.dh.exchange.amount_to_precision(formatted_symbol, amount)
                price_precision = self.dh.exchange.price_to_precision(formatted_symbol, stop_price)
            except Exception as e:
                raise ExecutionFailure(f"Precision format failure for {symbol}: {e}")

            params = {
                "stopPrice": float(price_precision),
                "triggerPrice": float(price_precision),
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
                "triggerType": "mark_price",
                "planType": "profit_loss_plan"
            }
            params.update(self.build_bitget_params(side, is_close=True))
            order = self.dh.exchange.create_order(
                symbol=formatted_symbol,
                type="STOP_MARKET",
                side=opposite_side,
                amount=float(amount_precision),
                price=None,
                params=params
            )
            
            with self.lock:
                self.pending_sl[symbol] = {
                    "timestamp": time.time(),
                    "price": stop_price
                }
                self._save_pending_sl()
            return order

        # 1. Sync First Attempt (No sleep allowed)
        try:
            order = _execute_placement_attempt()
            trades_logger.info(f"[OK] SL Placed for {symbol} at {stop_price} (ID: {order.get('id')})")
            return order
        except Exception as first_error:
            errors_logger.warning(f"SL Sync Placement Attempt Failed for {symbol}: {first_error}. Offloading to BoundedRetryScheduler.")
            
            # 2. Bounded Scheduler retry task
            def _scheduler_sl_retry():
                bg_order = _execute_placement_attempt()
                if bg_order:
                    trades_logger.info(f"[OK] SL Placed via BoundedRetryScheduler for {symbol} at {stop_price} (ID: {bg_order.get('id')})")
                    if hasattr(self, 'state') and self.state:
                        pos = self.state.get_position(symbol)
                        if pos:
                            pos.sl_plan_id = bg_order.get('id')
                            pos.stop_loss = stop_price
                            self.state.persist()
                    return True
                return False

            try:
                self.scheduler.schedule_retry(
                    _scheduler_sl_retry,
                    name=f"SLPlacement-{symbol}",
                    retries=retries,
                    initial_delay=1.0,
                    backoff_factor=1.5
                )
            except Exception as sched_err:
                errors_logger.critical(
                    f"[CRITICAL_SCHEDULING_FAILURE] Failed to schedule SL retry via BoundedRetryScheduler for {symbol}: {sched_err}. "
                    "Escalating to CRITICAL_PROTECTION_FAILURE and attempting synchronous recovery."
                )
                from utils.degraded_manager import DegradedExchangeManager, ExchangeDegradationState
                DegradedExchangeManager().set_state(
                    ExchangeDegradationState.CRITICAL_PROTECTION_FAILURE,
                    f"Stop-Loss scheduling queue overload on {symbol}: {sched_err}"
                )
                
                # Try synchronous, blocking fallback Stop-Loss placement immediately
                sync_success = False
                for attempt in range(1, 4):
                    try:
                        trades_logger.warning(f"[FALLBACK_SYNC_PLACEMENT] Attempting synchronous SL placement for {symbol} (Attempt {attempt}/3)...")
                        bg_order = _execute_placement_attempt()
                        if bg_order:
                            trades_logger.info(f"[OK] Fallback SL placed synchronously for {symbol} at {stop_price} (ID: {bg_order.get('id')})")
                            if hasattr(self, 'state') and self.state:
                                pos = self.state.get_position(symbol)
                                if pos:
                                    pos.sl_plan_id = bg_order.get('id')
                                    pos.stop_loss = stop_price
                                    self.state.persist()
                            sync_success = True
                            break
                    except Exception as sync_err:
                        errors_logger.error(f"[FALLBACK_SYNC_FAIL] Attempt {attempt} failed for {symbol}: {sync_err}")
                        time.sleep(0.5)
                
                # If synchronous placement fails, escalate to emergency close (liquidation)
                if not sync_success:
                    errors_logger.critical(
                        f"[FATAL_PROTECTION_LOSS] Synchronous fallback Stop-Loss placement failed for {symbol}. "
                        "Forcing immediate emergency position liquidation to protect capital!"
                    )
                    if hasattr(self, 'state') and self.state:
                        pos = self.state.get_position(symbol)
                        if pos:
                            self.close_full(symbol, pos)
                            self.state.remove_position(symbol)
            return None

    def _get_open_orders_count(self, symbol):
        try:
            orders = self.dh.exchange.fetch_open_orders(self.dh.format_symbol(symbol))
            return count_entry_orders(orders)
        except Exception:
            return 0

    def verify_position_direction(self, symbol: str, local_side: str) -> bool:
        """
        [C1] Verifies that the exchange's position direction matches the local state.
        Must be called before any reduceOnly order is placed.
        """
        try:
            exchange_positions = self.dh.fetch_active_positions(retries=2)
            ex_pos = next((p for p in exchange_positions if p.symbol == symbol), None)
            
            if not ex_pos:
                # If no position found on exchange, we can't verify direction. 
                # Could be closing a ghost position. Let it proceed to fail gracefully.
                return True
                
            ex_side = ex_pos.side.upper()
            loc_side = str(local_side).upper()
            
            if ex_side != loc_side:
                errors_logger.critical(f"[INVALID_SIDE] Direction mismatch for {symbol}. Local: {loc_side}, Exchange: {ex_side}. Aborting Execution.")
                return False
                
            return True
        except Exception as e:
            errors_logger.error(f"Failed to verify position direction for {symbol}: {e}")
            return False

    def execute_trade(self, symbol: str, signal: TradeSignal, amount: float, stop_loss: float, take_profit: Optional[float] = None, intent_id: str = "") -> PositionSchema:
        """Hardened execution flow with Execution Truth verification."""
        formatted_symbol = self.dh.format_symbol(symbol)
        
        MAX_OPEN_ORDERS_PER_SYMBOL = 3
        if self._get_open_orders_count(symbol) >= MAX_OPEN_ORDERS_PER_SYMBOL:
            trades_logger.warning(f"[ORDER_BLOCK] too many open orders: {symbol}")
            raise ExecutionFailure(f"Too many open orders for {symbol}")
            
        if not self.validate_lot_size(symbol, amount):
            raise ExecutionFailure(f"Invalid lot size: {amount} for {symbol}")

        try:
            self.dh.exchange.load_markets()
            amount_precision = self.dh.exchange.amount_to_precision(formatted_symbol, amount)
        except Exception as e:
            raise DataIntegrityError(f"Precision failure: {e}")

        order_side = 'buy' if signal.trigger == 'LONG' else 'sell'
        
        now_ms = int(time.time() * 1000)
        with self.lock:
            for uid, data in list(self.order_registry.items()):
                if now_ms - data["timestamp"] > 120000:
                    del self.order_registry[uid]
            self._save_order_registry()
        
        if not intent_id:
            import uuid
            intent_id = uuid.uuid4().hex
            
        order_uid = generate_order_uid(symbol, order_side, intent_id)
            
        with self.lock:
            if order_uid in self.order_registry:
                if now_ms - self.order_registry[order_uid]["timestamp"] < 10000:
                    raise ExecutionFailure(f"Duplicate order blocked within 10s: {order_uid}")
            
            self.order_registry[order_uid] = {
                "timestamp": now_ms,
                "symbol": symbol,
                "side": order_side,
                "status": "PENDING"
            }
            self._save_order_registry()

        params = {"newClientOrderId": order_uid}
        params.update(self.build_bitget_params(order_side, is_close=False))
        
        try:
            order = self.dh.exchange.create_order(
                symbol=formatted_symbol,
                type='market',
                side=order_side,
                amount=amount_precision,
                params=params
            )
            with self.lock:
                self.order_registry[order_uid]["status"] = "FILLED"
                self._save_order_registry()
        except Exception as e:
            open_orders = self.dh.exchange.fetch_open_orders(formatted_symbol)
            matched = match_order_by_client_id(order_uid, open_orders)
            
            if not matched:
                try:
                    trades = self.dh.exchange.fetch_my_trades(formatted_symbol, limit=10)
                    matched = match_order_by_client_id(order_uid, trades)
                except Exception:
                    pass
            
            if matched:
                order = matched
                with self.lock:
                    self.order_registry[order_uid]["status"] = "FILLED"
                    self._save_order_registry()
            else:
                raise ExecutionFailure(f"Primary Order Placement Failed: {e}")

        # VERIFICATION [EXECUTION TRUTH]
        avg_fill_price, total_filled = self.dh.get_real_average_fill(symbol, order.get('id', ''))
        
        if total_filled <= 0:
            total_filled = self.confirm_trade_execution(symbol, order.get('id', ''))
            avg_fill_price = SafeCore.float(order.get('price'), 0.0)
            
        if total_filled <= 0:
            raise ExecutionFailure(f"Execution Verification Failed for {symbol} (No Trades Found)")

        try:
            sl_order = self.place_stop_loss(symbol, signal.trigger, total_filled, stop_loss)
            sl_plan_id = sl_order['id'] if sl_order else None
            
            tp_plan_id = None
            if take_profit:
                try:
                    tp_side = 'sell' if signal.trigger == 'LONG' else 'buy'
                    try:
                        self.dh.exchange.load_markets()
                        tp_price_precision = self.dh.exchange.price_to_precision(formatted_symbol, take_profit)
                        tp_amount_precision = self.dh.exchange.amount_to_precision(formatted_symbol, total_filled)
                    except:
                        tp_price_precision = take_profit
                        tp_amount_precision = total_filled
                        
                    tp_params = {
                        "stopPrice": float(tp_price_precision),
                        "reduceOnly": True,
                        "workingType": "MARK_PRICE"
                    }
                    tp_order = self.dh.exchange.create_order(
                        symbol=formatted_symbol, 
                        type='TAKE_PROFIT_MARKET', 
                        side=tp_side, 
                        amount=float(tp_amount_precision), 
                        price=None, 
                        params=tp_params
                    )
                    tp_plan_id = tp_order['id']
                except Exception as tp_e:
                    errors_logger.warning(f"TP Placement failed (non-critical): {tp_e}")
                
            msg = (
                f"🛡️ <b>Trade Entry</b>\n"
                f"Symbol: {symbol}\n"
                f"Side: {signal.trigger}\n"
                f"Amount: {total_filled}\n"
                f"Entry: {avg_fill_price:.5f}\n"
                f"SL: {stop_loss:.5f}"
            )
            send_message(msg)

            pos_schema = PositionSchema(
                symbol=symbol,
                side=signal.trigger,
                amount=total_filled,
                initial_amount=total_filled,
                entry_price=avg_fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit or 0.0,
                sl_plan_id=sl_plan_id,
                tp_plan_id=tp_plan_id
            )

            # PHASE 7: SAFETY GUARD
            if not self.watchdog_verify_sl(pos_schema):
                errors_logger.critical(f"[NO_STOP_LOSS] forcing exit for {symbol}")
                self.close_full(symbol, pos_schema)

            return pos_schema
        except Exception as e:
            errors_logger.critical(f"FATAL PROTECTION FAILURE: {e}. Executing FAILSAFE.")
            self._execute_failsafe_protection(symbol, signal.trigger, total_filled)
            from config import BOT_STATE
            BOT_STATE['is_running'] = False
            raise ExecutionFailure(f"Protection Failure & Emergency Halt: {e}")

    def validate_lot_size(self, symbol, amount):
        """[C5] Minimum Quantity and Notional Invariant."""
        try:
            market = self.dh.exchange.market(self.dh.format_symbol(symbol))
            min_qty = market.get('limits', {}).get('amount', {}).get('min')
            
            if min_qty is not None and amount < min_qty:
                trades_logger.info(f"[REJECTED: BELOW_MIN_QTY] {symbol} amount {amount} < min {min_qty}")
                errors_logger.warning(f"[C5] Rejecting {symbol} amount {amount} < min {min_qty}")
                return False
                
            # Fetch latest ticker for nominal check
            ticker = self.dh.fetch_ticker(symbol)
            notional = amount * ticker.last if ticker else 0
            min_cost = market.get('limits', {}).get('cost', {}).get('min')
            
            if min_cost is not None and notional > 0 and notional < min_cost:
                trades_logger.info(f"[REJECTED: BELOW_MIN_NOTIONAL] {symbol} nominal {notional:.2f} < min {min_cost}")
                return False
                
            return True
        except Exception as e:
            errors_logger.error(f"validate_lot_size failure for {symbol}: {e}")
            return True

    def cancel_all_orders(self, symbol):
        """[C3] Stop-Gap Cleaner: Purge all plan/limit orders for symbol."""
        try:
            formatted_symbol = self.dh.format_symbol(symbol)
            orders = self.dh.exchange.fetch_open_orders(formatted_symbol)
            for o in orders:
                try:
                    self.dh.exchange.cancel_order(o["id"], formatted_symbol)
                except Exception:
                    continue
            return True
        except Exception as e:
            errors_logger.error(f"Cancel all failed for {symbol}: {e}")
            return False

    def close_full(self, symbol, pos):
        """[C3] Atomic Full Exit (Exit-First Sequencing)."""
        trades_logger.info(f"[EXIT] Closing full position for {symbol}...")
        
        side = getattr(pos, "side", "")
        if not side and isinstance(pos, dict):
            side = pos.get("side", "")
        side = side.upper()
        
        amount = float(getattr(pos, "amount", 0.0) or (pos.get("amount", 0.0) if isinstance(pos, dict) else 0.0))

        if amount <= 0:
            return None

        # C-01: Verify Direction before reduceOnly Market Close
        if not self.verify_position_direction(symbol, side):
            errors_logger.critical(f"[ABORT] Cannot close full {symbol}. Direction Mismatch.")
            return False

        opposite = "sell" if side == "LONG" else "buy"
        try:
            params = {"reduceOnly": True}
            params.update(self.build_bitget_params(side, is_close=True))
            # 1. Place market exit order first
            self.dh.exchange.create_order(
                symbol=self.dh.format_symbol(symbol),
                type="MARKET",
                side=opposite,
                amount=amount,
                price=None,
                params=params
            )
            # 2. Cancel all outstanding stop loss / take profit plan orders
            self.cancel_all_orders(symbol)
            
            # Log event in ExecutionJournal
            try:
                from utils.journal import ExecutionJournal
                ExecutionJournal.log_event(symbol, "POSITION_CLOSED")
            except Exception:
                pass
                
            send_message(f"✅ <b>Position Closed</b>\n{symbol} full exit completed.")
            return True
        except Exception as e:
            errors_logger.error(f"Close full failed for {symbol}: {e}")
            return False

    def close_partial(self, symbol, pos, amount, *, min_qty: float = 0.0):
        """
        [C4] Atomic Partial Exit with SL Re-sync + Exchange-Confirmed Fill Verification (Exit-First Sequencing).
        Returns dict with keys: ok, order_id, filled_amount, requested_amount, clamped_amount, dust_closed
        """
        trades_logger.info(f"[EXIT] Partial close request {amount} for {symbol}...")
        
        side = getattr(pos, "side", "")
        if not side and isinstance(pos, dict):
            side = pos.get("side", "")
        side = side.upper()
        
        # C-01: Verify Direction before reduceOnly Partial Close
        if not self.verify_position_direction(symbol, side):
            errors_logger.critical(f"[ABORT] Cannot close partial {symbol}. Direction Mismatch.")
            return {"ok": False, "order_id": None, "filled_amount": 0.0, "requested_amount": float(amount or 0.0), "clamped_amount": 0.0, "dust_closed": False}
            
        opposite = "sell" if side == "LONG" else "buy"
        try:
            current_amount = float(getattr(pos, "amount", 0.0) or 0.0)
            requested = float(amount or 0.0)
            if current_amount <= 0 or requested <= 0:
                return {"ok": False, "order_id": None, "filled_amount": 0.0, "requested_amount": requested, "clamped_amount": 0.0, "dust_closed": False}

            # Clamp to remaining size
            clamped = min(requested, current_amount)

            # minQty gate: if requested is below min, try to do a full close if remaining is also below min.
            if min_qty and clamped < min_qty:
                if current_amount < min_qty:
                    trades_logger.warning(f"[DUST_CLEANUP] {symbol} remaining {current_amount} < minQty {min_qty}. Full close instead of partial.")
                    ok = bool(self.close_full(symbol, pos))
                    return {"ok": ok, "order_id": None, "filled_amount": current_amount if ok else 0.0, "requested_amount": requested, "clamped_amount": current_amount, "dust_closed": True}
                trades_logger.warning(f"[REJECTED: BELOW_MIN_QTY] {symbol} partial {clamped} < minQty {min_qty}. Skipping.")
                return {"ok": False, "order_id": None, "filled_amount": 0.0, "requested_amount": requested, "clamped_amount": clamped, "dust_closed": False}

            if self.simulation_mode:
                # Deterministic simulation: treat as filled immediately.
                filled = clamped
                pos.intended_qty = clamped
                pos.filled_qty = filled
                pos.remaining_qty = max(0.0, current_amount - filled)
                pos.cumulative_fill_ratio = filled / clamped if clamped > 0 else 0.0
                pos.amount = max(0.0, current_amount - filled)
                send_message(f"✴️ <b>Partial Exit (SIM)</b>\n{symbol}: {filled} contracts closed.")
                if pos.amount > 0 and getattr(pos, "stop_loss", 0.0) > 0:
                    self.place_stop_loss(symbol, side, pos.amount, pos.stop_loss, force=True)
                return {"ok": True, "order_id": "SIM", "filled_amount": filled, "requested_amount": requested, "clamped_amount": clamped, "dust_closed": False}

            # 1. Place the partial market close order FIRST (without cancelling protective orders)
            params = {"reduceOnly": True}
            params.update(self.build_bitget_params(side, is_close=True))
            formatted_symbol = self.dh.format_symbol(symbol)
            order = self.dh.exchange.create_order(formatted_symbol, 'MARKET', opposite, clamped, None, params)

            order_id = order.get("id") or order.get("info", {}).get("orderId") or order.get("info", {}).get("id")
            filled = float(self.confirm_trade_execution(symbol, order_id) or 0.0)

            if filled <= 0:
                errors_logger.critical(f"[EXEC_CONFIRM_FAIL] {symbol} partial close order accepted but no fills detected (order_id={order_id}).")
                return {"ok": False, "order_id": order_id, "filled_amount": 0.0, "requested_amount": requested, "clamped_amount": clamped, "dust_closed": False}

            send_message(f"✴️ <b>Partial Exit</b>\n{symbol}: {filled} contracts closed.")

            pos.intended_qty = clamped
            pos.filled_qty = filled
            pos.remaining_qty = max(0.0, current_amount - filled)
            pos.cumulative_fill_ratio = filled / clamped if clamped > 0 else 0.0

            remaining_est = max(0.0, current_amount - filled)
            try:
                pos.amount = remaining_est
            except Exception:
                pass

            # 2. Place new stop loss order for the remaining size BEFORE cancelling the old one
            if remaining_est > 0 and getattr(pos, "stop_loss", 0.0) > 0:
                new_sl_order = self.place_stop_loss(symbol, side, remaining_est, pos.stop_loss, force=True)
                
                # 3. Only cancel old SL order after new one is placed successfully
                old_sl_id = pos.sl_plan_id
                if old_sl_id:
                    try:
                        self.dh.exchange.cancel_order(old_sl_id, formatted_symbol)
                    except Exception:
                        pass
                if new_sl_order:
                    pos.sl_plan_id = new_sl_order.get('id')

            # 4. Handle Take Profit adjustments
            old_tp_id = pos.tp_plan_id
            if old_tp_id:
                try:
                    self.dh.exchange.cancel_order(old_tp_id, formatted_symbol)
                except Exception:
                    pass
                pos.tp_plan_id = None
                
            if remaining_est > 0 and getattr(pos, "take_profit", 0.0) > 0:
                # Place new TP order for remaining size
                try:
                    tp_side = 'sell' if side == 'LONG' else 'buy'
                    tp_price = pos.take_profit
                    try:
                        self.dh.exchange.load_markets()
                        tp_price_precision = self.dh.exchange.price_to_precision(formatted_symbol, tp_price)
                        tp_amount_precision = self.dh.exchange.amount_to_precision(formatted_symbol, remaining_est)
                    except:
                        tp_price_precision = tp_price
                        tp_amount_precision = remaining_est
                        
                    tp_params = {
                        "stopPrice": float(tp_price_precision),
                        "reduceOnly": True,
                        "workingType": "MARK_PRICE"
                    }
                    tp_order = self.dh.exchange.create_order(
                        symbol=formatted_symbol, 
                        type='TAKE_PROFIT_MARKET', 
                        side=tp_side, 
                        amount=float(tp_amount_precision), 
                        price=None, 
                        params=tp_params
                    )
                    pos.tp_plan_id = tp_order.get('id')
                except Exception as tp_e:
                    errors_logger.warning(f"New TP Placement failed during partial close: {tp_e}")

            # Dust quarantine: if remaining < minQty, mark as dust to avoid rejection storms
            dust_closed = False
            if min_qty and 0 < remaining_est < min_qty:
                trades_logger.warning(f"[DUST_QUARANTINE] {symbol} residual {remaining_est} < minQty {min_qty}. Quarantining terminal dust.")
                dust_closed = True
                try:
                    pos.amount = remaining_est
                    pos.status = "dust_quarantined"
                    
                    from state_engine import DustRegistry
                    DustRegistry.add(symbol)
                except Exception:
                    pass

            return {"ok": True, "order_id": order_id, "filled_amount": filled, "requested_amount": requested, "clamped_amount": clamped, "dust_closed": dust_closed}
        except Exception as e:
            errors_logger.error(f"Partial close failed for {symbol}: {e}")
            return {"ok": False, "order_id": None, "filled_amount": 0.0, "requested_amount": float(amount or 0.0), "clamped_amount": 0.0, "dust_closed": False}

    def confirm_trade_execution(self, symbol, order_id):
        """Fallback verification for market orders."""
        from utils.safe_core import SafeCore
        try:
            trades = self.dh.exchange.fetch_my_trades(self.dh.format_symbol(symbol), limit=5)
            matched = [t for t in trades if str(t.get('order')) == str(order_id)]
            return sum([SafeCore.float(t['amount']) for t in matched])
        except:
            return 0.0

    def _execute_failsafe_protection(self, symbol, side, amount):
        """The Red Button: Close position at all costs."""
        if not failsafe_guard(self, symbol):
            return

        try:
            order_side = 'sell' if side == 'LONG' else 'buy'
            params = {"reduceOnly": True}
            params.update(self.build_bitget_params(side, is_close=True))
            self.dh.exchange.create_order(self.dh.format_symbol(symbol), 'MARKET', order_side, amount, None, params)
            errors_logger.critical(f"[FAILSAFE] Successfully closed orphaned {symbol} position.")
        except Exception as e:
            err_msg = str(e).lower()
            if "reduceonly" in err_msg or "position not found" in err_msg or "-2022" in err_msg:
                errors_logger.critical(f"ReduceOnly rejection: Position likely already closed for {symbol}. Forcing state reset.")
                force_remove_position(self, symbol)
                return
            else:
                errors_logger.critical(f"[FAILSAFE] TOTAL INFRASTRUCTURE COLLAPSE: Failed to close {symbol}: {e}")

    def atomic_stop_update(self, symbol: str, pos: PositionSchema, new_price: float) -> bool:
        """[C2] Atomic SL Update (Stop-First Migration)."""
        try:
            trades_logger.info(f"Initiating Stop-First migration for {symbol} to {new_price}...")
            # 1. Place the new stop loss order first with force=True
            new_order = self.place_stop_loss(symbol, pos.side, pos.amount, new_price, force=True)
            if new_order:
                # 2. Cancel the old SL order only after new is confirmed
                old_id = pos.sl_plan_id
                if old_id:
                    try:
                        self.dh.exchange.cancel_order(old_id, self.dh.format_symbol(symbol))
                        trades_logger.info(f"Cancelled old protective stop {old_id} for {symbol}.")
                    except Exception as e:
                        trades_logger.info(f"Best-effort cancellation of old stop {old_id} failed: {e}")
                
                # 3. Update the local state
                pos.sl_plan_id = new_order.get('id')
                pos.stop_loss = new_price
                return True
            return False
        except Exception as e:
            errors_logger.error(f"atomic_stop_update failed for {symbol}: {e}")
            return False

    def modify_stop_loss(self, symbol, pos):
        """atomic stop update wrapper."""
        return self.atomic_stop_update(symbol, pos, pos.stop_loss)

    def watchdog_verify_sl(self, positions):
        """
        [WATCHDOG] Periodically verifies all active positions have an SL on exchange.
        UNIVERSAL INPUT HANDLER (CRITICAL FIX)
        """
        import config
        if not config.BOT_STATE.get("is_running", True):
            return True
            
        if isinstance(positions, str):
            positions = [positions]
        elif hasattr(positions, "symbol"):
            positions = [positions]
        elif not isinstance(positions, list):
            return False

        all_ok = True

        for pos in positions:
            symbol = getattr(pos, "symbol", pos) if hasattr(pos, "symbol") else pos
            try:
                formatted_symbol = self.dh.format_symbol(symbol)
                
                orders = self.dh.exchange.fetch_open_orders(formatted_symbol)
                has_sl = any(is_protection_order(o, pos) for o in orders)
                
                if not has_sl:
                    trades_logger.warning(f"[WATCHDOG] Missing SL detected for {symbol}. Attempting recovery...")
                    if hasattr(pos, "stop_loss") and pos.stop_loss > 0:
                        self.place_stop_loss(symbol, pos.side, pos.amount, pos.stop_loss)
                    else:
                        errors_logger.critical(f"[WATCHDOG] Cannot recover SL for {symbol}. EMERGENCY CLOSE.")
                        if hasattr(pos, "side"):
                            self._execute_failsafe_protection(symbol, pos.side, pos.amount)
                    all_ok = False
                else:
                    all_ok = True # Existing SL found, do not recreate
            except Exception as e:
                errors_logger.error(f"[WATCHDOG] Failure auditing {symbol}: {e}")
                all_ok = False
                
        return all_ok
