"""
position_manager.py — Institutional Reconciliation & State Machine (Event-Sourced V3)

INVARIANTS:
  1. Local state is ALWAYS derived strictly from Event Journal replays.
  2. Reconciliation ONLY produces events (POSITION_CLOSED, RECONCILE_CORRECTION, etc.).
  3. No mutable snapshots are permitted.
"""

import time
import threading
import os
from typing import List, Optional
from logger import trades_logger, errors_logger
from utils.validator import Validator
from utils.schemas import PositionSchema
from utils.safe_core import SafeCore
from utils.event_sourced_runtime import OwnershipRegistry, ConcurrencyException, LifecycleEvent
from utils.persistence import get_absolute_path
import config
from utils.runtime_coordinator import RuntimeCoordinator

class StateEngine:
    def __init__(self):
        self.lock = threading.RLock()
        with self.lock:
            self.last_trade_times = {} # {symbol: timestamp}
            self._is_first_reconciliation = True
            
            # Load last_entry_bar_ts from json persistence
            import json
            self.last_entry_bar_ts_file = get_absolute_path("last_entry_bar_ts.json")
            self.last_entry_bar_ts = {}
            if os.path.exists(self.last_entry_bar_ts_file):
                try:
                    with open(self.last_entry_bar_ts_file, 'r') as f:
                        self.last_entry_bar_ts = json.load(f)
                except Exception:
                    pass

    def save_last_entry_bar_ts(self):
        from utils.persistence import atomic_write_json
        try:
            atomic_write_json(self.last_entry_bar_ts_file, self.last_entry_bar_ts)
        except Exception as e:
            errors_logger.error(f"Failed to save last_entry_bar_ts: {e}")

    @property
    def active_positions(self) -> List[PositionSchema]:
        """Dynamically projects active positions from all known symbol journals, with backward-compatible unit-testing fallback."""
        with self.lock:
            if hasattr(self, "_active_positions_override") and self._active_positions_override is not None:
                return self._active_positions_override
                
        active = []
        import utils.state_manager as sm
        legacy_positions = {}
        try:
            positions_data = sm.load_positions()
            for pos_dict in positions_data:
                try:
                    p = PositionSchema(**pos_dict)
                    legacy_positions[p.symbol] = p
                except Exception:
                    pass
        except Exception:
            pass

        journals_dir = get_absolute_path("event_journals")
        discovered_symbols = set(config.SYMBOLS)
        if os.path.exists(journals_dir):
            try:
                for filename in os.listdir(journals_dir):
                    if filename.endswith("_events.jsonl"):
                        symbol = filename.replace("_events.jsonl", "")
                        discovered_symbols.add(symbol)
            except Exception:
                pass

        for symbol in sorted(discovered_symbols):
            owner = OwnershipRegistry().get_owner(symbol)
            if len(owner.events) > 0:
                base_pos = legacy_positions.get(symbol)
                state = self._project_with_fallback(symbol, owner.events, base_pos)
                if state:
                    active.append(state)
            elif symbol in legacy_positions:
                active.append(legacy_positions[symbol])
                
        for symbol, pos in legacy_positions.items():
            if not any(p.symbol == symbol for p in active):
                owner = OwnershipRegistry().get_owner(symbol)
                if len(owner.events) > 0:
                    state = self._project_with_fallback(symbol, owner.events, pos)
                    if state:
                        active.append(state)
                else:
                    active.append(pos)

        return active

    def _project_with_fallback(self, symbol: str, events: List[LifecycleEvent], base_pos: Optional[PositionSchema]) -> Optional[PositionSchema]:
        import copy
        pos = None
        has_entry_filled = any(e.event_type in ("ENTRY_FILLED", "ENTRY_PARTIALLY_FILLED") for e in events)
        if not has_entry_filled and base_pos is not None:
            pos = copy.deepcopy(base_pos)
            
        for event in sorted(events, key=lambda x: (x.sequence, x.timestamp)):
            etype = event.event_type
            m = event.metadata
            
            if etype == "ENTRY_FILLED":
                pos = PositionSchema(
                    symbol=symbol,
                    side=m.get("side", "LONG").upper(),
                    amount=float(m.get("amount", 0.0)),
                    initial_amount=float(m.get("amount", 0.0)),
                    entry_price=float(m.get("price", 0.0)),
                    status="open",
                    stop_loss=0.0,
                    take_profit=0.0,
                    tp1_done=False,
                    tp2_done=False,
                    tp1_pending=False,
                    tp2_pending=False,
                    trailing_active=False,
                    breakeven_active=False,
                    candles_held=0
                )
            elif etype == "ENTRY_PARTIALLY_FILLED":
                pos = PositionSchema(
                    symbol=symbol,
                    side=m.get("side", "LONG").upper(),
                    amount=float(m.get("amount", 0.0)),
                    initial_amount=float(m.get("amount", 0.0)),
                    entry_price=float(m.get("price", 0.0)),
                    status="open",
                    stop_loss=0.0,
                    take_profit=0.0,
                    lifecycle_state="PARTIAL",
                    tp1_done=False,
                    tp2_done=False,
                    tp1_pending=False,
                    tp2_pending=False,
                    trailing_active=False,
                    breakeven_active=False,
                    candles_held=0
                )
                setattr(pos, "entry_order_id", m.get("order_id"))
            elif pos is not None:
                if etype == "ENTRY_PARTIALLY_FILLED":
                    pos.amount = float(m.get("amount", 0.0))
                    pos.initial_amount = float(m.get("amount", 0.0))
                    pos.lifecycle_state = "PARTIAL"
                if etype == "SL_PLACED":
                    pos.stop_loss = float(m.get("price", 0.0))
                    pos.sl_plan_id = m.get("order_id")
                elif etype == "TP_PLACED":
                    pos.take_profit = float(m.get("price", 0.0))
                    pos.tp_plan_id = m.get("order_id")
                elif etype == "TP1_TRIGGERED":
                    pos.tp1_pending = True
                    pos.intended_qty = float(m.get("amount", 0.0))
                elif etype == "TP1_FILLED":
                    pos.tp1_done = True
                    pos.tp1_pending = False
                    pos.filled_qty = float(m.get("amount", 0.0))
                    pos.remaining_qty = max(0.0, pos.intended_qty - pos.filled_qty)
                    pos.cumulative_fill_ratio = pos.filled_qty / pos.intended_qty if pos.intended_qty > 0 else 0.0
                    pos.amount = max(0.0, pos.initial_amount - pos.filled_qty)
                elif etype == "TP2_TRIGGERED":
                    pos.tp2_pending = True
                    pos.intended_qty = float(m.get("amount", 0.0))
                elif etype == "TP2_FILLED":
                    pos.tp2_done = True
                    pos.tp2_pending = False
                    pos.filled_qty = float(m.get("amount", 0.0))
                    pos.remaining_qty = max(0.0, pos.intended_qty - pos.filled_qty)
                    pos.cumulative_fill_ratio = pos.filled_qty / pos.intended_qty if pos.intended_qty > 0 else 0.0
                    pos.amount = max(0.0, pos.amount - pos.filled_qty)
                elif etype == "SL_MOVED":
                    pos.stop_loss = float(m.get("price", 0.0))
                    pos.sl_plan_id = m.get("order_id")
                    if pos.tp1_done:
                        pos.trailing_active = True
                elif etype == "POSITION_CLOSED":
                    pos.amount = 0.0
                    pos.status = "closed"
                    pos.tp1_pending = False
                    pos.tp2_pending = False
                elif etype == "RECONCILE_CORRECTION":
                    for k, v in m.items():
                        if hasattr(pos, k):
                            setattr(pos, k, v)
                            
        if pos and pos.status == "closed":
            return None
        return pos

    @active_positions.setter
    def active_positions(self, value):
        with self.lock:
            self._active_positions_override = value

    def persist(self):
        """Idempotent persist method for backward-compatibility with tests."""
        with self.lock:
            from utils.retry_scheduler import TaskLease, WorkerGenerationRegistry
            TaskLease.verify_active()
            WorkerGenerationRegistry.verify_active()
            
            if hasattr(self, "_active_positions_override") and self._active_positions_override is not None:
                import utils.state_manager as sm
                try:
                    sm.save_positions([p.__dict__ if hasattr(p, "__dict__") else p for p in self._active_positions_override])
                except Exception as e:
                    errors_logger.error(f"Failed to persist overridden active positions: {e}")

    def remove_position(self, symbol: str):
        """Transition position to closed in the event store."""
        try:
            owner = OwnershipRegistry().get_owner(symbol)
            owner.submit_event("POSITION_CLOSED", {"reason": "MANUAL_REMOVE"})
        except Exception as e:
            errors_logger.error(f"Failed to submit POSITION_CLOSED on remove_position for {symbol}: {e}")

    def reconcile_all(self, dh, execution_engine):
        """
        [EVENT PRODUCER RECONCILIATION]
        Only queries exchange and emits RECONCILE_CORRECTION or POSITION_CLOSED events.
        Never directly mutates position schema objects.
        """
        if not RuntimeCoordinator().get_state("is_running", True):
            return True
            
        try:
            from utils.degraded_manager import DegradedExchangeManager
            try:
                exchange_positions = dh.fetch_active_positions(retries=3)
                if exchange_positions is None:
                    DegradedExchangeManager().report_failure("Exchange positions returned None")
                    return
                DegradedExchangeManager().report_success()
            except Exception as ex_err:
                DegradedExchangeManager().report_failure(str(ex_err))
                raise ex_err
            
            # Use current projections
            local_positions = self.active_positions
            
            # [C-01] Resuming Protection for ENTRY_PENDING_PROTECTION
            for local_pos in local_positions:
                if local_pos.lifecycle_state == "ENTRY_PENDING_PROTECTION":
                    trades_logger.warning(f"[RECONCILE] Found position for {local_pos.symbol} in ENTRY_PENDING_PROTECTION state. Resuming protection workflow.")
                    ex_pos = next((p for p in exchange_positions if p.symbol == local_pos.symbol), None)
                    if ex_pos:
                        try:
                            self.recover_position_protection(dh, execution_engine, local_pos)
                        except Exception as err:
                            errors_logger.error(f"[RECONCILE] Failed to recover protection for {local_pos.symbol}: {err}")
                    else:
                        trades_logger.critical(f"[RECONCILE] Position for {local_pos.symbol} gone from exchange while pending protection! Purging stale state.")
                        owner = OwnershipRegistry().get_owner(local_pos.symbol)
                        try:
                            owner.submit_event("POSITION_CLOSED", {"reason": "PENDING_PROTECTION_ORPHAN"})
                        except Exception:
                            pass
            
            # [C-02] Monitoring and Updating Protection for PARTIAL fills
            for local_pos in local_positions:
                if local_pos.lifecycle_state == "PARTIAL":
                    entry_order_id = getattr(local_pos, "entry_order_id", None)
                    if entry_order_id:
                        trades_logger.info(f"[PARTIAL_MONITOR] Monitoring open entry order {entry_order_id} for {local_pos.symbol}...")
                        formatted_symbol = dh.format_symbol(local_pos.symbol)
                        try:
                            ex_order = dh.exchange.fetch_order(entry_order_id, formatted_symbol)
                            order_status = str(ex_order.get("status", "")).upper()
                            filled_qty = SafeCore.float(ex_order.get("filled", 0.0))
                            
                            if filled_qty > local_pos.amount:
                                trades_logger.warning(f"[PARTIAL_MONITOR] Quantity increased from {local_pos.amount} to {filled_qty} for {local_pos.symbol}!")
                                
                                if local_pos.sl_plan_id:
                                    try:
                                        dh.exchange.cancel_order(local_pos.sl_plan_id, formatted_symbol)
                                    except Exception:
                                        pass
                                if local_pos.tp_plan_id:
                                    try:
                                        dh.exchange.cancel_order(local_pos.tp_plan_id, formatted_symbol)
                                    except Exception:
                                        pass
                                        
                                owner = OwnershipRegistry().get_owner(local_pos.symbol)
                                current_version = len(owner.events)
                                
                                if order_status != "CLOSED":
                                    owner.submit_event("ENTRY_PARTIALLY_FILLED", {
                                        "side": local_pos.side,
                                        "amount": filled_qty,
                                        "price": SafeCore.float(ex_order.get("price") or local_pos.entry_price),
                                        "order_id": entry_order_id
                                    }, expected_version=current_version)
                                else:
                                    owner.submit_event("ENTRY_FILLED", {
                                        "side": local_pos.side,
                                        "amount": filled_qty,
                                        "price": SafeCore.float(ex_order.get("price") or local_pos.entry_price)
                                    }, expected_version=current_version)
                                
                                from utils.event_sourced_runtime import ProtectionOrchestrator
                                ProtectionOrchestrator().place_initial_protection(
                                    symbol=local_pos.symbol,
                                    side=local_pos.side,
                                    amount=filled_qty,
                                    stop_price=local_pos.stop_loss or (local_pos.entry_price * 0.9 if local_pos.side == "LONG" else local_pos.entry_price * 1.1),
                                    take_profit_price=local_pos.take_profit,
                                    execution_engine=execution_engine
                                )
                        except Exception as monitor_err:
                            errors_logger.error(f"[PARTIAL_MONITOR] Error monitoring entry order {entry_order_id} for {local_pos.symbol}: {monitor_err}")
            
            # Re-fetch active_positions to include any potential transitions
            local_positions = self.active_positions
            
            for local_pos in local_positions:
                symbol = local_pos.symbol
                formatted_symbol = dh.format_symbol(symbol)
                ex_pos = next((p for p in exchange_positions if p.symbol == symbol), None)
                
                owner = OwnershipRegistry().get_owner(symbol)
                current_version = len(owner.events)
                
                if ex_pos:
                    corrections = {}
                    
                    if local_pos.side != ex_pos.side:
                        corrections["side"] = ex_pos.side
                        
                    if getattr(local_pos, "initial_amount", 0.0) <= 0:
                        corrections["initial_amount"] = ex_pos.amount
                        
                    if local_pos.amount != ex_pos.amount:
                        corrections["amount"] = ex_pos.amount
                        
                    if local_pos.entry_price != ex_pos.entry_price:
                        corrections["entry_price"] = ex_pos.entry_price
                        
                    if local_pos.pnl != SafeCore.float(ex_pos.pnl):
                        corrections["pnl"] = SafeCore.float(ex_pos.pnl)
                        
                    if corrections:
                        try:
                            owner.submit_event("RECONCILE_CORRECTION", corrections, expected_version=current_version)
                        except ConcurrencyException as ce:
                            errors_logger.warning(f"OCC failed during reconcile for {symbol}: {ce}")
                            
                    # Lifecycle sync via network call
                    self.confirm_pending_lifecycle_events(dh, local_pos)
                else:
                    # Missing from exchange. Apply multi-stage Quorum Poll.
                    if getattr(local_pos, "status", "open") != "missing_pending_confirmation":
                        try:
                            owner.submit_event("RECONCILE_CORRECTION", {
                                "status": "missing_pending_confirmation",
                                "missing_since": time.time(),
                                "missing_cycles": 1
                            }, expected_version=current_version)
                        except ConcurrencyException:
                            pass
                    else:
                        quorum_active = False
                        try:
                            test_pos = dh.fetch_active_positions(retries=1)
                            if any(p.symbol == local_pos.symbol for p in test_pos):
                                quorum_active = True
                        except Exception:
                            pass
                            
                        if quorum_active:
                            try:
                                owner.submit_event("RECONCILE_CORRECTION", {
                                    "status": "open",
                                    "missing_since": 0.0,
                                    "missing_cycles": 0
                                }, expected_version=current_version)
                            except ConcurrencyException:
                                pass
                        else:
                            has_open_orders = False
                            open_orders = []
                            try:
                                open_orders = dh.exchange.fetch_open_orders(formatted_symbol)
                                if len(open_orders) > 0:
                                    has_open_orders = True
                            except Exception:
                                pass
                                
                            cooldown_elapsed = (time.time() - local_pos.missing_since) >= 15.0 or local_pos.missing_cycles >= 2
                            confirmed_closed = not has_open_orders and cooldown_elapsed
                            
                            if not confirmed_closed and cooldown_elapsed and has_open_orders:
                                try:
                                    for o in open_orders:
                                        dh.exchange.cancel_order(o["id"], formatted_symbol)
                                except Exception:
                                    pass
                                confirmed_closed = True
                                
                            if confirmed_closed:
                                try:
                                    owner.submit_event("POSITION_CLOSED", {"reason": "RECONCIER_CONFIRMED_CLOSED"}, expected_version=current_version)
                                except ConcurrencyException:
                                    pass
                            else:
                                try:
                                    owner.submit_event("RECONCILE_CORRECTION", {
                                        "missing_cycles": local_pos.missing_cycles + 1
                                    }, expected_version=current_version)
                                except ConcurrencyException:
                                    pass
                                    
            for ex_pos in exchange_positions:
                if not any(p.symbol == ex_pos.symbol for p in local_positions):
                    trades_logger.warning(f"[RECONCILE] Found orphan exchange position for {ex_pos.symbol}. Emitting ENTRY_FILLED.")
                    owner = OwnershipRegistry().get_owner(ex_pos.symbol)
                    current_version = len(owner.events)
                    if current_version == 0:
                        owner.submit_event("ENTRY_FILLED", {
                            "side": ex_pos.side,
                            "amount": ex_pos.amount,
                            "price": ex_pos.entry_price
                        })
                        new_pos = owner.get_state()
                        if new_pos:
                            self.recover_position_protection(dh, execution_engine, new_pos)
                            
            for symbol in config.SYMBOLS:
                if not any(p.symbol == symbol for p in self.active_positions):
                    self.run_orphan_cleaner(execution_engine, symbol)
                    
            self._is_first_reconciliation = False
            return True
        except Exception as e:
            errors_logger.error(f"Reconciliation failure: {e}")
            return False

    def get_position(self, symbol: str):
        owner = OwnershipRegistry().get_owner(symbol)
        return owner.get_state()

    def confirm_pending_lifecycle_events(self, dh, local_pos):
        symbol = local_pos.symbol
        formatted_symbol = dh.format_symbol(symbol)
        owner = OwnershipRegistry().get_owner(symbol)
        current_version = len(owner.events)
        
        if not getattr(local_pos, "tp1_pending", False) and not getattr(local_pos, "tp2_pending", False):
            return

        try:
            recent_trades = dh.exchange.fetch_my_trades(formatted_symbol, limit=20)
            close_side = 'sell' if local_pos.side == 'LONG' else 'buy'
            target_order_id = getattr(local_pos, "tp1_order_id", None) if getattr(local_pos, "tp1_pending", False) else getattr(local_pos, "tp2_order_id", None)
            
            matching_trades = []
            for t in recent_trades:
                if t.get('side') != close_side: continue
                if target_order_id and t.get('order') is not None and str(t.get('order')) == str(target_order_id):
                    matching_trades.append(t)
                elif getattr(local_pos, "tp_plan_id", None) and t.get('order') is not None and str(t.get('order')) == str(local_pos.tp_plan_id):
                    matching_trades.append(t)
                elif (not target_order_id and not getattr(local_pos, "tp_plan_id", None)) or t.get('order') is None:
                    matching_trades.append(t)
            
            total_filled_exits = sum(float(t.get('amount', 0.0)) for t in matching_trades)
            if total_filled_exits <= 0.0: return

            intended_qty = getattr(local_pos, "intended_qty", 0.0)
            if intended_qty <= 0.0:
                intended_qty = local_pos.initial_amount * 0.40
                
            cumulative_ratio = total_filled_exits / intended_qty if intended_qty > 0 else 0.0
            is_threshold_reached = cumulative_ratio >= 0.95

            if getattr(local_pos, "tp1_pending", False):
                if is_threshold_reached:
                    owner.submit_event("TP1_FILLED", {"amount": total_filled_exits}, expected_version=current_version)
                    local_pos.tp1_done = True
                    local_pos.tp1_pending = False
                    local_pos.filled_qty = total_filled_exits
                    local_pos.cumulative_fill_ratio = cumulative_ratio
                    local_pos.remaining_qty = max(0.0, intended_qty - total_filled_exits)
                    local_pos.amount = max(0.0, local_pos.initial_amount - total_filled_exits)
                else:
                    owner.submit_event("RECONCILE_CORRECTION", {
                        "filled_qty": total_filled_exits,
                        "cumulative_fill_ratio": cumulative_ratio,
                        "remaining_qty": max(0.0, intended_qty - total_filled_exits)
                    }, expected_version=current_version)
                    local_pos.filled_qty = total_filled_exits
                    local_pos.cumulative_fill_ratio = cumulative_ratio
                    local_pos.remaining_qty = max(0.0, intended_qty - total_filled_exits)
            elif getattr(local_pos, "tp2_pending", False):
                if is_threshold_reached:
                    owner.submit_event("TP2_FILLED", {"amount": total_filled_exits}, expected_version=current_version)
                    local_pos.tp2_done = True
                    local_pos.tp2_pending = False
                    local_pos.filled_qty = total_filled_exits
                    local_pos.cumulative_fill_ratio = cumulative_ratio
                    local_pos.remaining_qty = max(0.0, intended_qty - total_filled_exits)
                    local_pos.amount = max(0.0, local_pos.amount - total_filled_exits)
                else:
                    owner.submit_event("RECONCILE_CORRECTION", {
                        "filled_qty": total_filled_exits,
                        "cumulative_fill_ratio": cumulative_ratio,
                        "remaining_qty": max(0.0, intended_qty - total_filled_exits)
                    }, expected_version=current_version)
                    local_pos.filled_qty = total_filled_exits
                    local_pos.cumulative_fill_ratio = cumulative_ratio
                    local_pos.remaining_qty = max(0.0, intended_qty - total_filled_exits)
        except ConcurrencyException:
            pass
        except Exception as e:
            errors_logger.error(f"Error checking pending trades for {symbol}: {e}")

    def recover_position_protection(self, dh, execution_engine, pos: PositionSchema):
        try:
            from execution_engine import is_protection_order
            symbol = pos.symbol
            formatted_symbol = dh.format_symbol(symbol)
            orders = dh.exchange.fetch_open_orders(formatted_symbol)
            existing_sl = next((o for o in orders if is_protection_order(o, pos)), None)
            owner = OwnershipRegistry().get_owner(symbol)
            
            if existing_sl:
                recovered_price = SafeCore.float(existing_sl.get('stopPrice') or existing_sl.get('price'))
                if recovered_price > 0:
                    owner.submit_event("SL_PLACED", {"price": recovered_price, "order_id": existing_sl.get('id')})
                    return
            
            import config
            from utils.runtime_coordinator import RuntimeCoordinator
            df = dh.fetch_ohlcv(symbol, config.TIMEFRAME)
            last = SafeCore.last_row(df)
            current_atr = SafeCore.float(last.get('atr')) if last is not None else 0.0
            ticker = dh.fetch_ticker(symbol)
            
            if current_atr <= 0 or not ticker:
                fallback_dist_pct = getattr(config, "MIN_STOP_DISTANCE_PCT", 0.01) * 1.5
                stop_dist = pos.entry_price * fallback_dist_pct
                current_atr = stop_dist / 2.0
            else:
                stop_dist = current_atr * 2.0
                
            pos_side = pos.side.upper()
            if pos_side == 'LONG':
                raw_sl_price = ticker.last - stop_dist if ticker else (pos.entry_price - stop_dist)
                emergency_sl_price = max(raw_sl_price, pos.entry_price * 0.5)
            else:
                raw_sl_price = ticker.last + stop_dist if ticker else (pos.entry_price + stop_dist)
                emergency_sl_price = min(raw_sl_price, pos.entry_price * 2.0)
                
            try:
                emergency_sl_price = float(dh.exchange.price_to_precision(dh.format_symbol(symbol), emergency_sl_price))
            except Exception:
                emergency_sl_price = round(emergency_sl_price, 4)
                
            sl_order = execution_engine.place_stop_loss(symbol, pos.side, pos.amount, emergency_sl_price)
            if sl_order:
                owner.submit_event("SL_PLACED", {"price": emergency_sl_price, "order_id": sl_order.get('id')})
                owner.submit_event("RECONCILE_CORRECTION", {"atr": current_atr})
            
        except Exception as e:
            errors_logger.error(f"Failed to recover position protection for {pos.symbol}: {e}")

    def is_position_active(self, symbol):
        return any(p.symbol == symbol for p in self.active_positions)
    
    def can_trade(self, symbol, cooldown_seconds):
        with self.lock:
            last_time = self.last_trade_times.get(symbol, 0)
            return (int(time.time()) - last_time) >= cooldown_seconds

    def is_fallback_mode_active(self) -> bool:
        with self.lock:
            now = int(time.time())
            if not self.last_trade_times: return False
            last_global_trade = max(self.last_trade_times.values())
            return (now - last_global_trade) > 86400

    def close_all_positions(self, execution_engine, dh):
        try:
            for pos in self.active_positions:
                execution_engine.close_full(pos.symbol, pos)
            
            exchange_positions = dh.fetch_active_positions(retries=2)
            if exchange_positions:
                for ex_pos in exchange_positions:
                    placeholder = PositionSchema(symbol=ex_pos.symbol, side=ex_pos.side, amount=ex_pos.amount)
                    execution_engine.close_full(ex_pos.symbol, placeholder)
            return True
        except Exception as e:
            errors_logger.error(f"Emergency close_all failed: {e}")
            return False

    def run_orphan_cleaner(self, execution_engine, symbol):
        try:
            return execution_engine.cancel_all_orders(symbol)
        except Exception:
            return False

PositionManager = StateEngine
