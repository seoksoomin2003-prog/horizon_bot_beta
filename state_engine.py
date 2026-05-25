"""
position_manager.py — Institutional Reconciliation & State Machine (v2 Refactor)

INVARIANTS:
  1. Local state is ALWAYS a subset or reflection of Exchange Truth. [C4]
  2. PnL is updated in real-time (every loop).
  3. Orphan orders (no position) are purged within 15s.
"""

import time
from logger import trades_logger, errors_logger
from utils.state_manager import save_positions, load_positions
from utils.validator import Validator
from utils.schemas import PositionSchema
from utils.safe_core import SafeCore
from utils.exit_engine import ExitAction
from utils.lifecycle import validate_position_lifecycle, apply_tp_fill, derive_lifecycle_state

import time
import threading
from logger import trades_logger, errors_logger
from utils.state_manager import save_positions, load_positions
from utils.validator import Validator
from utils.schemas import PositionSchema
from utils.safe_core import SafeCore
from utils.exit_engine import ExitAction
from utils.lifecycle import validate_position_lifecycle, apply_tp_fill, derive_lifecycle_state

from utils.journal import ExecutionJournal

class DustRegistry:
    @classmethod
    def add(cls, symbol):
        ExecutionJournal.log_event(symbol, "DUST_QUARANTINE")

    @classmethod
    def is_quarantined(cls, symbol):
        snap = ExecutionJournal.get_snapshot(symbol)
        return snap.get("dust_quarantined", False)

class StateEngine:
    def __init__(self):
        self.lock = threading.Lock()
        with self.lock:
            self.active_positions = self._load_initial_state()
            self.last_trade_times = {} # {symbol: timestamp}
            self._is_first_reconciliation = True

    def _load_initial_state(self):
        raw_data = load_positions()
        return [Validator.position(p) for p in raw_data]

    def reconcile_all(self, dh, execution_engine):
        """
        [THE RECONCILIATION ENGINE]
        Runs every loop (15s) to ensure strict parity between Bot and Exchange.
        """
        with self.lock:
            import config
            if not config.BOT_STATE.get("is_running", True):
                return True
                
            try:
                # 1. Fetch Source of Truth
                from utils.degraded_manager import DegradedExchangeManager
                try:
                    exchange_positions = dh.fetch_active_positions(retries=3)
                    if exchange_positions is None:
                        DegradedExchangeManager().report_failure("Exchange positions returned None")
                        return # Skip if exchange unreachable
                    DegradedExchangeManager().report_success()
                except Exception as ex_err:
                    DegradedExchangeManager().report_failure(str(ex_err))
                    raise ex_err
                
                # 2. Update Real-time PnL & Market State
                self._update_realtime_metrics(dh, exchange_positions)
                
                # 3. Reconcile Symbols
                updated_locals = []
                
                # A. Purge Ghost Locals & Sync Existing
                for local_pos in self.active_positions:
                    ex_pos = next((p for p in exchange_positions if p.symbol == local_pos.symbol), None)
                    if ex_pos:
                        # C-01: Fix Side Mismatch / State Drift
                        if local_pos.side != ex_pos.side:
                            errors_logger.warning(f"[STATE_DRIFT] Side mismatch for {local_pos.symbol}. Local: {local_pos.side}, Exchange: {ex_pos.side}. Overwriting with exchange truth.")
                            local_pos.side = ex_pos.side

                        # HYBRID LIFECYCLE RECONCILER
                        if getattr(local_pos, "initial_amount", 0.0) <= 0:
                            local_pos.initial_amount = ex_pos.amount
                        
                        # 1. Reconcile and converge in-flight pending events via exchange trade logs
                        self.confirm_pending_lifecycle_events(dh, local_pos)
                        
                        # 2. Synchronize all local position flags strictly with Authoritative Journal Snapshot
                        snap = ExecutionJournal.get_snapshot(local_pos.symbol)
                        local_pos.tp1_done = snap.get("tp1_done", False)
                        local_pos.tp2_done = snap.get("tp2_done", False)
                        local_pos.tp1_pending = snap.get("tp1_pending", False)
                        local_pos.tp2_pending = snap.get("tp2_pending", False)
                        
                        # Sync amount (Truth)
                        local_pos.amount = ex_pos.amount
                        local_pos.entry_price = ex_pos.entry_price # Re-sync price if drift detected
                        
                        updated_locals.append(local_pos)
                    else:
                        # [C-01] Multi-Stage Confirmed position purging.
                        if getattr(local_pos, "status", "open") != "missing_pending_confirmation":
                            trades_logger.warning(f"[RECONCILE] Position for {local_pos.symbol} went missing from exchange. Transitioning to MISSING_PENDING_CONFIRMATION.")
                            local_pos.status = "missing_pending_confirmation"
                            local_pos.missing_since = time.time()
                            local_pos.missing_cycles = 1
                            updated_locals.append(local_pos)
                        else:
                            # 1. Quorum Poll
                            quorum_active = False
                            try:
                                for _ in range(2):
                                    test_pos = dh.fetch_active_positions(retries=1)
                                    if any(p.symbol == local_pos.symbol for p in test_pos):
                                        quorum_active = True
                                        break
                            except Exception:
                                pass

                            if quorum_active:
                                trades_logger.warning(f"[RECONCILE] Stale REST cache detected for {local_pos.symbol}. Position is actually open. Restoring status to 'open'.")
                                local_pos.status = "open"
                                local_pos.missing_since = 0.0
                                local_pos.missing_cycles = 0
                                updated_locals.append(local_pos)
                            else:
                                # 2. Journal Check
                                journal_closed = False
                                try:
                                    snap = ExecutionJournal.get_snapshot(local_pos.symbol)
                                    if snap.get("status") == "closed" or snap.get("position_closed", False):
                                        journal_closed = True
                                except Exception:
                                    pass

                                # 3. Open Orders Check
                                has_open_orders = False
                                try:
                                    open_orders = dh.exchange.fetch_open_orders(dh.format_symbol(local_pos.symbol))
                                    if len(open_orders) > 0:
                                        has_open_orders = True
                                except Exception:
                                    pass

                                # 4. Recent Fills Check
                                recent_close_fill = False
                                try:
                                    trades_list = dh.exchange.fetch_my_trades(dh.format_symbol(local_pos.symbol), limit=10)
                                    close_side = 'sell' if local_pos.side == 'LONG' else 'buy'
                                    if any(t.get('side') == close_side and (time.time() - t.get('timestamp', 0)/1000.0) < 600 for t in trades_list):
                                        recent_close_fill = True
                                except Exception:
                                    pass

                                # 5. Cooldown Check
                                cooldown_elapsed = (time.time() - local_pos.missing_since) >= 15.0 or local_pos.missing_cycles >= 2

                                confirmed_closed = journal_closed or recent_close_fill or (not has_open_orders and cooldown_elapsed)

                                if confirmed_closed:
                                    trades_logger.warning(f"[RECONCILE] Purging confirmed closed position: {local_pos.symbol} (via multi-source convergence).")
                                    ExecutionJournal.log_event(local_pos.symbol, "POSITION_CLOSED_CONFIRMED")
                                    # Purged (not appended to updated_locals)
                                else:
                                    trades_logger.warning(f"[RECONCILE] Position {local_pos.symbol} is missing but NOT yet confirmed closed. Retaining in quarantine (cycle {local_pos.missing_cycles}).")
                                    local_pos.missing_cycles += 1
                                    updated_locals.append(local_pos)
                
                # B. Reconstruct Missing Locals (Orphan Exchange Positions)
                for ex_pos in exchange_positions:
                    if not any(p.symbol == ex_pos.symbol for p in updated_locals):
                        # Dust Quarantine Check
                        is_dust = DustRegistry.is_quarantined(ex_pos.symbol)
                        
                        if is_dust:
                            trades_logger.warning(f"[DUST_QUARANTINE] Ignoring terminal dust for {ex_pos.symbol} ({ex_pos.amount}).")
                            continue
                            
                        trades_logger.warning(f"[RECONCILE] Found orphan exchange position for {ex_pos.symbol}. Reconstructing...")
                        # Reconstruct from exchange data
                        new_pos = PositionSchema(
                            symbol=ex_pos.symbol,
                            side=ex_pos.side,
                            amount=ex_pos.amount,
                            initial_amount=ex_pos.amount,
                            entry_price=ex_pos.entry_price,
                            stop_loss=0.0,
                            take_profit=0.0
                        )
                        
                        # C-02: Recovery Safety Rebuild
                        self.recover_position_protection(dh, execution_engine, new_pos)
                        updated_locals.append(new_pos)
                
                self.active_positions = updated_locals
                # Lifecycle invariant audit + normalization (preserve flags unless position closed)
                for p in self.active_positions:
                    validate_position_lifecycle(p, allow_repair=True, hard_halt_on_unrecoverable=True)
                save_positions([p.__dict__ for p in self.active_positions])
                
                # 4. Global Orphan Order Cleaner
                from config import SYMBOLS
                for symbol in SYMBOLS:
                    if not any(p.symbol == symbol for p in self.active_positions):
                        self.run_orphan_cleaner(execution_engine, symbol)
                        
                self._is_first_reconciliation = False
                return True
            except Exception as e:
                errors_logger.error(f"Reconciliation failure: {e}")
                return False

    def persist(self):
        with self.lock:
            save_positions([p.__dict__ for p in self.active_positions])

    def get_position(self, symbol: str):
        with self.lock:
            return next((p for p in self.active_positions if p.symbol == symbol), None)

    def confirm_pending_lifecycle_events(self, dh, local_pos):
        """
        Check exchange trades to confirm pending TP1/TP2 events.
        Transitions lifecycle based strictly on quantity-driven trade log analysis.
        """
        symbol = local_pos.symbol
        formatted_symbol = dh.format_symbol(symbol)
        
        # Only verify if there is an in-flight pending TP
        if not getattr(local_pos, "tp1_pending", False) and not getattr(local_pos, "tp2_pending", False):
            return

        try:
            # Fetch recent trades
            recent_trades = dh.exchange.fetch_my_trades(formatted_symbol, limit=20)
            close_side = 'sell' if local_pos.side == 'LONG' else 'buy'
            
            # Filter trades for this side within the last 10 minutes
            matching_trades = [
                t for t in recent_trades 
                if t.get('side') == close_side 
                and (time.time() - t.get('timestamp', 0)/1000.0) < 600
            ]
            
            total_filled_exits = sum(float(t.get('amount', 0.0)) for t in matching_trades)
            
            # If we don't have matching trades yet, return
            if total_filled_exits <= 0.0:
                return

            # Reconstruct intended quantity if it's not set
            if getattr(local_pos, "intended_qty", 0.0) <= 0.0:
                regime = getattr(local_pos, "regime", "RANGE")
                tp_ratio = 0.40
                if regime == "TREND":
                    tp_ratio = 0.20
                elif regime == "LOW_VOL":
                    tp_ratio = 0.30
                local_pos.intended_qty = local_pos.initial_amount * tp_ratio

            # Update quantity metrics
            local_pos.filled_qty = total_filled_exits
            local_pos.remaining_qty = max(0.0, local_pos.intended_qty - total_filled_exits)
            local_pos.cumulative_fill_ratio = total_filled_exits / local_pos.intended_qty if local_pos.intended_qty > 0 else 0.0
            
            is_threshold_reached = local_pos.cumulative_fill_ratio >= 0.95

            if getattr(local_pos, "tp1_pending", False):
                if is_threshold_reached:
                    trades_logger.info(f"[LIFECYCLE] Replay confirmed TP1 DONE for {symbol}. Filled: {total_filled_exits}/{local_pos.intended_qty}")
                    ExecutionJournal.log_event(symbol, "TP1_EXECUTED")
                    local_pos.tp1_done = True
                    local_pos.tp1_pending = False
                else:
                    trades_logger.info(f"[LIFECYCLE] Replay identified TP1 PARTIAL FILL for {symbol}. Filled: {total_filled_exits}/{local_pos.intended_qty}")
                    ExecutionJournal.log_event(symbol, "TP1_PARTIALLY_FILLED")
                    local_pos.tp1_done = False
                    # Keep tp1_pending = True to maintain the partial pending state

            elif getattr(local_pos, "tp2_pending", False):
                if is_threshold_reached:
                    trades_logger.info(f"[LIFECYCLE] Replay confirmed TP2 DONE for {symbol}. Filled: {total_filled_exits}/{local_pos.intended_qty}")
                    ExecutionJournal.log_event(symbol, "TP2_EXECUTED")
                    local_pos.tp2_done = True
                    local_pos.tp2_pending = False
                else:
                    trades_logger.info(f"[LIFECYCLE] Replay identified TP2 PARTIAL FILL for {symbol}. Filled: {total_filled_exits}/{local_pos.intended_qty}")
                    ExecutionJournal.log_event(symbol, "TP2_PARTIALLY_FILLED")
                    local_pos.tp2_done = False
                    # Keep tp2_pending = True to maintain the partial pending state
        except Exception as e:
            errors_logger.error(f"Error checking pending trades for {symbol}: {e}")

    def mark_tp_done(self, symbol: str, tp_level: int) -> bool:
        """
        STRICT transition: only called after exchange-confirmed fill.
        """
        with self.lock:
            pos = next((p for p in self.active_positions if p.symbol == symbol), None)
            if not pos:
                return False

            from utils.journal import ExecutionJournal
            ExecutionJournal.log_event(symbol, f"TP{tp_level}_EXECUTED")

            changed, transition = apply_tp_fill(pos, tp_level)
            if changed:
                if tp_level == 1:
                    pos.tp1_pending = False
                elif tp_level == 2:
                    pos.tp2_pending = False
                trades_logger.info(f"[LIFECYCLE] {symbol} TP{tp_level}_DONE transition {transition}")
                save_positions([p.__dict__ for p in self.active_positions])
            return changed

    def _update_realtime_metrics(self, dh, exchange_positions):
        """Updates internal pnl and trackers for active positions."""
        for local_pos in self.active_positions:
            ex_pos = next((p for p in exchange_positions if p.symbol == local_pos.symbol), None)
            if ex_pos:
                # Use exchange reported PnL for accuracy
                local_pos.pnl = SafeCore.float(ex_pos.pnl)
            else:
                local_pos.pnl = 0.0

    def recover_position_protection(self, dh, execution_engine, pos: PositionSchema):
        """
        [C-02] Restores full protection (SL/ATR) to an orphaned/reconstructed position.
        """
        try:
            from execution_engine import is_protection_order
            symbol = pos.symbol
            formatted_symbol = dh.format_symbol(symbol)
            
            # 1. Fetch exchange open orders
            orders = dh.exchange.fetch_open_orders(formatted_symbol)
            
            # 2. Locate existing SL
            existing_sl = next((o for o in orders if is_protection_order(o, pos)), None)
            
            if existing_sl:
                # 3. Recover stop_price from exchange
                recovered_price = SafeCore.float(existing_sl.get('stopPrice') or existing_sl.get('price'))
                if recovered_price > 0:
                    pos.stop_loss = recovered_price
                    pos.sl_plan_id = existing_sl.get('id')
                    trades_logger.info(f"[RECOVERY_PROTECTION] Restored existing SL for {symbol} at {recovered_price}")
                    # Estimate ATR from stop distance (very rough, but better than 0)
                    dist = abs(pos.entry_price - recovered_price)
                    pos.atr = dist / 2.0  # Assuming initial SL was 2 ATR
                    return
            
            # 4. If unavailable: reconstruct ATR stop and place emergency SL
            import config
            df = dh.fetch_ohlcv(symbol, config.TIMEFRAME)
            last = SafeCore.last_row(df)
            current_atr = SafeCore.float(last.get('atr')) if last is not None else 0.0
            ticker = dh.fetch_ticker(symbol)
            
            if current_atr <= 0 or not ticker:
                # Fallback to mathematically bounded emergency stop
                fallback_dist_pct = getattr(config, "MIN_STOP_DISTANCE_PCT", 0.01) * 1.5
                trades_logger.warning(f"[RECOVERY] ATR unavailable for {symbol}. Using bounded {fallback_dist_pct*100:.1f}% emergency stop.")
                stop_dist = pos.entry_price * fallback_dist_pct
                current_atr = stop_dist / 2.0
            else:
                stop_dist = current_atr * 2.0
                
            pos.atr = current_atr
            
            if pos.side == 'LONG':
                raw_sl_price = ticker.last - stop_dist if ticker else (pos.entry_price - stop_dist)
                # Mathematically bound to a positive floor (e.g. at least 50% of entry price drop max)
                emergency_sl_price = max(raw_sl_price, pos.entry_price * 0.5)
            else:
                raw_sl_price = ticker.last + stop_dist if ticker else (pos.entry_price + stop_dist)
                # Mathematically bound to a ceiling (e.g. max 200% of entry price)
                emergency_sl_price = min(raw_sl_price, pos.entry_price * 2.0)
                
            # Formatting precision (using dh precision helper if available)
            try:
                emergency_sl_price = float(dh.exchange.price_to_precision(dh.format_symbol(symbol), emergency_sl_price))
            except Exception:
                # Decimal-safe tick-size reconstruction from ticker.last
                try:
                    import decimal
                    tick_size = abs(decimal.Decimal(str(ticker.last)).normalize().as_tuple().exponent)
                    emergency_sl_price = round(emergency_sl_price, tick_size)
                    if emergency_sl_price <= 0.0:
                        raise ValueError("Rounded SL <= 0.0")
                except Exception as e:
                    errors_logger.error(f"[PRECISION_FAIL] Dynamic tick-size failed for {symbol}: {e}")
                    emergency_sl_price = round(emergency_sl_price, 4) # Absolute fallback
                    if emergency_sl_price <= 0.0:
                        errors_logger.critical(f"[ABORT] Emergency SL for {symbol} evaluates to <= 0.0. Aborting payload.")
                        return
                
            pos.stop_loss = emergency_sl_price
            
            # 5. Place emergency SL immediately
            sl_order = execution_engine.place_stop_loss(symbol, pos.side, pos.amount, emergency_sl_price)
            if sl_order:
                pos.sl_plan_id = sl_order.get('id')
                trades_logger.warning(f"[EMERGENCY_STOP_RESTORED] Placed new SL for orphaned {symbol} at {emergency_sl_price}")
            
        except Exception as e:
            errors_logger.error(f"Failed to recover position protection for {pos.symbol}: {e}")

    def is_position_active(self, symbol):
        with self.lock:
            return any(p.symbol == symbol for p in self.active_positions)

    def remove_position(self, symbol):
        with self.lock:
            target = next((p for p in self.active_positions if p.symbol == symbol), None)
            if target:
                self.active_positions.remove(target)
                save_positions([p.__dict__ for p in self.active_positions])
                return True
            return False
    
    def can_trade(self, symbol, cooldown_seconds):
        with self.lock:
            last_time = self.last_trade_times.get(symbol, 0)
            return (int(time.time()) - last_time) >= cooldown_seconds

    def is_fallback_mode_active(self) -> bool:
        with self.lock:
            now = int(time.time())
            if not self.last_trade_times:
                return False
            
            last_global_trade = max(self.last_trade_times.values())
            return (now - last_global_trade) > 86400

    def close_all_positions(self, execution_engine, dh):
        """
        [EMERGENCY STOP] Closes all tracked and untracked positions on the exchange.
        """
        with self.lock:
            try:
                trades_logger.warning("!!! EMERGENCY STOP: Closing all positions !!!")
                
                # 1. Close tracked locals
                for pos in list(self.active_positions):
                    execution_engine.close_full(pos.symbol, pos)
                    target = next((p for p in self.active_positions if p.symbol == pos.symbol), None)
                    if target:
                        self.active_positions.remove(target)
                save_positions([p.__dict__ for p in self.active_positions])
                
                # 2. Re-verify with exchange (Scrape for orphans)
                exchange_positions = dh.fetch_active_positions(retries=2)
                if exchange_positions:
                    for ex_pos in exchange_positions:
                        trades_logger.warning(f"[EMERGENCY] Closing orphaned exchange position: {ex_pos.symbol}")
                        from utils.schemas import PositionSchema
                        placeholder = PositionSchema(symbol=ex_pos.symbol, side=ex_pos.side, amount=ex_pos.amount)
                        execution_engine.close_full(ex_pos.symbol, placeholder)
                
                return True
            except Exception as e:
                errors_logger.error(f"Emergency close_all failed: {e}")
                return False

    def run_orphan_cleaner(self, execution_engine, symbol):
        """
        [C3] Institutional Orphan Cleaner
        Purge all limit/plan orders for a symbol if no position is active.
        """
        try:
            return execution_engine.cancel_all_orders(symbol)
        except Exception as e:
            errors_logger.error(f"Orphan cleaner failed for {symbol}: {e}")
            return False

PositionManager = StateEngine
