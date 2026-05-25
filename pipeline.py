"""
pipeline.py — Horizon Bot v3 Crystalline Orchestrator

SINGLE CONTINUOUS EXECUTION PIPELINE:
SIGNAL -> CONVICTION -> RISK -> EXECUTION -> EXIT -> STATE_SYNC
"""

import time
from typing import List, Dict, Any, Optional
from logger import trades_logger, errors_logger
from utils.invariants_v2 import InvariantsV2
from signal_engine import SignalEngine
from conviction_engine import ConvictionEngine, ConvictionScore
from risk_engine import RiskEngine, FirmOrderRequest
from execution_engine import ExecutionEngine
from utils.exit_engine import ExitEngine, ExitAction
from state_engine import StateEngine
from utils.schemas import TradeSignal, PositionSchema
from utils.safe_core import SafeCore
from utils.lifecycle import validate_position_lifecycle
import config

class Pipeline:
    def __init__(self, 
                 dh,
                 signal_engine: SignalEngine,
                 conviction_engine: ConvictionEngine,
                 risk_engine: RiskEngine,
                 execution_engine: ExecutionEngine,
                 exit_engine: ExitEngine,
                 state_engine: StateEngine):
        self.dh = dh
        self.signal = signal_engine
        self.conviction = conviction_engine
        self.risk = risk_engine
        self.execution = execution_engine
        self.exit = exit_engine
        self.state = state_engine
        
        # Logging Enforcement Metrics
        self.signal_generated_count = 0
        self.signal_rejected_count = 0
        self.risk_rejection_count = 0
        self.execution_attempt_count = 0

    def run_cycle(self):
        """
        Executes a full system cycle across all symbols concurrently.
        Handles both entry logic and active position management (exits) in parallel.
        """
        try:
            # 1. STATE RECONCILIATION & SYNC [MANDATORY] (sequential for stable base state)
            self.state.reconcile_all(self.dh, self.execution)
            self.risk.update_metrics(self.dh.get_account_state().equity)

            import concurrent.futures

            # 2. POSITION MANAGEMENT (EXIT STAGE - PARALLEL)
            active_positions = list(self.state.active_positions)
            if active_positions:
                futures = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    for pos in active_positions:
                        future = executor.submit(self._process_single_exit, pos)
                        futures[future] = pos.symbol
                    
                    # Enforce strict 5-second timeout-isolation per cycle
                    done, not_done = concurrent.futures.wait(futures.keys(), timeout=5.0)
                    
                    for fut in done:
                        try:
                            fut.result()
                        except Exception as e:
                            errors_logger.error(f"[EXIT ERROR] Worker exception for {futures[fut]}: {e}", exc_info=True)
                            
                    for fut in not_done:
                        errors_logger.critical(f"[TIMEOUT] Exit worker for {futures[fut]} hung / exceeded timeout limit of 5.0s!")

            # 3. OPPORTUNITY SCAN (ENTRY STAGE - PARALLEL)
            import config
            if config.BOT_STATE.get('trading_enabled', True):
                account_state = self.dh.get_account_state()
                symbols_to_scan = [s for s in config.SYMBOLS if not self.state.is_position_active(s)]
                if symbols_to_scan:
                    futures = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        for sym in symbols_to_scan:
                            # Use helper function to submit entry to executor safely
                            future = executor.submit(self._process_single_entry, sym, account_state)
                            futures[future] = sym
                            
                        done, not_done = concurrent.futures.wait(futures.keys(), timeout=5.0)
                        
                        for fut in done:
                            try:
                                fut.result()
                            except Exception as e:
                                errors_logger.error(f"[ENTRY ERROR] Worker exception for {futures[fut]}: {e}", exc_info=True)
                                
                        for fut in not_done:
                            errors_logger.critical(f"[TIMEOUT] Entry worker for {futures[fut]} hung / exceeded timeout limit of 5.0s!")

        except Exception as e:
            errors_logger.error(f"[PIPELINE CRASH] Global cycle failure: {e}", exc_info=True)

    def _process_single_exit(self, pos: PositionSchema):
        """
        Evaluates ExitEngine for a single active position concurrently.
        """
        try:
            # Strict lifecycle invariant audit (repairable invariants are normalized here)
            v = validate_position_lifecycle(pos, allow_repair=True, hard_halt_on_unrecoverable=True)
            if v.repaired:
                self.state.persist()

            # FIX 2: Global Watchdog
            try:
                self.execution.watchdog_verify_sl(pos)
            except Exception:
                errors_logger.critical(
                    f"[WATCHDOG_LOOP_FATAL] {pos.symbol}",
                    exc_info=True
                )
                try:
                    self.execution.close_full(pos.symbol, pos)
                except Exception:
                    errors_logger.critical(
                        f"[FORCE_CLOSE_FAIL] symbol={pos.symbol}",
                        exc_info=True
                    )

            # If degraded, freeze exits and suspend lifecycle advancement, but allow watchdog to recover SL
            from utils.degraded_manager import DegradedExchangeManager, ExchangeDegradationState
            if DegradedExchangeManager().get_state() != ExchangeDegradationState.HEALTHY:
                trades_logger.warning(f"[DEGRADED_MODE] Degraded mode active for {pos.symbol}. Exits and lifecycle updates are frozen. Watchdog audit complete.")
                return

            # Refresh data for exit evaluation
            df = self.dh.fetch_ohlcv(pos.symbol, config.TIMEFRAME)
            df = self.signal.compute_all_signals(df, pos.symbol)
            ticker = self.dh.fetch_ticker(pos.symbol)
            last = SafeCore.last_row(df)
            
            if last is None or not ticker: return

            # minQty for exit sizing / dust protection
            min_qty = 0.0
            try:
                market = self.dh.exchange.market(self.dh.format_symbol(pos.symbol))
                min_qty = SafeCore.float(market.get("limits", {}).get("amount", {}).get("min", 0.0))
            except Exception:
                min_qty = 0.0

            # Update candles held for time decay exit
            last_ts = getattr(pos, "last_bar_ts", None)
            current_bar_ts = str(last.get('timestamp', ''))

            if current_bar_ts and last_ts != current_bar_ts:
                pos.candles_held = int(getattr(pos, "candles_held", 0) or 0) + 1
                pos.last_bar_ts = current_bar_ts

            action, amount, reason, new_stop = self.exit.compute_exit(
                pos=pos,
                current_price=ticker.last,
                current_atr=SafeCore.float(last.get('atr')),
                trend_bias=last.get('trend_bias', 'NEUTRAL'),
                regime=last.get('regime', 'RANGE'),
                min_qty=min_qty
            )

            if action == ExitAction.HOLD:
                return

            trades_logger.info(f"[EXIT ENGINE] {pos.symbol} Action: {action.value} | Reason: {reason}")

            if action == ExitAction.EMERGENCY_RECOVERY:
                trades_logger.warning(f"[EMERGENCY_PROTECTION_RECOVERY] Invalid state detected for {pos.symbol}. Attempting recovery...")
                self.state.recover_position_protection(self.dh, self.execution, pos)
                if pos.stop_loss <= 0:
                    errors_logger.critical(f"[HARD_HALT] Recovery failed for {pos.symbol}. Forcing emergency exit.")
                    if self.execution.close_full(pos.symbol, pos):
                        self.state.remove_position(pos.symbol)
                return

            if action == ExitAction.FULL_CLOSE:
                if self.execution.close_full(pos.symbol, pos):
                    self.state.remove_position(pos.symbol)
            
            elif action in (ExitAction.PARTIAL_CLOSE_TP1, ExitAction.PARTIAL_CLOSE_TP2):
                tp_level = 1 if action == ExitAction.PARTIAL_CLOSE_TP1 else 2

                # Strict event-sourced pending & done idempotency guards
                if tp_level == 1 and (getattr(pos, "tp1_done", False) or getattr(pos, "tp1_pending", False)):
                    trades_logger.warning(f"[IDEMPOTENCY_BLOCK] {pos.symbol} TP1 already done or pending confirmation. Blocking duplicate trigger.")
                    return
                if tp_level == 2 and (getattr(pos, "tp2_done", False) or getattr(pos, "tp2_pending", False)):
                    trades_logger.warning(f"[IDEMPOTENCY_BLOCK] {pos.symbol} TP2 already done or pending confirmation. Blocking duplicate trigger.")
                    return

                # Mark pending transition state prior to execution
                if tp_level == 1:
                    pos.tp1_pending = True
                else:
                    pos.tp2_pending = True
                self.state.persist()
                
                # Log event so it's committed to execution log
                from utils.journal import ExecutionJournal
                ExecutionJournal.log_event(pos.symbol, f"TP{tp_level}_TRIGGERED")

                result = self.execution.close_partial(pos.symbol, pos, amount, min_qty=min_qty)
                if result and result.get("filled_amount", 0.0) > 0:
                    self.state.mark_tp_done(pos.symbol, tp_level)
            
            elif action == ExitAction.UPDATE_STOP:
                if self.execution.atomic_stop_update(pos.symbol, pos, new_stop):
                    pos.stop_loss = new_stop
                    # trailing becomes valid only after TP1
                    if getattr(pos, "tp1_done", False):
                        pos.trailing_active = True
                        self.state.persist()
                    try:
                        self.execution.watchdog_verify_sl(pos)
                    except Exception as e:
                        errors_logger.critical(f"[SL_WATCHDOG_FAIL] symbol={pos.symbol}", exc_info=True)
                        self.execution.close_full(pos.symbol, pos)

        except Exception as e:
            errors_logger.error(f"[EXIT FAILURE] {pos.symbol}: {e}", exc_info=True)

    def _process_single_entry(self, symbol: str, account):
        """
        Executes SIGNAL -> CONVICTION -> RISK -> EXECUTION flow for a single symbol concurrently.
        """
        import config
        from utils.degraded_manager import DegradedExchangeManager
        if not config.BOT_STATE.get("is_running", True) or not DegradedExchangeManager().is_entry_allowed():
            return
            
        try:
            # A. SIGNAL STAGE
            df = self.dh.fetch_ohlcv(symbol, config.TIMEFRAME)
            df = self.signal.compute_all_signals(df, symbol)
            signal = self.signal.get_last_signal(symbol, df)

            if not signal.trigger: return

            import uuid
            intent_id = uuid.uuid4().hex

            self.signal_generated_count += 1

            # [PHASE 5] ENTRY SPAM HARD GUARD
            current_bar_ts = str(SafeCore.last_row(df).get('timestamp', ''))
            if not hasattr(self.state, "last_entry_bar_ts"):
                with self.state.lock:
                    if not hasattr(self.state, "last_entry_bar_ts"):
                        self.state.last_entry_bar_ts = {}
                
            with self.state.lock:
                already_entered = self.state.last_entry_bar_ts.get(symbol) == current_bar_ts
                
            if already_entered:
                return
            
            # [PHASE 6] STATE SYNCHRONIZATION FIX (Pre-entry check)
            # Prevent entering if exchange already has a position but state desynced
            exchange_positions = self.dh.fetch_active_positions(retries=1)
            ex_pos = next((p for p in exchange_positions if p.symbol == symbol), None)
            if ex_pos and not self.state.is_position_active(symbol):
                errors_logger.warning(f"[STATE_DESYNC] Exchange has position for {symbol} but state does not. Forcing sync.")
                self.state.reconcile_all(self.dh, self.execution)
                return

            with self.state.lock:
                self.state.last_entry_bar_ts[symbol] = current_bar_ts

            # B. CONVICTION STAGE
            start_time = time.time()
            ob = self.dh.fetch_orderbook(symbol)
            
            # [INVARIANTS -> SOFT GATES] Phase 6: Microstructure Multipliers
            micro_score = 1.0
            is_fallback = self.state.is_fallback_mode_active()
            
            if is_fallback:
                trades_logger.info(f"[FALLBACK MODE] Bypassing microstructure penalties for {symbol}")
            else:
                if not InvariantsV2.audit_latency(start_time):
                    trades_logger.info(f"[PENALTY: LATENCY] {symbol} (API too slow) -> 0.7x")
                    micro_score *= 0.7
                    
                if not InvariantsV2.audit_spread(ob):
                    trades_logger.info(f"[PENALTY: SPREAD] {symbol} (Spread too wide) -> 0.6x")
                    micro_score *= 0.6
            
            obi = InvariantsV2.audit_obi(ob)
            conviction = self.conviction.evaluate_signal(signal, obi)
            
            # Phase 6 & 13: Apply micro_score and log
            conviction.micro_score = micro_score
            trades_logger.info(f"[SIGNAL GENERATED] {symbol} {signal.trigger} | Base Score: {conviction.composite_score:.2f} | Micro Score: {micro_score:.2f}")

            # C. RISK STAGE
            ticker = self.dh.fetch_ticker(symbol)
            
            # SL Calc (2x ATR)
            last = SafeCore.last_row(df)
            atr = SafeCore.float(last.get('atr'))
            stop_dist = atr * 2.0
            stop_price = ticker.last - stop_dist if conviction.side == 'LONG' else ticker.last + stop_dist

            order_request = self.risk.approve_order(
                conviction=conviction,
                current_equity=account.equity,
                active_positions=self.state.active_positions,
                atr=atr,
                last_price=ticker.last,
                stop_loss_price=stop_price,
                used_margin=account.used_margin,
                intent_id=intent_id
            )

            # D. EXECUTION STAGE
            if order_request.approved:
                if order_request.amount < 0.001:
                    trades_logger.info(f"[SKIPPED: TOO SMALL] {symbol} {conviction.side} Size: {order_request.amount}")
                    return
                
                self.execution_attempt_count += 1
                trades_logger.info(f"[ENTRY APPROVED] {symbol} {conviction.side} (Size: {order_request.amount}, Risk Scaling: {conviction.composite_score * micro_score:.2f})")
                new_pos = self.execution.execute_firm_order(order_request)
                
                if new_pos:
                    new_pos.candles_held = 0
                    new_pos.atr = float(atr or 0.0)
                    # E. STATE SYNC (Immediate)
                    with self.state.lock:
                        self.state.active_positions.append(new_pos)
                        self.state.last_trade_times[symbol] = int(time.time())
                        self.state.persist()
                    try:
                        self.execution.watchdog_verify_sl(new_pos)
                    except Exception as e:
                        errors_logger.critical(f"[SL_WATCHDOG_FAIL] symbol={symbol}", exc_info=True)
                        self.execution.close_full(symbol, new_pos)
            else:
                self.risk_rejection_count += 1
                reason = order_request.rejection_reason or "UNKNOWN_REJECTION"
                trades_logger.info(f"[REJECTED: RISK_ENGINE] {symbol}: {reason}")

        except Exception as e:
            errors_logger.error(f"[ENTRY FAILURE] {symbol}: {e}", exc_info=True)
