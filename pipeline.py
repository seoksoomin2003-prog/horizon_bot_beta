"""
pipeline.py — Horizon Bot v3 Crystalline Orchestrator

SINGLE CONTINUOUS EXECUTION PIPELINE:
SIGNAL -> CONVICTION -> RISK -> EXECUTION -> EXIT -> STATE_SYNC
"""

import pandas as pd
import time
import threading
from logger import trades_logger, errors_logger
from utils.invariants_v2 import InvariantsV2
from signal_engine import SignalEngine
from conviction_engine import ConvictionEngine
from risk_engine import RiskEngine
from execution_engine import ExecutionEngine
from utils.exit_engine import ExitEngine, ExitAction
from state_engine import StateEngine
from utils.schemas import PositionSchema
from utils.safe_core import SafeCore
from utils.lifecycle import validate_position_lifecycle
import config
from utils.runtime_coordinator import RuntimeCoordinator

# Well above the normal 15 s cycle (config.RECONCILE_INTERVAL), so only a real sleep or stall reads as a wake-up.
WAKE_GAP_SECONDS = 60.0
# An entry worker in this state has passed its last check and is sending an order, so it must not be flagged as hung
# or replaced, or the same symbol could be ordered twice. The limit only guards against a thread that never returns.
LEASE_ORDERING = "ORDERING"
ORDER_IN_FLIGHT_LIMIT_SECONDS = 120.0
CLAIMED, STALE_WORKER, POSITION_LIMIT = "CLAIMED", "STALE_WORKER", "POSITION_LIMIT"


def trend_was_with_position_since_entry(df, side: str, opened_at: float) -> bool:
    """True if the trend bias matched the position on the entry bar or later, so an opposing bias now is a real flip.

    Unknown or out-of-window entry times return True, which keeps the plain trend-exit behaviour for those positions.
    """
    if not opened_at or opened_at <= 0 or df is None or df.empty:
        return True
    bar_open = (df["timestamp"] - pd.Timestamp(0)).dt.total_seconds()
    started = bar_open <= opened_at
    if not started.any() or bool(started.iloc[0]) is False:
        return True
    entry_bar = started[started].index[-1]
    return bool((df.loc[entry_bar:, "trend_bias"] == side).any())


class WorkerLease:
    def __init__(self, symbol: str, worker_type: str):
        import uuid
        self.symbol = symbol
        self.worker_type = worker_type
        self.start_time = time.time()
        self.heartbeat = time.time()
        self.thread_id = 0
        self.status = "ACTIVE"
        self.generation_id = uuid.uuid4().hex

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
        
        # Concurrency Shield Registry for active workers (Target A)
        import concurrent.futures
        self._worker_lock = threading.Lock()
        self._entry_gate_lock = threading.Lock()
        self.active_entry_workers = {} # {symbol: WorkerLease}
        self.active_exit_workers = {}  # {symbol: WorkerLease}
        
        # Persistent ThreadPool Executors for CCXT Connection Reuse
        self.exit_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="PipelineExitWorker")
        self.entry_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="PipelineEntryWorker")
        self.last_tick_time = time.monotonic()
        
        # Logging Enforcement Metrics
        self.signal_generated_count = 0
        self.signal_rejected_count = 0
        self.risk_rejection_count = 0
        self.execution_attempt_count = 0

    def shutdown(self):
        try:
            self.exit_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.entry_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def run_cycle(self):
        """
        [PHASE 3] Unified Run Cycle with exact Step Sequencing and Concurrency.
        """
        cycle_failed = False
        try:
            # Sleep-Wake Watchdog Detection
            now_mono = time.monotonic()
            # Must exceed the normal cycle (RECONCILE_INTERVAL), or every ordinary tick reads as a wake-up.
            if now_mono - self.last_tick_time > WAKE_GAP_SECONDS:
                errors_logger.critical(
                    f"[SYSTEM_WAKE] System sleep or major tick timing gap detected (elapsed: {now_mono - self.last_tick_time:.1f}s). "
                    f"Invalidating all active leases and clearing thread registries to prevent avalanches."
                )
                with self._worker_lock:
                    for lease in list(self.active_entry_workers.values()):
                        lease.status = "STALLED"
                    for lease in list(self.active_exit_workers.values()):
                        lease.status = "STALLED"
                    self.active_entry_workers.clear()
                    self.active_exit_workers.clear()
            self.last_tick_time = now_mono

            # 1. STATE RECONCILIATION & SYNC [MANDATORY] (sequential for stable base state)
            self.state.reconcile_all(self.dh, self.execution)
            
            # Fetch and verify Account State (Fail-Closed Policy)
            account_state = self.dh.get_account_state()
            is_available = getattr(account_state, "is_available", True)
            account_verified = is_available and (account_state.equity > 0 or is_available)
            
            if not account_verified:
                errors_logger.critical("[BALANCE UNCERTAINTY] Account balance is currently unavailable or unverified. Freezing all entry activities.")
            
            self.risk.update_metrics(account_state.equity, is_available=is_available)
            
            # Enforce Dead System Health Circuit Breakers
            system_ok, health_reason = self.risk.check_system_health(account_state.equity, is_available=is_available)
            if not system_ok:
                if health_reason.startswith("ENTRY RATE LIMIT"):
                    trades_logger.warning(f"[ENTRY PAUSED] {health_reason}")
                else:
                    errors_logger.critical(f"[SYSTEM HEALTH BREACH] {health_reason} — Freezing all entry activities.")

            import concurrent.futures

            # 2. POSITION MANAGEMENT (EXIT STAGE - PARALLEL)
            active_positions = list(self.state.active_positions)
            if active_positions:
                futures = {}
                try:
                    for pos in active_positions:
                        with self._worker_lock:
                            lease = self.active_exit_workers.get(pos.symbol)
                            now = time.time()
                            if lease is not None:
                                elapsed = now - lease.start_time
                                if elapsed > 10.0 or lease.status == "STALLED":
                                    errors_logger.warning(
                                        f"[LEASE_EXPIRED] Exit worker lease expired/stalled for {pos.symbol} (elapsed: {elapsed:.1f}s). "
                                        f"Reclaiming lease and initiating recovery worker."
                                    )
                                    self.active_exit_workers.pop(pos.symbol, None)
                                else:
                                    errors_logger.warning(f"[CONCURRENCY_SHIELD] Exit worker for {pos.symbol} is active (elapsed: {elapsed:.1f}s). Skipping duplicate submission.")
                                    continue
                            
                            new_lease = WorkerLease(pos.symbol, "EXIT")
                            self.active_exit_workers[pos.symbol] = new_lease
                            from utils.retry_scheduler import WorkerGenerationRegistry
                            WorkerGenerationRegistry.set_active_generation(pos.symbol, "EXIT", new_lease.generation_id)
                            
                        future = self.exit_executor.submit(self._process_single_exit, pos, new_lease)
                        futures[future] = pos.symbol
                    
                    # Enforce strict 5-second timeout-isolation per cycle
                    done, not_done = concurrent.futures.wait(futures.keys(), timeout=5.0)
                    
                    for fut in done:
                        symbol = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            cycle_failed = True
                            self.risk.record_system_failure()
                            errors_logger.error(f"[EXIT ERROR] Worker exception for {symbol}: {e}", exc_info=True)
                        finally:
                            pass # Worker cleanup is managed by thread in finally block with lease verification
                            
                    for fut in not_done:
                        symbol = futures[fut]
                        if fut.done():
                            try:
                                fut.result()
                            except Exception as e:
                                cycle_failed = True
                                self.risk.record_system_failure()
                            continue
                            
                        fut.cancel() # Unfinished task cancellation
                        cycle_failed = True
                        self.risk.record_system_failure()
                        errors_logger.critical(f"[TIMEOUT] Exit worker for {symbol} hung / exceeded timeout limit of 5.0s!")
                        
                        # 5. Transition worker into WORKER_STALLED state
                        with self._worker_lock:
                            lease = self.active_exit_workers.get(symbol)
                            if lease:
                                lease.status = "STALLED"
                        
                        # 6. Block duplicate, emit event, trigger degraded diagnostics
                        try:
                            from utils.event_sourced_runtime import OwnershipRegistry
                            owner = OwnershipRegistry().get_owner(symbol)
                            owner.submit_event("WORKER_STALLED", {"worker_type": "EXIT"})
                        except Exception as ev_err:
                            errors_logger.error(f"Failed to emit WORKER_STALLED event for {symbol}: {ev_err}")
                        
                        from utils.degraded_manager import DegradedExchangeManager
                        DegradedExchangeManager().report_failure(f"Exit worker stalled for {symbol}")
                except Exception as ex_run:
                    errors_logger.error(f"Failed to execute exit parallel cycle: {ex_run}")

            # 3. OPPORTUNITY SCAN (ENTRY STAGE - PARALLEL)
            if RuntimeCoordinator().get_state('trading_enabled', True) and system_ok and account_verified:
                symbols_to_scan = [s for s in config.SYMBOLS if not self.state.is_position_active(s)]
                if symbols_to_scan:
                    futures = {}
                    try:
                        for sym in symbols_to_scan:
                            with self._worker_lock:
                                lease = self.active_entry_workers.get(sym)
                                now = time.time()
                                if lease is not None:
                                    elapsed = now - lease.start_time
                                    if lease.status == LEASE_ORDERING and elapsed <= ORDER_IN_FLIGHT_LIMIT_SECONDS:
                                        trades_logger.warning(f"[ORDER_IN_FLIGHT] Entry order for {sym} still running (elapsed: {elapsed:.1f}s). Skipping duplicate submission.")
                                        continue
                                    if elapsed > 10.0 or lease.status == "STALLED":
                                        errors_logger.warning(
                                            f"[LEASE_EXPIRED] Entry worker lease expired/stalled for {sym} (elapsed: {elapsed:.1f}s). "
                                            f"Reclaiming lease and initiating recovery worker."
                                        )
                                        self.active_entry_workers.pop(sym, None)
                                    else:
                                        errors_logger.warning(f"[CONCURRENCY_SHIELD] Entry worker for {sym} is active (elapsed: {elapsed:.1f}s). Skipping duplicate submission.")
                                        continue
                                
                                new_lease = WorkerLease(sym, "ENTRY")
                                self.active_entry_workers[sym] = new_lease
                                from utils.retry_scheduler import WorkerGenerationRegistry
                                WorkerGenerationRegistry.set_active_generation(sym, "ENTRY", new_lease.generation_id)
                                
                            future = self.entry_executor.submit(self._process_single_entry, sym, account_state, new_lease)
                            futures[future] = sym
                            
                        done, not_done = concurrent.futures.wait(futures.keys(), timeout=5.0)
                        
                        for fut in done:
                            symbol = futures[fut]
                            try:
                                fut.result()
                            except Exception as e:
                                cycle_failed = True
                                self.risk.record_system_failure()
                                errors_logger.error(f"[ENTRY ERROR] Worker exception for {symbol}: {e}", exc_info=True)
                            finally:
                                pass # Worker cleanup managed inside thread
                                
                        for fut in not_done:
                            symbol = futures[fut]
                            if fut.done():
                                try:
                                    fut.result()
                                except Exception as e:
                                    cycle_failed = True
                                    self.risk.record_system_failure()
                                continue
                                
                            # 5. Transition worker into WORKER_STALLED state, unless its order is already being sent.
                            # Same lock as _claim_entry_order, so a worker is either stopped here or ordering, never both.
                            with self._worker_lock:
                                lease = self.active_entry_workers.get(symbol)
                                order_in_flight = bool(lease and lease.status == LEASE_ORDERING)
                                if lease and not order_in_flight:
                                    lease.status = "STALLED"
                            if order_in_flight:
                                trades_logger.warning(f"[ORDER_IN_FLIGHT] Entry order for {symbol} took over 5.0s and is still running.")
                                continue

                            fut.cancel() # Unfinished task cancellation
                            cycle_failed = True
                            self.risk.record_system_failure()
                            errors_logger.critical(f"[TIMEOUT] Entry worker for {symbol} hung / exceeded timeout limit of 5.0s!")

                            # 6. Block duplicate, emit event, trigger degraded diagnostics
                            try:
                                from utils.event_sourced_runtime import OwnershipRegistry
                                owner = OwnershipRegistry().get_owner(symbol)
                                owner.submit_event("WORKER_STALLED", {"worker_type": "ENTRY"})
                            except Exception as ev_err:
                                errors_logger.error(f"Failed to emit WORKER_STALLED event for {symbol}: {ev_err}")
                            
                            from utils.degraded_manager import DegradedExchangeManager
                            DegradedExchangeManager().report_failure(f"Entry worker stalled for {symbol}")
                    except Exception as ex_run:
                        errors_logger.error(f"Failed to execute entry parallel cycle: {ex_run}")

            # If the entire cycle completes without a new critical breach, reset system failures
            if not cycle_failed and account_verified:
                self.risk.reset_system_failures()

        except Exception as e:
            cycle_failed = True
            self.risk.record_system_failure()
            errors_logger.error(f"[PIPELINE CRASH] Global cycle failure: {e}", exc_info=True)

    def _process_single_exit(self, pos: PositionSchema, lease: WorkerLease):
        """
        Evaluates ExitEngine for a single active position concurrently.
        """
        with self._worker_lock:
            lease.thread_id = threading.get_ident()
            from utils.retry_scheduler import WorkerGenerationRegistry
            WorkerGenerationRegistry.set_thread_generation(pos.symbol, "EXIT", lease.generation_id)
            WorkerGenerationRegistry.verify_active()
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
                self.risk.record_system_failure()
                try:
                    self.execution.close_full(pos.symbol, pos)
                except Exception:
                    errors_logger.critical(
                        f"[FORCE_CLOSE_FAIL] symbol={pos.symbol}",
                        exc_info=True
                    )



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
                if config.IS_PAPER and config.PAPER_MARKET == "spot" and ticker.last > 0:
                    # Spot sells below the minimum notional are refused, so partial exits that small are skipped.
                    from paper.exchange import order_minimums
                    spot_qty, spot_cost, _ = order_minimums(self.dh.exchange, self.dh.format_symbol(pos.symbol), "spot")
                    min_qty = max(spot_qty, spot_cost / ticker.last)
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
                min_qty=min_qty,
                trend_flipped=trend_was_with_position_since_entry(df, pos.side, getattr(pos, "timestamp", 0.0))
            )

            if action == ExitAction.HOLD:
                return

            trades_logger.info(f"[EXIT ENGINE] {pos.symbol} Action: {action.value} | Reason: {reason}")

            if action == ExitAction.EMERGENCY_RECOVERY:
                trades_logger.warning(f"[EMERGENCY_PROTECTION_RECOVERY] Invalid state detected for {pos.symbol}. Attempting recovery...")
                self.state.recover_position_protection(self.dh, self.execution, pos)
                if pos.stop_loss <= 0:
                    errors_logger.critical(f"[HARD_HALT] Recovery failed for {pos.symbol}. Forcing emergency exit.")
                    try:
                        entry_equity = getattr(self.dh, "last_equity", 1000.0) or 1000.0
                        risk_amount = entry_equity * getattr(pos, "risk_pct", 0.0075)
                        exit_price = ticker.last
                        pnl = (exit_price - pos.entry_price) * pos.amount if pos.side == 'LONG' else (pos.entry_price - exit_price) * pos.amount
                        self.risk.record_trade_result(pos.symbol, pnl, risk_amount, entry_equity)
                    except Exception as err:
                        errors_logger.error(f"Failed to record trade result on emergency recovery failure: {err}")
                    
                    if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
                        if self.execution.close_full(pos.symbol, pos):
                            self.state.remove_position(pos.symbol)
                    else:
                        from utils.event_sourced_runtime import ProtectionOrchestrator
                        if ProtectionOrchestrator().execute_emergency_close(pos.symbol, pos, self.execution):
                            self.state.remove_position(pos.symbol)
                return

            if action == ExitAction.FULL_CLOSE:
                try:
                    entry_equity = getattr(self.dh, "last_equity", 1000.0) or 1000.0
                    risk_amount = entry_equity * getattr(pos, "risk_pct", 0.0075)
                    exit_price = ticker.last
                    pnl = (exit_price - pos.entry_price) * pos.amount if pos.side == 'LONG' else (pos.entry_price - exit_price) * pos.amount
                    self.risk.record_trade_result(pos.symbol, pnl, risk_amount, entry_equity)
                except Exception as err:
                    errors_logger.error(f"Failed to record trade result on full close: {err}")
                
                if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
                    if self.execution.close_full(pos.symbol, pos):
                        self.state.remove_position(pos.symbol)
                else:
                    from utils.event_sourced_runtime import ProtectionOrchestrator
                    if ProtectionOrchestrator().execute_emergency_close(pos.symbol, pos, self.execution):
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
                if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
                    if tp_level == 1:
                        pos.tp1_pending = True
                    else:
                        pos.tp2_pending = True
                    self.state.persist()
                else:
                    from utils.event_sourced_runtime import OwnershipRegistry
                    owner = OwnershipRegistry().get_owner(pos.symbol)
                    event_type = "TP1_TRIGGERED" if tp_level == 1 else "TP2_TRIGGERED"
                    owner.submit_event(event_type, {"amount": amount})
                
                # Log event so it's committed to execution log
                from utils.journal import ExecutionJournal, JournalEvent
                ExecutionJournal.log_event(pos.symbol, JournalEvent.TP1_TRIGGERED if tp_level == 1 else JournalEvent.TP2_TRIGGERED)

                stop_before = pos.sl_plan_id
                result = self.execution.close_partial(
                    pos.symbol,
                    pos, 
                    amount, 
                    min_qty=min_qty, 
                    new_stop=new_stop if tp_level == 1 else None
                )
                if result and result.get("filled_amount", 0.0) > 0:
                    filled = result.get("filled_amount")
                    order_id = result.get("order_id")
                    
                    try:
                        entry_equity = getattr(self.dh, "last_equity", 1000.0) or 1000.0
                        fraction_closed = filled / pos.initial_amount if pos.initial_amount > 0 else 0.0
                        risk_amount = (entry_equity * getattr(pos, "risk_pct", 0.0075)) * fraction_closed
                        exit_price = ticker.last
                        pnl = (exit_price - pos.entry_price) * filled if pos.side == 'LONG' else (pos.entry_price - exit_price) * filled
                        self.risk.record_trade_result(pos.symbol, pnl, risk_amount, entry_equity)
                    except Exception as err:
                        errors_logger.error(f"Failed to record trade result on partial TP exit: {err}")
                    
                    if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
                        if tp_level == 1:
                            pos.tp1_order_id = order_id
                        else:
                            pos.tp2_order_id = order_id
                        self.state.mark_tp_done(pos.symbol, tp_level)
                    else:
                        from utils.event_sourced_runtime import OwnershipRegistry
                        owner = OwnershipRegistry().get_owner(pos.symbol)
                        event_type = "TP1_FILLED" if tp_level == 1 else "TP2_FILLED"
                        owner.submit_event(event_type, {
                            "amount": filled,
                            "order_id": order_id
                        })
                        # The partial close replaced the stop, and positions are rebuilt from events.
                        if pos.sl_plan_id != stop_before:
                            owner.submit_event("SL_MOVED", {"price": pos.stop_loss, "order_id": pos.sl_plan_id})

            elif action == ExitAction.UPDATE_STOP:
                if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
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
                else:
                    from utils.event_sourced_runtime import ProtectionOrchestrator
                    if ProtectionOrchestrator().migrate_stop_loss(pos.symbol, pos, new_stop, self.execution):
                        try:
                            self.execution.watchdog_verify_sl(pos)
                        except Exception as e:
                            errors_logger.critical(f"[SL_WATCHDOG_FAIL] symbol={pos.symbol}", exc_info=True)
                            ProtectionOrchestrator().execute_emergency_close(pos.symbol, pos, self.execution)

        except Exception as e:
            errors_logger.error(f"[EXIT FAILURE] {pos.symbol}: {e}", exc_info=True)
        finally:
            with self._worker_lock:
                current_lease = self.active_exit_workers.get(pos.symbol)
                if current_lease and current_lease.thread_id == threading.get_ident():
                    self.active_exit_workers.pop(pos.symbol, None)

    def _claim_entry_order(self, symbol: str, lease: WorkerLease) -> str:
        """Last check before an order, returning CLAIMED or why not.

        Workers approved together all saw the same position count, so it is counted again here, one claim at a
        time, with orders still being sent included. An order that finishes between the two reads below is in
        the lease list, the position list or both, never neither. The position read is slow, so it stays outside
        _worker_lock, which the main cycle needs.
        """
        with self._entry_gate_lock:
            with self._worker_lock:
                sending = {s for s, other in self.active_entry_workers.items() if other.status == LEASE_ORDERING}
            taken = sending | {p.symbol for p in self.state.active_positions}
            with self._worker_lock:
                if lease.status == "STALLED" or self.active_entry_workers.get(symbol) is not lease:
                    return STALE_WORKER
                if len(taken - {symbol}) >= config.MAX_CONCURRENT_POSITIONS:
                    return POSITION_LIMIT
                lease.status = LEASE_ORDERING
                return CLAIMED

    def _process_single_entry(self, symbol: str, account, lease: WorkerLease):
        """
        Executes SIGNAL -> CONVICTION -> RISK -> EXECUTION flow for a single symbol concurrently.
        """
        with self._worker_lock:
            lease.thread_id = threading.get_ident()
            from utils.retry_scheduler import WorkerGenerationRegistry
            WorkerGenerationRegistry.set_thread_generation(symbol, "ENTRY", lease.generation_id)
            WorkerGenerationRegistry.verify_active()
        from utils.degraded_manager import DegradedExchangeManager
        if not RuntimeCoordinator().get_state("is_running", True) or not DegradedExchangeManager().is_entry_allowed():
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
                intent_id=intent_id,
                account_verified=getattr(account, "is_available", True)
            )

            # D. EXECUTION STAGE
            if order_request.approved and config.IS_PAPER:
                from paper.exchange import fit_order, order_minimums
                fitted, note = fit_order(
                    order_request.amount, ticker.last, stop_price, conviction.side, account.equity,
                    getattr(account, "used_margin", 0.0) or 0.0,
                    minimums=order_minimums(self.dh.exchange, self.dh.format_symbol(symbol), config.PAPER_MARKET),
                    market=config.PAPER_MARKET, leverage=config.PAPER_LEVERAGE, raise_to_minimum=config.PAPER_MIN_ORDER)
                if fitted is None:
                    trades_logger.info(f"[SKIPPED: PAPER_RULES] {symbol} {conviction.side}: {note}")
                    return
                if note:
                    trades_logger.info(f"[MIN_ORDER] {symbol} {conviction.side}: {note}")
                order_request.amount = fitted

            if order_request.approved:
                if not config.IS_PAPER and order_request.amount < 0.001:
                    trades_logger.info(f"[SKIPPED: TOO SMALL] {symbol} {conviction.side} Size: {order_request.amount}")
                    return

                # Hard slippage limit, measured on the order book for this order's size. An order that cannot be bounded is not sent.
                book_slip = InvariantsV2.book_slippage(ob, conviction.side, order_request.amount)
                slip_limit = config.SYMBOL_OVERRIDES.get(symbol, {}).get("max_slippage_pct", config.MAX_SLIPPAGE_PCT)
                if book_slip is None or book_slip > slip_limit:
                    self.risk_rejection_count += 1
                    measured = "order book missing or too thin for the size" if book_slip is None else f"{book_slip:.3%} expected"
                    trades_logger.info(f"[REJECTED: SLIPPAGE] {symbol} {conviction.side}: {measured}, limit {slip_limit:.3%}, no order sent")
                    return

                if not self.risk.reserve_entry_slot():
                    self.risk_rejection_count += 1
                    trades_logger.info(f"[REJECTED: RATE_LIMIT] {symbol}: entry limit per minute reached")
                    return

                claim = self._claim_entry_order(symbol, lease)
                if claim != CLAIMED:
                    self.risk.release_entry_slot()
                    if claim == POSITION_LIMIT:
                        self.risk_rejection_count += 1
                        trades_logger.info(f"[REJECTED: POSITION_LIMIT] {symbol}: position limit reached by an entry at the same moment, no order sent")
                    else:
                        trades_logger.warning(f"[ENTRY ABORTED: STALE WORKER] {symbol}: worker timed out before ordering, no order sent")
                    return

                self.execution_attempt_count += 1
                trades_logger.info(f"[ENTRY APPROVED] {symbol} {conviction.side} (Size: {order_request.amount}, Risk Scaling: {conviction.composite_score * micro_score:.2f})")
                new_pos = self.execution.execute_firm_order(order_request)
                
                if new_pos:
                    entry_fill = SafeCore.float(getattr(new_pos, "entry_price", 0.0))
                    if entry_fill > 0 and ticker.last > 0:
                        paid = (entry_fill / ticker.last - 1) * (1 if conviction.side == 'LONG' else -1)
                        trades_logger.info(f"[SLIPPAGE] {symbol} {conviction.side}: filled {paid:+.3%} from the decision price (book estimate {book_slip:.3%})")
                    new_pos.candles_held = 0
                    new_pos.atr = float(atr or 0.0)
                    # Positions are re-projected from events, so the entry ATR must be an event too.
                    try:
                        from utils.event_sourced_runtime import OwnershipRegistry
                        OwnershipRegistry().get_owner(symbol).submit_event("RECONCILE_CORRECTION", {"atr": new_pos.atr, "timestamp": time.time()})
                    except Exception as atr_err:
                        errors_logger.error(f"[ATR_EVENT_FAILED] {symbol}: {atr_err}")
                    self.risk.record_execution_event(symbol, slot_reserved=True)
                    # E. STATE SYNC (Immediate)
                    with self.state.lock:
                        try:
                            from utils.journal import ExecutionJournal, JournalEvent
                            ExecutionJournal.log_event(symbol, JournalEvent.DUST_CLEARED)
                        except Exception as je:
                            errors_logger.critical(f"[EVENT_COMMIT_FAILED] Failed to append execution journal log event for {symbol}: {je}. HALTING TRADING.")
                            RuntimeCoordinator().set_state("trading_enabled", False)
                            RuntimeCoordinator().set_state("is_running", False)
                            raise RuntimeError(f"EVENT_COMMIT_FAILED: {je}") from je

                        self.state.last_entry_bar_ts[symbol] = current_bar_ts # Commit cooldown only after successful execution confirmation!
                        self.state.save_last_entry_bar_ts()
                        self.state.last_trade_times[symbol] = int(time.time())
                        
                        # Only append and persist in legacy override/test mode!
                        if hasattr(self.state, "_active_positions_override") and self.state._active_positions_override is not None:
                            self.state.active_positions.append(new_pos)
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
        finally:
            with self._worker_lock:
                current_lease = self.active_entry_workers.get(symbol)
                if current_lease and current_lease.thread_id == threading.get_ident():
                    self.active_entry_workers.pop(symbol, None)
