"""
main.py — Horizon Bot v3 Entry Point

Institutional-grade controller for the unified Pipeline.
"""

import os
import time
import sys
from logger import trades_logger, errors_logger

# 1. CORE ENGINES (Unified v3)
import config
from utils.runtime_coordinator import RuntimeCoordinator
from data_handler import DataHandler
from signal_engine import SignalEngine
from conviction_engine import ConvictionEngine
from risk_engine import RiskEngine
from execution_engine import ExecutionEngine
from state_engine import StateEngine
from pipeline import Pipeline
from utils.exit_engine import ExitEngine
from utils.invariants_v2 import InvariantsV2

def run_v3():
    trades_logger.info("Initializing Horizon Bot v3 [FULL RECONSTRUCTION]...")
    
    from utils.persistence import get_absolute_path
    STOP_FLAG_PATH = get_absolute_path("trading_bot.stop")
    if os.path.exists(STOP_FLAG_PATH):
        RuntimeCoordinator().set_state("is_running", False)
        trades_logger.warning("STOP FLAG DETECTED — BOT STARTING IN PAUSED MODE")

    # A. Environment & Safety Audit
    from utils.persistence import get_absolute_path, ProcessLock
    LOCK_PATH = get_absolute_path("horizon.lock")
    process_lock = ProcessLock(LOCK_PATH)
    if not process_lock.acquire():
        trades_logger.critical(f"Failed to acquire exclusive system lock. Dual live instance prevented. Halt.")
        sys.exit(1)

    # Credential Check for Live Mode
    if not config.DEMO_MODE:
        active_key = config.BINANCE_API_KEY if config.EXCHANGE == "binance" else config.BITGET_API_KEY
        if not active_key or "your_" in active_key:
            trades_logger.critical(f"!!! LIVE MODE DETECTED WITH MISSING/PLACEHOLDER KEYS [{config.EXCHANGE.upper()}] !!!")
            trades_logger.critical("Check your .env file and replace placeholders with real API keys.")
            process_lock.release()
            sys.exit(1)

    try:
        # B. Dependency Injection
        dh = DataHandler()
        signal = SignalEngine()
        conviction = ConvictionEngine()
        risk = RiskEngine()
        # Initializing execution with LIVE mode
        execution = ExecutionEngine(dh, simulation_mode=False)
        exit_e = ExitEngine()
        
        # Enforce 'Journal Always Wins' state verification prior to StateEngine load
        from utils.journal import ExecutionJournal
        ExecutionJournal.verify_and_align_snapshot()
        
        state = StateEngine()
        execution.state = state
        execution.recover_pending_sl_registry()

        # C. Pipeline Orchestration
        bot_pipeline = Pipeline(
            dh=dh,
            signal_engine=signal,
            conviction_engine=conviction,
            risk_engine=risk,
            execution_engine=execution,
            exit_engine=exit_e,
            state_engine=state
        )

        # D. Start Telegram C2
        from utils.telegram_commands import start_telegram_listener, send_message
        start_telegram_listener(pos_manager=state, execution_engine=execution, dh=dh)
        send_message("🚀 <b>Horizon Bot V3 Online</b>\nInitializing connection to exchange...")

        # Initialize connection (blocking/bounded setup calls)
        dh.initialize_exchange_connection()

        send_message("✅ <b>Exchange Connection Initialized</b>\nLive Execution Pipeline Active.")
        trades_logger.info("V3 System Active. Main Loop Started [LIVE MODE].")

        # D. Continuous Execution Pipeline
        while True:
            # Process Telegram Commands
            for cmd_obj in RuntimeCoordinator().drain_commands():
                cmd = cmd_obj.get("command")
                if cmd == "PAUSE":
                    RuntimeCoordinator().set_state("is_running", False)
                    RuntimeCoordinator().set_state("stop_mode", "pause_only")
                    try:
                        from utils.persistence import get_absolute_path
                        with open(get_absolute_path("trading_bot.stop"), "w") as f:
                            f.write("STOP")
                    except Exception: pass
                    send_message("[STOP] BOT STOPPED\nTrading halted.", priority=0)
                elif cmd == "STOP_ALL":
                    RuntimeCoordinator().set_state("is_running", False)
                    RuntimeCoordinator().set_state("stop_mode", "close_all_and_stop")
                    try:
                        from utils.persistence import get_absolute_path
                        with open(get_absolute_path("trading_bot.stop"), "w") as f:
                            f.write("STOP")
                    except Exception: pass
                    try:
                        if state:
                            state.close_all_positions(execution, dh)
                        send_message("[STOP] BOT STOPPED\nAll positions closed.", priority=0)
                    except Exception as e:
                        send_message(f"[ERROR] Emergency close failed: {e}", priority=0)
                elif cmd == "RESUME":
                    RuntimeCoordinator().set_state("is_running", True)
                    try:
                        from utils.persistence import get_absolute_path
                        path = get_absolute_path("trading_bot.stop")
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception: pass
                    send_message("[START] BOT STARTED\nTrading resumed.", priority=0)

            if not RuntimeCoordinator().get_state('is_running', True):
                trades_logger.warning("BOT PAUSED — skipping execution cycle")
                time.sleep(1.0)
                continue
                
            cycle_start = time.time()
            
            bot_pipeline.run_cycle()
            
            # Dynamic Cycle Control
            elapsed = time.time() - cycle_start
            sleep_time = max(0, config.RECONCILE_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        trades_logger.info("Shutdown requested. Cleaning up...")
    except Exception as e:
        errors_logger.critical(f"UNRECOVERABLE SYSTEM CRASH: {e}", exc_info=True)
    finally:
        process_lock.release()
        trades_logger.info("System Halted.")

if __name__ == "__main__":
    run_v3()
