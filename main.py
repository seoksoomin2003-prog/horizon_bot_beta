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
    
    import config
    STOP_FLAG_PATH = "/tmp/trading_bot.stop"
    if os.path.exists(STOP_FLAG_PATH):
        config.BOT_STATE["is_running"] = False
        trades_logger.warning("STOP FLAG DETECTED — BOT STARTING IN PAUSED MODE")

    # A. Environment & Safety Audit
    LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'horizon.lock')
    if InvariantsV2.audit_process_lock(LOCK_PATH):
        try:
            with open(LOCK_PATH, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is actually alive
            os.kill(pid, 0)
            trades_logger.critical(f"Another process detected (PID {pid}) — HALT")
            sys.exit(1)
        except (ValueError, ProcessLookupError, OSError):
            trades_logger.warning("Stale lock detected. Auto-clearing system lock.")
            os.remove(LOCK_PATH)

    # Credential Check for Live Mode
    if not config.DEMO_MODE:
        active_key = config.BINANCE_API_KEY if config.EXCHANGE == "binance" else config.BITGET_API_KEY
        if not active_key or "your_" in active_key:
            trades_logger.critical(f"!!! LIVE MODE DETECTED WITH MISSING/PLACEHOLDER KEYS [{config.EXCHANGE.upper()}] !!!")
            trades_logger.critical("Check your .env file and replace placeholders with real API keys.")
            sys.exit(1)

    # create lock AFTER check
    if not os.path.exists(os.path.dirname(LOCK_PATH)):
        os.makedirs(os.path.dirname(LOCK_PATH))
    with open(LOCK_PATH, 'w') as f:
        f.write(str(os.getpid()))

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
        send_message("🚀 <b>Horizon Bot V3 Online</b>\nLive Execution Pipeline Active.")

        trades_logger.info("V3 System Active. Main Loop Started [LIVE MODE].")

        # D. Continuous Execution Pipeline
        while True:
            if not config.BOT_STATE.get('is_running', True):
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
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
        trades_logger.info("System Halted.")

if __name__ == "__main__":
    run_v3()
