import time
import threading
import logging
from datetime import datetime
import config
from config import BOT_STATE
from logger import trades_logger, errors_logger
from utils.telegram_commands import send_message

def _monitor_loop(data_handler, position_manager, risk_engine):
    """
    Background loop that sends status report every 6 hours.
    """
    # 21600 seconds = 6 hours
    INTERVAL = 21600 
    
    trades_logger.info("Heartbeat Monitor started (6H interval).")
    
    while True:
        try:
            # Wait for the next interval
            time.sleep(INTERVAL)
            
            # 1. Collect Data
            exchange = config.EXCHANGE.upper()
            is_running = BOT_STATE.get("is_running", False)
            status_text = "RUNNING" if is_running else "STOPPED (PAUSED)"
            
            # Fetch real-time equity
            equity = data_handler.get_account_state().equity
            
            # Performance Metrics (Institutional v4)
            from config import SHARPE_LOOKBACK
            sharpe = risk_engine._calculate_sharpe(SHARPE_LOOKBACK)
            pf = risk_engine.get_profit_factor()
            drawdown = risk_engine.current_drawdown
            
            # Win Rate
            history = risk_engine.performance_history[-50:]
            wins = [t for t in history if t['pnl'] > 0]
            win_rate = (len(wins) / len(history)) if history else 0.0
            
            # Count active positions
            pos_count = len(position_manager.active_positions)
            
            # Current Time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 2. Build Message
            report = (
                f"INSTITUTIONAL STATUS (6H)\n\n"
                f"Exchange: <code>{exchange}</code>\n"
                f"Status: {status_text}\n"
                f"Balance: <b>{equity:.2f} USDT</b>\n"
                f"Drawdown: <b>{drawdown:.2%}</b>\n"
                f"-------------------\n"
                f"PERFORMANCE (Last 50)\n"
                f"Sharpe Ratio: <b>{sharpe:.2f}</b>\n"
                f"Profit Factor: <b>{pf:.2f}</b>\n"
                f"Win Rate: <b>{win_rate:.1%}</b>\n"
                f"Open Positions: <b>{pos_count}</b>\n"
                f"-------------------\n"
                f"Time: <code>{now}</code>\n"
            )
            
            # 3. Send to Telegram (Priority 1 = Normal)
            send_message(report, priority=1)
            trades_logger.info("6-Hour Institutional Heartbeat report sent.")
            
        except Exception as e:
            errors_logger.error(f"Heartbeat Monitor Error: {e}")
            time.sleep(60)

def start_heartbeat_monitor(data_handler, position_manager, risk_engine):
    """
    Launches the heartbeat monitor as a background daemon thread.
    """
    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(data_handler, position_manager, risk_engine),
        daemon=True,
        name="HeartbeatMonitorTask"
    )
    monitor_thread.start()
