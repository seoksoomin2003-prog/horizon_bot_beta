import sys
import os
import logging
from logging.handlers import RotatingFileHandler

# --- ABSOLUTE EARLY-BOOT ENCODING ENFORCEMENT (WINDOWS RELIABILITY) ---
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def setup_logger():
    """
    Setup logging for trades and errors.
    Returns: (trades_logger, errors_logger)
    """
    # Use ABSOLUTE paths to prevent Windows drive/current-dir ambiguity
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    import config
    log_dir = os.environ.get("HORIZON_LOG_DIR") or os.path.join(BASE_DIR, 'logs', 'paper' if config.IS_PAPER else '')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    trades_log_file = os.path.join(log_dir, 'trades.log')
    errors_log_file = os.path.join(log_dir, 'errors.log')

    # Shared formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Trades Logger
    trades_logger = logging.getLogger('trades_logger')
    trades_logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not trades_logger.handlers:
        # Standardize UTF-8 for log files
        th = RotatingFileHandler(trades_log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
        th.setFormatter(formatter)
        
        # Safe Stream Handler for Windows terminals
        ch_t = logging.StreamHandler(sys.stdout)
        ch_t.setFormatter(formatter)
        
        trades_logger.addHandler(th)
        trades_logger.addHandler(ch_t)

    # Errors Logger
    errors_logger = logging.getLogger('errors_logger')
    errors_logger.setLevel(logging.ERROR)
    
    # Avoid duplicate handlers
    if not errors_logger.handlers:
        eh = RotatingFileHandler(errors_log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
        eh.setFormatter(formatter)
        
        ch_e = logging.StreamHandler(sys.stderr)
        ch_e.setFormatter(formatter)
        
        errors_logger.addHandler(eh)
        errors_logger.addHandler(ch_e)

    return trades_logger, errors_logger

trades_logger, errors_logger = setup_logger()
