import os
from dotenv import load_dotenv

# --- PHASE 1: Loading ENV ---
load_dotenv()

# --- PHASE 3: Safe Default System ---
def safe_config(name: str, default=None):
    """
    Returns the value from environment variables if present, 
    otherwise returns the specified default value.
    """
    return os.getenv(name, default)

def set_runtime_telegram_enabled(enabled: bool):
    """
    Runtime flag used after environment validation.
    This lets every module share the same Telegram safe-mode state.
    """
    normalized = bool(enabled)
    Config.TELEGRAM_ENABLED = normalized
    globals()["TELEGRAM_ENABLED"] = normalized

# --- PHASE 7: CONFIG CLASS STRUCTURE ---
class Config:
    # 1. Exchange & Auth
    EXCHANGE = safe_config("EXCHANGE", "binance").lower()
    
    # Binance Credentials (Dual Support)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", os.getenv("API_KEY", ""))
    BINANCE_SECRET = os.getenv("BINANCE_SECRET", os.getenv("BINANCE_API_SECRET", os.getenv("SECRET_KEY", "")))
    BINANCE_API_SECRET = BINANCE_SECRET

    # Bitget Credentials
    BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
    BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
    BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

    # Telegram Alerting
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("BINANCE_TELEGRAM_TOKEN", ""))
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", os.getenv("BINANCE_TELEGRAM_CHAT_ID", ""))
    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    DEMO_MODE = safe_config("DEMO_MODE", "false").lower() == "true"

    # Global Bot State
    BOT_STATE = {
        "is_running": True,
        "trading_enabled": True,   # NEW: Set to False if Position Mode is unknown (Read-Only)
        "stop_mode": "pause_only" # 'pause_only' or 'close_all_and_stop'
    }

    # 2. Institutional Risk Parameters (PHASE 1 CORE)
    STOP_LOSS_ATR_MULT = 1.5
    TARGET_RR = 2.0
    SYMBOL_COOLDOWN_SECONDS = 60
    MINIMUM_EQUITY_THRESHOLD = 50.0

    # Additional Risk Control
    RISK_PER_TRADE = 0.0075      # 0.75% equity risked per trade
    MAX_PORTFOLIO_RISK = 0.05    # 5% max risk across all positions
    MAX_CONCURRENT_POSITIONS = 3
    MAX_SLIPPAGE_PCT = 0.005     # 0.5% Hard limit for market order slippage
    # Institutional exposure ceilings (hard caps)
    MAX_NOTIONAL_MULTIPLIER = 3.0      # position_notional <= equity * multiplier
    MAX_EFFECTIVE_LEVERAGE = 3.0       # effective_leverage = notional / equity
    MIN_STOP_DISTANCE_PCT = 0.002      # 0.2% stop floor to prevent ATR-compression overflow
    RECONCILE_INTERVAL = 15      # 15s polling for reconciliation
    EQUITY_ROLLING_WINDOW_DAYS = 10 # 10-day peak for drawdown baseline
    COOLDOWN_AFTER_LOSSES = 3
    MAX_TRADES_PER_MINUTE = 2
    MAX_EQUITY_DROP_PCT = 0.05
    EQUITY_MONITOR_WINDOW_MIN = 30
    MAX_CONSECUTIVE_SYSTEM_FAILURES = 3

    # 3. Market Universe
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", 
        "XRPUSDT", "LINKUSDT", "AVAXUSDT", 
        "DOGEUSDT", "BNBUSDT", "SUIUSDT", "ADAUSDT"
    ]
    SYMBOL_OVERRIDES = {
        "DOGEUSDT": {
            "max_slippage_pct": 0.008
        },
        "SUIUSDT": {
            "max_slippage_pct": 0.007
        }
    }
    TIMEFRAME = "15m"
    POSITION_MODE = "one_way" # 'one_way', 'hedge'
    SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h"]

    # 4. Strategy Thresholds (v4 Transformation)
    BASE_RISK_PER_TRADE = 0.0075    # 0.75% base risk
    MAX_DRAWDOWN_THRESHOLD = 0.10   # 10% hard stop
    MAX_DRAWDOWN_REDUCTION = 0.05   # 5% drawdown -> Reduce risk by 50%
    SHARPE_LOOKBACK = 20            # trades for rolling metrics
    REGIME_MULTIPLIERS = {
        'TREND': 1.2,
        'RANGE': 0.6,
        'LOW_VOL': 0.8
    }

    # Engine Thresholds
    REGIME_ADX_THRESHOLD = 25
    REGIME_ATR_PCT_THRESHOLD = 20
    N_PERIOD_BREAKOUT = 20
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    # Trading Excellence
    MAX_SPREAD_PCT = 0.0015         # Max bid-ask spread allowed (0.15%)
    SIGNAL_SCORE_THRESHOLD = 2     # Minimum confluence score required
    MAX_TRADES_PER_HOUR = 3        # Max trades per symbol per hour
    SYMBOLS_PERFORMANCE_STATE = "state/performance.json"

# --- BACKWARD COMPATIBILITY EXPORTS ---
# To prevent Refactor-Breakage in main.py, export class attributes to module level
EXCHANGE = Config.EXCHANGE
BINANCE_API_KEY = Config.BINANCE_API_KEY
BINANCE_SECRET = Config.BINANCE_SECRET
BINANCE_API_SECRET = Config.BINANCE_API_SECRET
BITGET_API_KEY = Config.BITGET_API_KEY
BITGET_API_SECRET = Config.BITGET_API_SECRET
BITGET_PASSPHRASE = Config.BITGET_PASSPHRASE
TELEGRAM_BOT_TOKEN = Config.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = Config.TELEGRAM_CHAT_ID
TELEGRAM_ENABLED = Config.TELEGRAM_ENABLED
BOT_STATE = Config.BOT_STATE
SYMBOLS = Config.SYMBOLS
SYMBOL_OVERRIDES = Config.SYMBOL_OVERRIDES
TIMEFRAME = Config.TIMEFRAME
RISK_PER_TRADE = Config.RISK_PER_TRADE
MAX_PORTFOLIO_RISK = Config.MAX_PORTFOLIO_RISK
MAX_CONCURRENT_POSITIONS = Config.MAX_CONCURRENT_POSITIONS
COOLDOWN_AFTER_LOSSES = Config.COOLDOWN_AFTER_LOSSES
SYMBOL_COOLDOWN_SECONDS = Config.SYMBOL_COOLDOWN_SECONDS
MINIMUM_EQUITY_THRESHOLD = Config.MINIMUM_EQUITY_THRESHOLD
MAX_TRADES_PER_MINUTE = Config.MAX_TRADES_PER_MINUTE
MAX_EQUITY_DROP_PCT = Config.MAX_EQUITY_DROP_PCT
EQUITY_MONITOR_WINDOW_MIN = Config.EQUITY_MONITOR_WINDOW_MIN
MAX_CONSECUTIVE_SYSTEM_FAILURES = Config.MAX_CONSECUTIVE_SYSTEM_FAILURES
POSITION_MODE = Config.POSITION_MODE
BASE_RISK_PER_TRADE = Config.BASE_RISK_PER_TRADE
MAX_DRAWDOWN_THRESHOLD = Config.MAX_DRAWDOWN_THRESHOLD
MAX_DRAWDOWN_REDUCTION = Config.MAX_DRAWDOWN_REDUCTION
SHARPE_LOOKBACK = Config.SHARPE_LOOKBACK
REGIME_MULTIPLIERS = Config.REGIME_MULTIPLIERS
REGIME_ADX_THRESHOLD = Config.REGIME_ADX_THRESHOLD
REGIME_ATR_PCT_THRESHOLD = Config.REGIME_ATR_PCT_THRESHOLD
N_PERIOD_BREAKOUT = Config.N_PERIOD_BREAKOUT
RSI_OVERSOLD = Config.RSI_OVERSOLD
RSI_OVERBOUGHT = Config.RSI_OVERBOUGHT
MAX_SPREAD_PCT = Config.MAX_SPREAD_PCT
SIGNAL_SCORE_THRESHOLD = Config.SIGNAL_SCORE_THRESHOLD
MAX_TRADES_PER_HOUR = Config.MAX_TRADES_PER_HOUR
SYMBOLS_PERFORMANCE_STATE = Config.SYMBOLS_PERFORMANCE_STATE
MAX_SLIPPAGE_PCT = Config.MAX_SLIPPAGE_PCT
MAX_NOTIONAL_MULTIPLIER = Config.MAX_NOTIONAL_MULTIPLIER
MAX_EFFECTIVE_LEVERAGE = Config.MAX_EFFECTIVE_LEVERAGE
MIN_STOP_DISTANCE_PCT = Config.MIN_STOP_DISTANCE_PCT
RECONCILE_INTERVAL = Config.RECONCILE_INTERVAL
EQUITY_ROLLING_WINDOW_DAYS = Config.EQUITY_ROLLING_WINDOW_DAYS

# Core Indicators as per user Phase 1 mandatory
STOP_LOSS_ATR_MULT = Config.STOP_LOSS_ATR_MULT
TARGET_RR = Config.TARGET_RR
DEMO_MODE = Config.DEMO_MODE
SUPPORTED_TIMEFRAMES = Config.SUPPORTED_TIMEFRAMES
