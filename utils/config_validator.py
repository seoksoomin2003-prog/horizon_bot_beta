import os
import logging
from typing import Dict, Any

# Configure loggers for independent validation messages
logger = logging.getLogger("config_validator")

# --- PHASE 1: ENV Categorization ---
SUPPORTED_EXCHANGES = {"binance", "bitget"}

REQUIRED_ENV_BY_EXCHANGE = {
    "binance": [
        "BINANCE_API_KEY",
        "BINANCE_SECRET",
    ],
    "bitget": [
        "BITGET_API_KEY",
        "BITGET_API_SECRET",
        "BITGET_PASSPHRASE",
    ],
}

OPTIONAL_ENV = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID"
]

# Use ABSOLUTE paths for Windows reliability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def validate_env() -> bool:
    """
    Validate only the selected exchange credentials and return Telegram status.
    Missing trading ENV for the active exchange must hard-fail the launcher.
    """
    exchange = os.getenv("EXCHANGE", "binance").strip().lower()
    if exchange not in SUPPORTED_EXCHANGES:
        raise Exception(f"[FATAL] Unsupported EXCHANGE: {exchange}")

    required_critical_env = REQUIRED_ENV_BY_EXCHANGE[exchange]
    optional_env = OPTIONAL_ENV

    # 1. Critical ENV Check
    for key in required_critical_env:
        val = os.getenv(key, "").strip()
        if not val:
            raise Exception(f"[FATAL] Missing ENV: {key}")

        # Phase 5: Security Check
        if key in {"BINANCE_API_KEY", "BITGET_API_KEY"} and len(val) < 20:
            raise Exception("Invalid API Key")

    if exchange == "binance":
        if len(os.getenv("BINANCE_SECRET", "").strip()) < 20:
            raise Exception("Invalid API Secret")
    elif exchange == "bitget":
        if len(os.getenv("BITGET_API_SECRET", "").strip()) < 20:
            raise Exception("Invalid API Secret")
        if len(os.getenv("BITGET_PASSPHRASE", "").strip()) < 8:
            raise Exception("Invalid API Passphrase")

    # 2. Optional ENV Check (Telegram)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BINANCE_TELEGRAM_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("BINANCE_TELEGRAM_CHAT_ID")

    telegram_enabled = bool(telegram_token and telegram_chat)

    if not telegram_enabled:
        logger.warning(
            "Telegram disabled. Missing optional env from %s",
            optional_env,
        )

    return telegram_enabled

def validate_config_values(config_obj: Any):
    """
    PHASE 3: TYPE SAFETY & RANGE VALIDATION
    """
    required_fields = {
        "STOP_LOSS_ATR_MULT": (int, float),
        "TARGET_RR": (int, float),
        "SYMBOL_COOLDOWN_SECONDS": (int, float),
        "MINIMUM_EQUITY_THRESHOLD": (int, float)
    }

    for field, expected_types in required_fields.items():
        if not hasattr(config_obj, field):
            raise Exception(f"[CONFIG ERROR] Missing field: {field}")
        
        value = getattr(config_obj, field)
        
        # Type Check
        if not isinstance(value, expected_types):
            raise Exception(f"[TYPE ERROR] {field} must be {expected_types}, got {type(value)}")
        
        # Range Check
        if value <= 0:
            raise Exception(f"[RANGE ERROR] {field} must be greater than 0, got {value}")

    # PHASE 4: Bitget Specific Validation
    if getattr(config_obj, "EXCHANGE", "").lower() == "bitget":
        supported = getattr(config_obj, "SUPPORTED_TIMEFRAMES", ["1m", "5m", "15m", "1h"])
        current_tf = getattr(config_obj, "TIMEFRAME", "")
        if current_tf not in supported:
            raise Exception(f"[CONFIG ERROR] Unsupported TIMEFRAME for Bitget: {current_tf}. Must be one of {supported}")

def get_config_summary(config_obj: Any) -> Dict[str, Any]:
    """
    Returns a redacted summary of the current configuration.
    """
    summary = {}
    tracked_fields = [
        "EXCHANGE", "SYMBOLS", "TIMEFRAME", "STOP_LOSS_ATR_MULT", 
        "TARGET_RR", "SYMBOL_COOLDOWN_SECONDS", "MINIMUM_EQUITY_THRESHOLD"
    ]
    for field in tracked_fields:
        if hasattr(config_obj, field):
            summary[field] = getattr(config_obj, field)
    return summary
