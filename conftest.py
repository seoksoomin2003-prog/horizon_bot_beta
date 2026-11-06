"""Pytest isolation: runs before any project import, so tests cannot see .env or touch real runtime or log folders."""
import os
import tempfile

_root = tempfile.mkdtemp(prefix="horizon_test_")
os.environ.update({
    "TRADING_MODE": "paper", "HORIZON_RUNTIME_DIR": _root, "HORIZON_LOG_DIR": os.path.join(_root, "logs"),
    "BINANCE_API_KEY": "", "API_KEY": "", "BINANCE_SECRET": "", "BINANCE_API_SECRET": "", "SECRET_KEY": "",
    "BITGET_API_KEY": "", "BITGET_API_SECRET": "", "BITGET_PASSPHRASE": "",
    "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "", "BINANCE_TELEGRAM_TOKEN": "", "BINANCE_TELEGRAM_CHAT_ID": "",
})
