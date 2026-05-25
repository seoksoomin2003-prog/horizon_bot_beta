import pandas as pd
from typing import List
import numpy as np
import ccxt
import time
import config
import threading
from logger import trades_logger, errors_logger
from utils.safe_core import SafeCore
from utils.validator import Validator
from utils.schemas import Ticker, PositionSchema, AccountState
from utils.exceptions import DataIntegrityError
from utils.safe_ops import safe_float

class DataHandler:
    def __init__(self):
        self.exchange_name = config.EXCHANGE
        self._local = threading.local()
        
        # Prepare thread-safe exchange configuration
        if self.exchange_name == "binance":
            self._exchange_config = {
                'apiKey': config.BINANCE_API_KEY,
                'secret': config.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            }
        elif self.exchange_name == "bitget":
            self._exchange_config = {
                'apiKey': config.BITGET_API_KEY,
                'secret': config.BITGET_API_SECRET,
                'password': config.BITGET_PASSPHRASE,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'
                }
            }
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange_name}")

        trades_logger.info(f"Using exchange: {self.exchange_name.upper()}")
        self.last_equity = 0.0
        self.last_used_margin = 0.0
        self.validate_api_keys()
        self._safe_load_markets() # Robust init
        
        # [FIX] Enforce configured position mode at startup
        if self.exchange_name == "bitget":
            self.enforce_bitget_position_mode()

    @property
    def exchange(self):
        """Thread-local exchange accessor. Creates instance per thread dynamically."""
        if not hasattr(self._local, 'instance'):
            if self.exchange_name == "binance":
                inst = ccxt.binance(self._exchange_config)
            elif self.exchange_name == "bitget":
                inst = ccxt.bitget(self._exchange_config)
                # [FIX] Demo Mode Header support
                if getattr(config, "DEMO_MODE", False):
                    inst.headers.update({"ACCESS-PAP": "true"})
                # [FIX] Time Synchronization
                inst.options["adjustForTimeDifference"] = True
            
            self._local.instance = inst
        return self._local.instance

    def _safe_load_markets(self):
        """[C8] Exponential Backoff Market Loading (Daemon Mode)."""
        attempt = 0
        while True:
            try:
                self.exchange.load_markets()
                trades_logger.info(f"[OK] Markets loaded on attempt {attempt+1}")
                self.validate_supported_markets()
                config.BOT_STATE['system_state'] = 'NORMAL'
                config.BOT_STATE['allow_data_polling'] = True
                config.BOT_STATE['trading_enabled'] = True
                return
            except Exception as e:
                backoff = min(60, 2 ** attempt)
                errors_logger.warning(f"Market load failed ({attempt+1}). Retrying in {backoff}s: {e}")
                
                if attempt >= 4:
                    errors_logger.critical("EXCH_INIT_FAIL: Entering DEGRADED state. Daemon keeps retrying.")
                    config.BOT_STATE['system_state'] = 'DEGRADED'
                    config.BOT_STATE['allow_data_polling'] = False
                    config.BOT_STATE['trading_enabled'] = False
                
                time.sleep(backoff)
                attempt += 1

    def validate_supported_markets(self):
        """[EXPANSION] Verify symbols exist in loaded markets. Safely drop unsupported ones."""
        valid_symbols = []
        for symbol in config.SYMBOLS:
            formatted = self.format_symbol(symbol)
            if formatted in self.exchange.markets:
                market = self.exchange.markets[formatted]
                if market.get('active', True):
                    valid_symbols.append(symbol)
                    trades_logger.info(f"[MARKET_VALIDATED] {symbol} supported and active.")
                else:
                    trades_logger.warning(f"[MARKET_SKIPPED] {symbol} is inactive on exchange.")
            else:
                trades_logger.warning(f"[MARKET_UNSUPPORTED] {symbol} does not exist on {self.exchange_name}.")
        
        # Override runtime SYMBOLS safely
        if len(valid_symbols) == 0:
            errors_logger.critical("[CRITICAL] No valid markets found. Trading disabled.")
            config.BOT_STATE['trading_enabled'] = False
        
        config.SYMBOLS = valid_symbols
        config.BOT_STATE['active_symbols'] = valid_symbols

    def validate_api_keys(self):
        """[PHASE 2] API Validation Guard."""
        try:
            balance = self.exchange.fetch_balance()
            if balance is None:
                raise ValueError("Balance returned None")
            trades_logger.info("[OK] API Keys validated and permissions are active.")
        except Exception as e:
            try:
                from utils.telegram_commands import send_message
                send_message("🚨 <b>CRITICAL_CONFIG_ERROR</b>\nAPI Invalid, lacks permissions, or IP not whitelisted.")
            except:
                pass
            errors_logger.critical(f"CRITICAL_CONFIG_ERROR: API Invalid or IP not whitelisted: {e}")
            raise RuntimeError(f"CRITICAL_CONFIG_ERROR: {e}")

    def format_symbol(self, symbol):
        """
        Normalize symbol for the current exchange.
        Expected input: 'BTCUSDT' or 'BTC/USDT'
        """
        # Ensure base/quote format for formatting
        if '/' not in symbol:
            # Simple heuristic for USDT pairs if slash is missing
            if symbol.endswith('USDT'):
                base = symbol.replace('USDT', '')
                symbol = f"{base}/USDT"
        
        if self.exchange_name == "binance":
            return symbol # CCXT binance handles BTC/USDT or BTCUSDT
        elif self.exchange_name == "bitget":
            # Bitget Futures in CCXT often expects BTC/USDT:USDT
            if ':' not in symbol:
                return f"{symbol}:USDT"
            return symbol
        return symbol

    def validate_ohlcv(self, df):
        """[C9] Strict OHLCV Data Validation."""
        if df is None or df.empty:
            raise ValueError("OHLCV Data is EMPTY")
        if len(df) < 50:
            raise ValueError(f"Insufficient OHLCV data: {len(df)} rows")
        if df['close'].isnull().any():
            raise ValueError("OHLCV Data contains NULL values")

    def fetch_ohlcv(self, symbol, timeframe, limit=300):
        """
        Fetches OHLCV data and returns a pandas DataFrame.
        """
        symbol = self.format_symbol(symbol)
        retries = 3
        for attempt in range(retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                if not SafeCore.validate_ohlcv(df):
                    raise DataIntegrityError(f"Insufficient or invalid OHLCV for {symbol}")
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df
            except Exception as e:
                backoff = 2 ** attempt
                errors_logger.error(f"Error fetching OHLCV for {symbol} on attempt {attempt+1}: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
        return pd.DataFrame()

    def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch the latest ticker price data as a typed object.
        """
        try:
            formatted = self.format_symbol(symbol)
            raw = self.exchange.fetch_ticker(formatted)
            return Validator.ticker(raw, symbol)
        except Exception as e:
            errors_logger.error(f"Ticker Integrity Failure for {symbol}: {e}")
            raise DataIntegrityError(f"Could not fetch valid ticker for {symbol}")
            
    def fetch_orderbook(self, symbol, limit=10):
        """
        Fetch current orderbook to determine spread.
        """
        try:
            symbol = self.format_symbol(symbol)
            ob = self.exchange.fetch_order_book(symbol, limit)
            return ob
        except Exception as e:
            errors_logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

    def fetch_active_positions(self, retries: int = 3) -> List[PositionSchema]:
        """
        Fetch currently open positions from the exchange as typed schemas.
        """
        for attempt in range(retries):
            try:
                raw_positions = self.exchange.fetch_positions()
                positions = SafeCore.ensure_list(raw_positions)
                
                active = []
                for p in positions:
                    normalized = Validator.position(p)
                    if normalized.amount > 0:
                        active.append(normalized)
                return active
            except Exception as e:
                errors_logger.error(f"Position sync attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        
        raise DataIntegrityError("Critical Position Sync Failure")

    def get_real_average_fill(self, symbol: str, order_id: str, retries: int = 3) -> tuple:
        """
        [EXECUTION TRUTH] Fetches actual trades for an order to determine 
        weighted average fill price and total quantity.
        Returns: (avg_price, total_amount)
        """
        formatted_symbol = self.format_symbol(symbol)
        for attempt in range(retries):
            try:
                # CCXT fetch_my_trades
                raw_trades = self.exchange.fetch_my_trades(formatted_symbol, limit=20)
                trades = SafeCore.ensure_list(raw_trades)
                
                # Filter trades by order_id
                matched = [t for t in trades if str(t.get('order')) == str(order_id)]
                
                if matched:
                    total_amount = sum([SafeCore.float(t.get('amount')) for t in matched])
                    weighted_sum = sum([SafeCore.float(t.get('price')) * SafeCore.float(t.get('amount')) for t in matched])
                    
                    if total_amount > 0:
                        avg_price = weighted_sum / total_amount
                        trades_logger.info(f"[TRUTH] Resolved average fill for {order_id}: {avg_price} (Qty: {total_amount})")
                        return avg_price, total_amount
                
                # Small delay if not found yet (orderbook sync lag)
                time.sleep(1.0 + attempt)
            except Exception as e:
                errors_logger.warning(f"Trade truth lookup attempt {attempt+1} failed: {e}")
                time.sleep(1.0)
        
        return 0.0, 0.0

    def parse_plan_order_status(self, order):
        """
        Bitget V2 custom planStatus parser.
        [C8] Standard CCXT status can be None for Plan Orders.
        """
        try:
            raw_info = order.get('info', {})
            # Bitget V2 planStatus: 'not_triggered', 'executing', 'triggered', 'fail', 'cancel'
            plan_status = raw_info.get('planStatus', '').lower()
            
            if plan_status in ['not_triggered', 'live']:
                return 'open'
            elif plan_status in ['executing', 'triggered']:
                return 'closed' # Or 'triggered' depending on needs
            elif plan_status in ['fail', 'cancel']:
                return 'canceled'
            return order.get('status', 'unknown')
        except:
            return 'unknown'

    def get_account_state(self) -> AccountState:
        """
        Fetch real-time account data as a typed schema.
        """
        try:
            balance = self.exchange.fetch_balance()
            return Validator.balance(balance)
        except Exception as e:
            # Fallback to cached equity if possible, otherwise raise
            errors_logger.error(f"Account State Integrity Failure: {e}")
            return AccountState(equity=self.last_equity, used_margin=self.last_used_margin)

    def get_used_margin(self):
        """[POLICY 1] Institutional Margin Monitoring."""
        try:
            balance = self.exchange.fetch_balance()
            # Bitget V2 Margin used is found in 'used' field of USDT
            used = safe_float(balance.get('USDT', {}).get('used'), 0.0)
            self.last_used_margin = used
            return used
        except Exception as e:
            errors_logger.error(f"Error fetching used margin: {e}")
            return self.last_used_margin

    def sync_time_difference(self):
        """[C7] Periodic Time Drift Correction."""
        try:
            self.exchange.load_time_difference()
            trades_logger.info(f"[SYNC] Clock drift corrected. Offset: {self.exchange.milliseconds_offset}ms")
        except Exception as e:
            errors_logger.error(f"Failed to sync time: {e}")

    def get_position_mode(self, symbol=None):
        """
        Returns 'hedge' or 'one_way' aligned with config.POSITION_MODE.
        Safe detection with Bitget V2 Fallback.
        """
        if self.exchange_name != "bitget":
            return None
            
        from config import SYMBOLS
        sym = self.format_symbol(symbol or SYMBOLS[0])
        
        try:
            self.exchange.load_markets()
            # 1. Attempt Unified CCXT method
            if self.exchange.has.get('fetchPositionMode'):
                raw = self.exchange.fetch_position_mode(sym)
                if isinstance(raw, dict):
                    hedged = raw.get('hedged')
                    if hedged is True: return 'hedge'
                    if hedged is False: return 'one_way'
            
            # 2. FALLBACK - Bitget V2 API [PART 1 HARDENING]
            # CCXT naming: private_mix_get_v2_mix_account_account (or similar depending on version)
            # We use a robust lookup to avoid attribute errors
            fallback_methods = [
                'private_mix_get_v2_mix_account_account',
                'private_get_v2_mix_account_account',
                'privateGetV2MixAccountAccount'
            ]
            
            method = None
            for m in fallback_methods:
                if hasattr(self.exchange, m):
                    method = getattr(self.exchange, m)
                    break
            
            if method:
                # Bitget V2 account endpoint requires marginCoin, NOT symbol
                response = method({"productType": "USDT-FUTURES", "marginCoin": "USDT"})
                data = response.get("data", {})
                hold_mode = data.get("holdMode") # 'single_hold' or 'double_hold'
                
                if hold_mode == "single_hold": return "one_way"
                if hold_mode == "double_hold": return "hedge"
            
            return None

        except Exception as e:
            errors_logger.error(f"[ERROR] Bitget Position Mode Detection Failed: {e}")
            return None

    def enforce_bitget_position_mode(self):
        """
        Force the configured position mode on Bitget for all symbols.
        """
        if self.exchange_name != "bitget":
            return

        from config import POSITION_MODE, SYMBOLS
        is_hedged = True if POSITION_MODE == "hedge" else False
        mode_text = POSITION_MODE.upper()

        trades_logger.info(f"Enforcing {mode_text} position mode on Bitget...")
        for symbol in SYMBOLS:
            formatted_symbol = self.format_symbol(symbol)
            try:
                # CCXT unified method: set_position_mode(hedged=bool)
                self.exchange.set_position_mode(is_hedged, formatted_symbol)
                trades_logger.info(f"Position mode confirmed {mode_text} for {symbol}")
            except Exception as e:
                # Fails if already set, which is fine
                error_str = str(e).lower()
                if "already" in error_str or "no change" in error_str or "30030" in error_str:
                    trades_logger.info(f"{symbol} already in {mode_text} mode.")
                else:
                    errors_logger.warning(f"Could not enforce {mode_text} mode for {symbol}: {e}")
