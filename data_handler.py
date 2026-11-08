import pandas as pd
from typing import List
import ccxt
import time
import config
from utils.runtime_coordinator import RuntimeCoordinator
import threading
from logger import trades_logger, errors_logger
from utils.safe_core import SafeCore
from utils.validator import Validator
from utils.schemas import Ticker, PositionSchema, AccountState
from utils.exceptions import DataIntegrityError

class DataHandler:
    def __init__(self):
        self.exchange_name = config.EXCHANGE
        self._local = threading.local()
        
        from adapters import get_adapter
        self.adapter = get_adapter(self.exchange_name)
        
        # Prepare thread-safe exchange configuration
        if self.exchange_name == "binance":
            self._exchange_config = {
                'apiKey': config.BINANCE_API_KEY,
                'secret': config.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'timeout': 10000,
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
                'timeout': 10000,
                'options': {
                    'defaultType': 'swap'
                }
            }
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange_name}")

        trades_logger.info(f"Using exchange: {self.exchange_name.upper()}")
        self.last_equity = 0.0
        self.last_used_margin = 0.0

    def initialize_exchange_connection(self):
        """Perform all blocking network setup and validation routines."""
        self.validate_api_keys()
        self._safe_load_markets()
        if hasattr(self, "adapter") and self.adapter is not None:
            self.adapter.enforce_position_mode(self.exchange, config.POSITION_MODE, config.SYMBOLS, self.format_symbol)

    @property
    def exchange(self):
        """Thread-local exchange accessor. Creates instance per thread dynamically."""
        if not hasattr(self._local, 'instance'):
            if config.IS_PAPER:
                from paper.exchange import PaperExchange, get_broker
                self._local.instance = PaperExchange(self.exchange_name, get_broker())
                return self._local.instance
            if self.exchange_name == "binance":
                inst = ccxt.binance(self._exchange_config)
            elif self.exchange_name == "bitget":
                inst = ccxt.bitget(self._exchange_config)
                # [FIX] Demo Mode Header support
                if getattr(config, "DEMO_MODE", False):
                    inst.headers.update({"ACCESS-PAP": "true"})
                # [FIX] Time Synchronization
                inst.options["adjustForTimeDifference"] = True
            
            if hasattr(self, "adapter") and self.adapter is not None:
                if hasattr(self.adapter, "patch_exchange"):
                    self.adapter.patch_exchange(inst)

            self._local.instance = inst
        return self._local.instance

    def _safe_load_markets(self):
        """[C8] Exponential Backoff Market Loading (Daemon Mode)."""
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                self.exchange.load_markets()
                trades_logger.info(f"[OK] Markets loaded on attempt {attempt+1}")
                self.validate_supported_markets()
                RuntimeCoordinator().set_state("system_state", 'NORMAL')
                RuntimeCoordinator().set_state("allow_data_polling", True)
                RuntimeCoordinator().set_state("trading_enabled", True)
                return
            except Exception as e:
                backoff = min(60, 2 ** attempt)
                errors_logger.warning(f"Market load failed ({attempt+1}/{max_attempts}). Retrying in {backoff}s: {e}")
                
                if attempt >= max_attempts - 1:
                    errors_logger.critical("EXCH_INIT_FAIL: Bounded startup retries exhausted. Entering DEGRADED startup mode.")
                    RuntimeCoordinator().set_state("system_state", 'DEGRADED')
                    RuntimeCoordinator().set_state("allow_data_polling", False)
                    RuntimeCoordinator().set_state("trading_enabled", False)
                    return
                
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
            RuntimeCoordinator().set_state("trading_enabled", False)
        
        config.SYMBOLS = valid_symbols
        RuntimeCoordinator().set_state("active_symbols", valid_symbols)

    def validate_api_keys(self):
        """[PHASE 2] API Validation Guard."""
        attempt = 0
        max_attempts = 3
        while attempt < max_attempts:
            try:
                balance = self.exchange.fetch_balance()
                if balance is None:
                    raise ValueError("Balance returned None")
                trades_logger.info("[OK] API Keys validated and permissions are active.")
                return
            except (ccxt.AuthenticationError, ccxt.PermissionDenied) as auth_err:
                try:
                    from utils.telegram_commands import send_message
                    send_message(f"🚨 <b>CRITICAL_CONFIG_ERROR</b>\nAPI credentials rejected: {auth_err}")
                except:
                    pass
                errors_logger.critical(f"CRITICAL_CONFIG_ERROR: API credentials rejected: {auth_err}")
                raise RuntimeError(f"CRITICAL_CONFIG_ERROR: {auth_err}")
            except Exception as e:
                attempt += 1
                backoff = 2 ** attempt
                errors_logger.warning(f"API key validation attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(backoff)
                else:
                    errors_logger.critical(f"EXCH_INIT_FAIL: API key validation failed after {max_attempts} attempts. Entering DEGRADED startup mode.")
                    RuntimeCoordinator().set_state("system_state", 'DEGRADED')
                    RuntimeCoordinator().set_state("allow_data_polling", False)
                    RuntimeCoordinator().set_state("trading_enabled", False)
                    return

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
        
        if hasattr(self, "adapter") and self.adapter is not None:
            return self.adapter.format_symbol(symbol)
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

    def get_account_state(self) -> AccountState:
        """
        Fetch real-time account data as a typed schema.
        """
        try:
            balance = self.exchange.fetch_balance()
            acc_state = Validator.balance(balance)
            self.last_equity = acc_state.equity
            self.last_used_margin = acc_state.used_margin
            return acc_state
        except Exception as e:
            errors_logger.error(f"Account State Integrity Failure: {e}")
            from utils.degraded_manager import DegradedExchangeManager
            DegradedExchangeManager().report_failure(f"Account State fetch failure: {e}")
            from utils.schemas import ACCOUNT_STATE_UNAVAILABLE
            return ACCOUNT_STATE_UNAVAILABLE

    def get_position_mode(self, symbol=None):
        """
        Returns 'hedge' or 'one_way' aligned with config.POSITION_MODE.
        """
        mode = self.adapter.get_position_mode(self.exchange)
        # Normalize returned mode string to 'one_way' or 'hedge'
        if mode == 'oneway':
            return 'one_way'
        return mode


