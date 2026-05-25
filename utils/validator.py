from typing import Any, List, Dict, Optional
from utils.schemas import Ticker, PositionSchema, AccountState, PerformanceTrade
from utils.safe_core import SafeCore
from utils.exceptions import DataIntegrityError
import time

class Validator:
    @staticmethod
    def ticker(raw: Any, symbol: str) -> Ticker:
        """API -> TICKER typed object."""
        raw = SafeCore.ensure_dict(raw)
        if not raw:
            raise DataIntegrityError(f"Empty ticker for {symbol}")
        
        last = SafeCore.float(raw.get('last'))
        bid = SafeCore.float(raw.get('bid'))
        ask = SafeCore.float(raw.get('ask'))
        
        if last <= 0:
            raise DataIntegrityError(f"Invalid price (<= 0) for {symbol}")
            
        return Ticker(
            symbol=symbol,
            last=last,
            bid=bid,
            ask=ask,
            timestamp=SafeCore.float(raw.get('timestamp'), time.time())
        )

    @staticmethod
    def position(raw: Any) -> PositionSchema:
        """API -> POSITION typed object."""
        raw = SafeCore.ensure_dict(raw)
        symbol = raw.get('symbol')
        if not symbol:
            raise DataIntegrityError("Position data missing symbol")
            
        contracts = SafeCore.float(raw.get('contracts', 0.0) or raw.get('size', 0.0))
        entry_price = SafeCore.float(raw.get('entryPrice', 0.0))
        
        # C-01 FIX: Institutional Side Reconstruction
        raw_side = str(raw.get('side', '') or raw.get('positionSide', '') or '').upper()
        
        # In One-Way mode, sometimes side is returned in info.side
        if not raw_side and isinstance(raw.get('info'), dict):
            raw_side = str(raw['info'].get('side', '') or raw['info'].get('positionSide', '')).upper()

        if 'SHORT' in raw_side or 'SELL' in raw_side:
            side = 'SHORT'
        elif 'LONG' in raw_side or 'BUY' in raw_side:
            side = 'LONG'
        else:
            # Fallback to size sign ONLY IF absolutely necessary as legacy fallback, 
            # but we prioritize explicit failure over blind long enforcement
            size_val = SafeCore.float(raw.get('size', 0.0))
            if size_val < 0:
                side = 'SHORT'
            elif size_val > 0:
                side = 'LONG'
            else:
                # Failsafe for zero or ambiguous state
                raise DataIntegrityError(f"UNABLE TO DETERMINE POSITION SIDE: Raw Side '{raw_side}' is ambiguous")
        
        return PositionSchema(
            symbol=symbol,
            side=side,
            amount=abs(contracts),
            entry_price=entry_price,
            pnl=SafeCore.float(raw.get('unrealizedPnl', 0.0) or raw.get('pnl', 0.0)),
            leverage=int(SafeCore.float(raw.get('leverage', 1.0))),
            sl_plan_id=raw.get('sl_plan_id'),
            tp_plan_id=raw.get('tp_plan_id'),
            take_profit=SafeCore.float(raw.get('take_profit', 0.0)),
            stop_loss=SafeCore.float(raw.get('stop_loss', 0.0)),
            atr=SafeCore.float(raw.get('atr', 0.0)),
            # Institutional State Fields (from persistence)
            status=str(raw.get('status', 'open')),
            lifecycle_state=str(raw.get('lifecycle_state', raw.get('lifecycle', 'OPEN'))),
            candles_held=int(SafeCore.float(raw.get('candles_held', 0))),
            tp1_done=bool(raw.get('tp1_done', False)),
            tp2_done=bool(raw.get('tp2_done', False)),
            breakeven_active=bool(raw.get('breakeven_active', False)),
            trailing_active=bool(raw.get('trailing_active', False)),
            regime=str(raw.get('regime', 'RANGE')),
            timestamp=SafeCore.float(raw.get('timestamp', time.time())),
            last_sync_ts=SafeCore.float(raw.get('last_sync_ts', time.time())),
            entry_time=str(raw.get('entry_time', ''))
        )

    @staticmethod
    def balance(raw: Any) -> AccountState:
        """API -> ACCOUNTSTATE typed object."""
        raw = SafeCore.ensure_dict(raw)
        # Bitget V2 standard structure in CCXT
        usdt = SafeCore.ensure_dict(raw.get('USDT'))
        if not usdt:
            raise DataIntegrityError("Balance USDT key missing")
            
        return AccountState(
            equity=SafeCore.float(usdt.get('total', 0.0)),
            used_margin=SafeCore.float(usdt.get('used', 0.0)),
            available_balance=SafeCore.float(usdt.get('free', 0.0)),
            timestamp=time.time()
        )

    @staticmethod
    def performance(raw: Any) -> PerformanceTrade:
        """DICT/LIST -> PERFORMANCETRADE typed object."""
        # Institutional Integrity: Normalize legacy performance data
        if not isinstance(raw, dict):
            # Legacy list-of-floats detection
            pnl = SafeCore.float(raw)
            return PerformanceTrade(symbol="LEGACY", pnl=pnl)
            
        return PerformanceTrade(
            symbol=raw.get('symbol', 'UNKNOWN'),
            pnl=SafeCore.float(raw.get('pnl', 0.0)),
            pnl_pct=SafeCore.float(raw.get('pnl_pct', 0.0)),
            r_multiple=SafeCore.float(raw.get('r', 0.0)),
            timestamp=SafeCore.float(raw.get('timestamp', time.time())),
            side=raw.get('side', 'UNKNOWN')
        )
