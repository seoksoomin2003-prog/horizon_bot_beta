from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Ticker:
    symbol: str
    last: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        from utils.validator import normalize_symbol
        object.__setattr__(self, 'symbol', normalize_symbol(self.symbol))

@dataclass
class PositionSchema:
    symbol: str
    side: str # 'LONG' or 'SHORT'
    amount: float = 0.0
    initial_amount: float = 0.0
    entry_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    leverage: int = 1
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    sl_plan_id: Optional[str] = field(default=None)
    tp_plan_id: Optional[str] = field(default=None)
    risk_pct: float = 0.0075
    
    # Institutional State Tracking
    status: str = 'open'  # 'open' | 'missing_pending_confirmation' | 'closed'
    missing_since: float = 0.0
    missing_cycles: int = 0
    intended_qty: float = 0.0
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    cumulative_fill_ratio: float = 0.0
    lifecycle_state: str = 'OPEN'  # STRICT lifecycle state machine: OPEN|TP1_DONE|TP2_DONE|CLOSED
    candles_held: int = 0
    tp1_done: bool = False
    tp2_done: bool = False
    tp1_pending: bool = False
    tp2_pending: bool = False
    breakeven_active: bool = False
    trailing_active: bool = False
    regime: str = 'RANGE'
    timestamp: float = 0.0
    last_sync_ts: float = 0.0
    entry_time: str = ""
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None

    def __post_init__(self):
        from utils.validator import normalize_symbol
        self.symbol = normalize_symbol(self.symbol)

@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    trigger: Optional[str] = None # 'LONG', 'SHORT', or None
    confidence: float = 0.0
    score: int = 0
    regime: str = 'RANGE'
    trend_bias: Optional[str] = None

    def __post_init__(self):
        from utils.validator import normalize_symbol
        object.__setattr__(self, 'symbol', normalize_symbol(self.symbol))

from enum import Enum

class EquityState(str, Enum):
    VALID = "VALID"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"

@dataclass(frozen=True)
class AccountState:
    equity: float = 0.0
    used_margin: float = 0.0
    available_balance: float = 0.0
    timestamp: float = 0.0
    is_available: bool = True
    equity_state: EquityState = EquityState.VALID

ACCOUNT_STATE_UNAVAILABLE = AccountState(
    equity=0.0,
    used_margin=0.0,
    available_balance=0.0,
    timestamp=0.0,
    is_available=False,
    equity_state=EquityState.UNAVAILABLE
)

@dataclass(frozen=True)
class PerformanceTrade:
    symbol: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    timestamp: float = 0.0
    side: str = 'UNKNOWN'
