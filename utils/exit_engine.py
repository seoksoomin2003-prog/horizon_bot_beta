from enum import Enum
from logger import trades_logger
from utils.safe_core import SafeCore
from utils.schemas import PositionSchema

class ExitAction(Enum):
    HOLD = "HOLD"
    PARTIAL_CLOSE_TP1 = "PARTIAL_CLOSE_TP1"
    PARTIAL_CLOSE_TP2 = "PARTIAL_CLOSE_TP2"
    FULL_CLOSE = "FULL_CLOSE"
    UPDATE_STOP = "UPDATE_STOP"
    EMERGENCY_RECOVERY = "EMERGENCY_RECOVERY"

class ExitEngine:
    def __init__(self):
        # Constants for Min Update Delta (0.1% to prevent rate-limits)
        self.MIN_UPDATE_DELTA_PCT = 0.001
        self.VOL_SPIKE_MULT = 3.0
        self.VOL_COLLAPSE_MULT = 0.7
        self.TIME_EXIT_CANDLES = 288  # 15m = 3 days

    def compute_exit(self, pos: PositionSchema, current_price: float, current_atr: float, trend_bias: str, regime: str, min_qty: float = 0.0):
        """
        Hardened Institutional Exit Computation (Typed Schema version).
        Returns: (ExitAction, amount_to_close, reason, new_stop_price)
        """
        # 0. Defensive Guard: State Validation [A3]
        if not pos or pos.entry_price <= 0 or pos.stop_loss <= 0:
            return ExitAction.EMERGENCY_RECOVERY, 0, "INVALID_STATE", None
            
        if getattr(pos, "status", "") == "dust_quarantined":
            from logger import trades_logger
            trades_logger.debug(f"[{pos.symbol}] Ignoring exit logic for terminal dust.")
            return ExitAction.HOLD, 0.0, "DUST_QUARANTINE", None
        
        entry = pos.entry_price
        stop = pos.stop_loss
        atr_entry = pos.atr # Schema uses 'atr'
        side = pos.side
        amount = pos.amount
        
        # 1. R-Multiple Calculation
        risk = abs(entry - stop)
        if risk <= 0: 
            # Rebuild risk dynamically to prevent Zombie State paralysis
            risk = current_atr * 2.0
            if risk <= 0:
                risk = entry * 0.05 # absolute fallback
        
        if side == 'LONG':
            profit_r = SafeCore.div((current_price - entry), risk)
        else:
            profit_r = SafeCore.div((entry - current_price), risk)
        
        profit_r = round(profit_r, 4)

        # 2. VOLATILITY SPIKE EXIT
        candle_range = abs(current_price - entry)
        if current_atr > 0 and candle_range > (atr_entry * self.VOL_SPIKE_MULT):
            return ExitAction.FULL_CLOSE, amount, "VOL_SPIKE_PROTECTION", None

        # 3. VOLATILITY COLLAPSE
        if current_atr < (atr_entry * self.VOL_COLLAPSE_MULT):
            return ExitAction.FULL_CLOSE, amount, "VOL_COLLAPSE_EXIT", None

        # 4. TREND FLIP
        if regime in ['TREND', 'LOW_VOL']:
            if (side == 'LONG' and trend_bias == 'SHORT') or (side == 'SHORT' and trend_bias == 'LONG'):
                return ExitAction.FULL_CLOSE, amount, "TREND_FLIP_EXIT", None

        # 5. TIME EXIT (Attribute Held)
        if regime != 'TREND' and pos.candles_held >= self.TIME_EXIT_CANDLES:
            if profit_r < 0.5:
                return ExitAction.FULL_CLOSE, amount, "TIME_DECAY_EXIT", None

        # 6. HARD STOP LOSS
        if (side == 'LONG' and current_price <= stop) or (side == 'SHORT' and current_price >= stop):
            return ExitAction.FULL_CLOSE, amount, "STOP_LOSS_HIT", None

        # 7. REGIME-ADAPTIVE PARTIAL EXIT
        tp1_r, tp2_r = 1.0, 2.0
        
        if regime == 'TREND':
            tp1_ratio, tp2_ratio = 0.20, 0.20
        elif regime == 'RANGE':
            tp1_ratio, tp2_ratio = 0.40, 0.40
        else: # LOW_VOL
            tp1_ratio, tp2_ratio = 0.30, 0.30

        # TP 1 Check (1.0R) [IDEMPOTENT]
        if profit_r >= tp1_r and not pos.tp1_done and not getattr(pos, 'tp1_pending', False):
            partial_size = amount * tp1_ratio
            if partial_size >= min_qty:
                return ExitAction.PARTIAL_CLOSE_TP1, partial_size, "TP1_SCALING", entry
            else:
                trades_logger.warning(f"TP1 Size {partial_size} below minQty {min_qty}. Skipping.")

        # TP 2 Check (2.0R) [STRICT ORDERING + IDEMPOTENT]
        # TP2 MUST NOT execute before TP1 is done.
        if profit_r >= tp2_r and pos.tp1_done and not pos.tp2_done and not getattr(pos, 'tp2_pending', False):
            partial_size = amount * tp2_ratio
            if partial_size >= min_qty:
                 return ExitAction.PARTIAL_CLOSE_TP2, partial_size, "TP2_SCALING", None
            else:
                trades_logger.warning(f"TP2 Size {partial_size} below minQty {min_qty}. Skipping.")

        # 8. ADAPTIVE TRAILING STOP
        if profit_r >= 1.2:
            base_mult = 2.0 if regime == 'TREND' else 1.0 if regime == 'RANGE' else 1.5
            if profit_r > 2.0:
                base_mult *= 0.6
            
            trail_dist = current_atr * base_mult
            new_stop = current_price - trail_dist if side == 'LONG' else current_price + trail_dist
            
            # C2: MIN UPDATE DELTA
            move_delta = SafeCore.div(abs(new_stop - stop), current_price)
            is_tighter = (side == 'LONG' and new_stop > stop) or (side == 'SHORT' and new_stop < stop)
            
            if is_tighter and move_delta >= self.MIN_UPDATE_DELTA_PCT:
                return ExitAction.UPDATE_STOP, 0, "TRAILING_UPDATE", new_stop

        return ExitAction.HOLD, 0.0, "NO_ACTION", None
