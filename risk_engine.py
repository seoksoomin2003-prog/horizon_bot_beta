import time
import threading
import json
import os
from typing import List, Dict, Any, Optional
from config import (RISK_PER_TRADE, MAX_PORTFOLIO_RISK, MAX_CONCURRENT_POSITIONS, 
                    COOLDOWN_AFTER_LOSSES, MAX_TRADES_PER_MINUTE, MAX_EQUITY_DROP_PCT, 
                    EQUITY_MONITOR_WINDOW_MIN, MAX_CONSECUTIVE_SYSTEM_FAILURES,
                    MAX_TRADES_PER_HOUR, SYMBOLS_PERFORMANCE_STATE)
from dataclasses import dataclass
from logger import trades_logger, errors_logger
from utils.safe_core import SafeCore
from utils.validator import Validator
from utils.schemas import PositionSchema, AccountState, PerformanceTrade, EquityState
from conviction_engine import ConvictionScore
from utils.safe_ops import safe_float, safe_div

@dataclass
class FirmOrderRequest:
    symbol: str
    side: str
    amount: float
    stop_loss: float
    take_profit: Optional[float]
    conviction_at_entry: float
    intent_id: str = ""
    approved: bool = False
    rejection_reason: str = ""

class RiskEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.consecutive_losses = 0
        self.cooldown_active = False
        
        # Capital Protection State
        self.trade_timestamps = []
        self.equity_history = [] # List of (timestamp, equity)
        self.system_failure_count = 0
        
        # Equity State Tracking (Target D)
        self.last_valid_equity = 1000.0
        self.last_valid_equity_time = time.time()
        self.equity_state = EquityState.VALID

        # Institutional Tracking (v4)
        self.hourly_trades = {} # {symbol: [timestamps]}
        self.performance_history = self._load_performance() # List of PerformanceTrade
        
        # Hybrid Equity Model [Withdrawal-Aware]
        self.total_withdrawals = 0.0
        self.total_deposits = 0.0
        self.adjusted_equity_history = [] # List of (timestamp, adj_equity)
        self.current_drawdown = 0.0
        
        # Dynamic Scaling Factors
        self.regime_mult = 1.0
        self.sharpe_mult = 1.0
        self.drawdown_mult = 1.0
        self.performance_multiplier = 1.0 # Combined

    def approve_order(self, 
                      conviction: ConvictionScore, 
                      current_equity: float, 
                      active_positions: List[PositionSchema],
                      atr: float,
                      last_price: float,
                      stop_loss_price: float,
                      intent_id: str = "",
                      **kwargs) -> FirmOrderRequest:
        """
        [PHASE 3] Validates conviction against institutional risk limits.
        Converts ConvictionScore into a FirmOrderRequest.
        """
        # Call core validation logic
        result = self.validate_trade(
            symbol=conviction.symbol,
            current_equity=current_equity,
            active_positions=active_positions,
            atr=atr,
            confidence=conviction.composite_score, # Use composite conviction
            price=last_price,
            side=conviction.side,
            stop_loss=stop_loss_price,
            micro_score=getattr(conviction, 'micro_score', 1.0),
            **kwargs
        )
        
        if not result['approved']:
            return FirmOrderRequest(
                symbol=conviction.symbol,
                side=conviction.side,
                amount=0.0,
                stop_loss=0.0,
                take_profit=None,
                conviction_at_entry=conviction.composite_score,
                intent_id=intent_id,
                approved=False,
                rejection_reason=result.get('reason', 'Risk Rejection')
            )
            
        return FirmOrderRequest(
            symbol=conviction.symbol,
            side=conviction.side,
            amount=result['position_size'],
            stop_loss=stop_loss_price,
            take_profit=kwargs.get('take_profit'),
            conviction_at_entry=conviction.composite_score,
            intent_id=intent_id,
            approved=True
        )

    def _load_performance(self) -> List[PerformanceTrade]:
        """
        Loads and NORMALIZES persistent performance data.
        """
        import os, json
        from config import SYMBOLS_PERFORMANCE_STATE
        from utils.persistence import get_absolute_path
        abs_state_path = get_absolute_path(SYMBOLS_PERFORMANCE_STATE)
        
        if os.path.exists(abs_state_path):
            try:
                with open(abs_state_path, 'r') as f:
                    raw_data = json.load(f)
                    data = SafeCore.ensure_dict(raw_data)
                    
                    self.total_withdrawals = SafeCore.float(data.get('total_withdrawals', 0.0))
                    self.total_deposits = SafeCore.float(data.get('total_deposits', 0.0))
                    self.adjusted_equity_history = SafeCore.ensure_list(data.get('adj_equity_history', []))
                    
                    history = SafeCore.ensure_list(data.get('history', []))
                    
                    # NORMALIZATION PIPELINE: Legacy -> Typed
                    return [Validator.performance(t) for t in history]
            except Exception as e:
                errors_logger.error(f"Performance Integrity Failure: {e}")
                return []
        return []

    def _save_performance(self):
        """
        Saves institutional performance metrics.
        """
        import os
        from config import SYMBOLS_PERFORMANCE_STATE
        from utils.persistence import get_absolute_path, atomic_write_json
        abs_state_path = get_absolute_path(SYMBOLS_PERFORMANCE_STATE)
        
        try:
            # Convert schemas back to dict for JSON serialization
            history_json = [pd.__dict__ for pd in self.performance_history[-100:]]
            state = {
                'total_withdrawals': self.total_withdrawals,
                'total_deposits': self.total_deposits,
                'adj_equity_history': self.adjusted_equity_history[-500:], # Preserve baseline
                'history': history_json
            }
            atomic_write_json(abs_state_path, state)
        except Exception as e:
            errors_logger.error(f"Failed to save performance: {e}")

    def update_metrics(self, current_equity: float, regime: str = 'RANGE', is_available: bool = True):
        """
        Calculates all institutional metrics (Sharpe, PF, DD) and updates scaling factors.
        Uses Hybrid Withdrawal-Aware Equity Model with stale caching.
        """
        with self.lock:
            now = time.time()
            
            # 5. Stale equity caching with TTL
            if not is_available:
                ttl = 60.0 # 60 seconds TTL
                if now - self.last_valid_equity_time <= ttl:
                    self.equity_state = EquityState.STALE
                    # Use last valid cached equity
                    equity = self.last_valid_equity
                else:
                    # 6. Expired equity: must become UNAVAILABLE
                    self.equity_state = EquityState.UNAVAILABLE
                    errors_logger.warning("Equity state cache expired. Setting state to UNAVAILABLE.")
                    return
            else:
                self.equity_state = EquityState.VALID
                self.last_valid_equity = current_equity
                self.last_valid_equity_time = now
                equity = current_equity
    
            # 3. UNKNOWN / Stale equity must NOT mutate drawdown history.
            # 4. Drawdown calculations require: VALID equity only.
            if self.equity_state != EquityState.VALID:
                return
    
            from config import (REGIME_MULTIPLIERS, MAX_DRAWDOWN_REDUCTION, 
                                MAX_DRAWDOWN_THRESHOLD, SHARPE_LOOKBACK, EQUITY_ROLLING_WINDOW_DAYS)
            
            MIN_EQUITY_FLOOR = 50.0
            safe_equity = max(equity, MIN_EQUITY_FLOOR)
            
            # 1. Hybrid Equity Adjustment
            adj_equity = safe_equity + self.total_withdrawals - self.total_deposits
            self.adjusted_equity_history.append([now, adj_equity])
            
            # Prune history to rolling window (e.g. 10 days)
            cutoff = now - (EQUITY_ROLLING_WINDOW_DAYS * 86400)
            self.adjusted_equity_history = [h for h in self.adjusted_equity_history if h[0] > cutoff]
            if len(self.adjusted_equity_history) > 500:
                self.adjusted_equity_history = self.adjusted_equity_history[-500:]
            
            # Calculate Rolling Peak
            adj_peak = max([h[1] for h in self.adjusted_equity_history]) if self.adjusted_equity_history else adj_equity
            
            if adj_peak > 0:
                self.current_drawdown = (adj_peak - adj_equity) / adj_peak
                
            # 2. Scaling Factor: Drawdown (Phase 2 Softening)
            if self.current_drawdown >= MAX_DRAWDOWN_THRESHOLD:
                self.drawdown_mult = 0.25 # SOFT HALT: Size reduction instead of absolute stop
            elif self.current_drawdown >= MAX_DRAWDOWN_REDUCTION:
                self.drawdown_mult = 0.5 # Reduce by 50%
            else:
                self.drawdown_mult = 1.0
    
            # 3. Scaling Factor: Regime
            self.regime_mult = REGIME_MULTIPLIERS.get(regime, 1.0)
    
            # 4. Scaling Factor: Sharpe [Institutional Clipping]
            sharpe = self._calculate_sharpe(SHARPE_LOOKBACK)
            if sharpe > 2.0: self.sharpe_mult = 1.2
            elif sharpe > 1.0: self.sharpe_mult = 1.0
            elif sharpe > 0.5: self.sharpe_mult = 0.8
            else: self.sharpe_mult = 0.5
            
            # 5. [FIX] Profit Factor Adaptive Scaling (Avoid Death Spiral)
            pf = self.get_profit_factor()
            pf_mult = 1.0
            if pf < 1.3:
                # Neutral scaling to allow recovery
                pf_mult = 1.0 
            elif pf > 2.5:
                pf_mult = 1.2
                
            self.performance_multiplier = self.regime_mult * self.sharpe_mult * self.drawdown_mult * pf_mult

    def _calculate_sharpe(self, lookback: int) -> float:
        """
        Rolling Sharpe = Mean(Returns) / Std(Returns)
        """
        with self.lock:
            if len(self.performance_history) < 5:
                return 1.0 # Neutral starting point
                
            try:
                # Institutional Logic: Use normalized PerformanceTrade objects
                recent = [t.pnl_pct for t in self.performance_history[-lookback:]]
                
                if not recent: return 1.0
                
                import statistics
                mean_ret = statistics.mean(recent)
                std_ret = statistics.stdev(recent) if len(recent) > 1 else 0.001
                return mean_ret / std_ret if std_ret > 0 else 0.0
            except Exception as e:
                errors_logger.error(f"Sharpe calculation failed: {e}")
                return 1.0

    def get_profit_factor(self) -> float:
        """
        PF = Gross Profit / Gross Loss
        """
        with self.lock:
            # Institutional Logic: Use normalized PerformanceTrade objects
            profits = [t.pnl for t in self.performance_history[-50:] if t.pnl > 0]
            losses = [abs(t.pnl) for t in self.performance_history[-50:] if t.pnl < 0]
            
            gross_profit = sum(profits) if profits else 0.1
            gross_loss = sum(losses) if losses else 0.1
            
            return gross_profit / gross_loss

    def validate_trade(self,
                        symbol: str,
                        current_equity: float,
                        active_positions: List[PositionSchema],
                        atr: float,
                        confidence: float = 0.5,
                        **kwargs) -> Dict[str, Any]:
        """
        Validates the trade and returns risk parameters.
        """
        with self.lock:
            from config import (
                RISK_PER_TRADE,
                MAX_PORTFOLIO_RISK,
                MAX_CONCURRENT_POSITIONS,
                MAX_TRADES_PER_HOUR,
                MAX_NOTIONAL_MULTIPLIER,
                MAX_EFFECTIVE_LEVERAGE,
                MIN_STOP_DISTANCE_PCT,
            )
            
            # 0. Margin Safety Invariant [MANDATORY]
            # Strict Fail-Closed Account State Policy:
            # Bypassing synthetic equity floor. If equity is not verified or <= 0, halt execution!
            equity = kwargs.get('equity', current_equity)
            account_verified = kwargs.get('account_verified', True)
            if equity <= 0 or not account_verified or self.equity_state in (EquityState.UNAVAILABLE, EquityState.STALE):
                errors_logger.critical(f"[FAIL-CLOSED] Refusing to validate trade for {symbol} due to balance uncertainty (State: {self.equity_state}).")
                return {'approved': False, 'position_size': 0, 'reason': 'ACCOUNT_STATE_UNAVAILABLE'}
                
            safe_equity = equity
            used_margin = kwargs.get('used_margin', 0)
            margin_ok, margin_msg = self.check_margin_safety(safe_equity, used_margin)
            if not margin_ok:
                return {'approved': False, 'position_size': 0, 'reason': margin_msg}
                
            # Phase 2: Removed drawdown_mult == 0.0 hard block. Drawdown now purely limits size.
                
            if self.cooldown_active:
                return {'approved': False, 'position_size': 0, 'reason': 'Cooldown active after 3 losses'}
    
            if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
                return {'approved': False, 'position_size': 0, 'reason': 'Max concurrent positions reached'}
    
            # 1. Hourly Gating
            now = time.time()
            sym_trades = [t for t in self.hourly_trades.get(symbol, []) if now - t < 3600]
            self.hourly_trades[symbol] = sym_trades
            if len(sym_trades) >= MAX_TRADES_PER_HOUR:
                return {'approved': False, 'position_size': 0, 'reason': f'Hourly limit reached for {symbol}'}
    
            # [PHASE 11] RATE LIMIT SAFETY
            self.trade_timestamps = [t for t in self.trade_timestamps if now - t < 60]
            if len(self.trade_timestamps) > 20:
                errors_logger.critical("[ORDER_SPAM_DETECTED]")
                return {'approved': False, 'position_size': 0, 'reason': 'ORDER_SPAM_DETECTED'}
    
            # 2. Portfolio Risk Calculation
            current_risk_pct = sum([pos.risk_pct for pos in active_positions])
            if current_risk_pct + RISK_PER_TRADE > MAX_PORTFOLIO_RISK:
                return {'approved': False, 'position_size': 0, 'reason': 'Max portfolio risk reached'}
    
            if atr <= 0:
                return {'approved': False, 'position_size': 0, 'reason': 'Invalid ATR <= 0'}
            
            # 3. [SLIPPAGE-AWARE] Position Size Calculation
            from config import RISK_PER_TRADE, STOP_LOSS_ATR_MULT, MAX_SLIPPAGE_PCT, SYMBOL_OVERRIDES
            
            # Override parameters if defined for this symbol
            max_slip_pct = MAX_SLIPPAGE_PCT
            if SYMBOL_OVERRIDES and symbol in SYMBOL_OVERRIDES:
                max_slip_pct = SYMBOL_OVERRIDES[symbol].get("max_slippage_pct", MAX_SLIPPAGE_PCT)
                
            # Adaptive Slippage Buffer
            last_price = kwargs.get('price', 0.0)
            slippage_buffer = max(atr * 0.15, last_price * 0.0015)
            
            # Theoretical Entry adjusted for slippage
            side = kwargs.get('side', 'LONG')
            adjusted_entry = last_price + slippage_buffer if side == 'LONG' else last_price - slippage_buffer
            
            # Hard Slippage Limit
            if last_price > 0:
                slippage_pct = slippage_buffer / last_price
                if slippage_pct > max_slip_pct:
                    return {'approved': False, 'position_size': 0, 'reason': f'Excessive Slippage ({slippage_pct:.2%}) > {max_slip_pct:.2%}'}
    
            risk_amount = safe_equity * RISK_PER_TRADE
            
            # [PHASE 9] CONVICTION -> SOFT WEIGHT
            confidence = max(1.0, float(confidence))
            confidence_multiplier = confidence / 4.0
            
            micro = float(kwargs.get('micro_score', 1.0))
            final_risk_amount = risk_amount * confidence_multiplier * self.performance_multiplier * micro
            
            # distance from adjusted entry to stop (with ATR-compression protection floor)
            stop_price = kwargs.get('stop_loss', 0.0)
            stop_distance_raw = abs(adjusted_entry - stop_price)
            stop_distance_floor = (last_price * float(MIN_STOP_DISTANCE_PCT)) if last_price > 0 else 0.0
            stop_distance = max(stop_distance_raw, stop_distance_floor)
            
            if stop_distance <= 0:
                return {'approved': False, 'position_size': 0, 'reason': 'Zero stop distance'}
                
            position_size = safe_div(final_risk_amount, stop_distance)
    
            # Final sanity checks (mathematical gatekeeper)
            if position_size <= 0:
                return {'approved': False, 'position_size': 0, 'reason': 'Invalid size <= 0'}
            if last_price <= 0:
                return {'approved': False, 'position_size': 0, 'reason': 'Invalid last_price <= 0'}
    
            position_notional = position_size * last_price
            if position_notional <= 0:
                return {'approved': False, 'position_size': 0, 'reason': 'Invalid notional <= 0'}
    
            max_notional = safe_equity * float(MAX_NOTIONAL_MULTIPLIER)
            if position_notional > max_notional:
                return {
                    'approved': False,
                    'position_size': 0,
                    'reason': f'Notional cap breach: {position_notional:.2f} > {max_notional:.2f}'
                }
    
            effective_leverage = safe_div(position_notional, safe_equity)
            if effective_leverage > float(MAX_EFFECTIVE_LEVERAGE):
                return {
                    'approved': False,
                    'position_size': 0,
                    'reason': f'Leverage cap breach: {effective_leverage:.2f}x > {float(MAX_EFFECTIVE_LEVERAGE):.2f}x'
                }
    
            return {
                'approved': True,
                'position_size': position_size,
                'risk_pct': RISK_PER_TRADE * confidence_multiplier * self.performance_multiplier,
                'reason': (
                    f'Approved (StopDistRaw: {stop_distance_raw:.6f}, StopDistUsed: {stop_distance:.6f}, '
                    f'Notional: {position_notional:.2f}, Lev: {effective_leverage:.2f}x)'
                )
            }

    def record_trade_result(self, symbol: str, pnl: float, risk_amount: float, entry_equity: float):
        """
        Update performance history with institutional percentage metrics.
        """
        with self.lock:
            from config import COOLDOWN_AFTER_LOSSES
            
            pnl_pct = pnl / entry_equity if entry_equity > 0 else 0
            r_multiple = pnl / risk_amount if risk_amount > 0 else 0
            
            self.performance_history.append(PerformanceTrade(
                symbol=symbol,
                pnl=pnl, 
                pnl_pct=pnl_pct,
                r_multiple=r_multiple, 
                timestamp=time.time()
            ))
            
            # Heap memory hardening: enforce 100 rolling performance history cap
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
            self._save_performance()
    
            if pnl < 0:
                self.consecutive_losses += 1
                self.hourly_trades.setdefault(symbol, []).append(time.time())
                if self.consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                    self.cooldown_active = True
                    trades_logger.info("[WARN] COOLDOWN ACTIVATED due to 3 consecutive losses.")
            else:
                self.consecutive_losses = 0
                self.cooldown_active = False

    def check_system_health(self, current_equity: float, is_available: bool = True) -> tuple:
        """
        Circuit breaker logic with unverified balance protection.
        """
        with self.lock:
            from config import MAX_TRADES_PER_MINUTE, EQUITY_MONITOR_WINDOW_MIN, MAX_EQUITY_DROP_PCT, MAX_CONSECUTIVE_SYSTEM_FAILURES
            now = time.time()
            
            self.trade_timestamps = [t for t in self.trade_timestamps if now - t < 60]
            if len(self.trade_timestamps) >= MAX_TRADES_PER_MINUTE:
                return False, f"LOSS LOOP DETECTED: {len(self.trade_timestamps)} trades in last 60s"
                
            # 2. Never accept 0.0 as synthetic failure equity
            if is_available and current_equity > 0.0:
                self.equity_history.append((now, current_equity))
                window_sec = EQUITY_MONITOR_WINDOW_MIN * 60
                self.equity_history = [e for e in self.equity_history if now - e[0] < window_sec * 2]
                
                ref_equity = None
                for timestamp, equity in self.equity_history:
                    if now - timestamp >= window_sec:
                        ref_equity = equity
                        break 
                
                if ref_equity is not None:
                    drop_pct = SafeCore.div((ref_equity - current_equity), ref_equity)
                    if drop_pct >= MAX_EQUITY_DROP_PCT:
                        return False, f"FEE BLEED DETECTED: Equity dropped {drop_pct:.2%} in last {EQUITY_MONITOR_WINDOW_MIN}m"
    
            if self.system_failure_count >= MAX_CONSECUTIVE_SYSTEM_FAILURES:
                return False, f"CRITICAL FAILURES: {self.system_failure_count} consecutive execution failures"
    
            return True, ""

    def check_margin_safety(self, current_equity: float, used_margin: float) -> tuple:
        """
        [MANDATORY] Margin Safety Policy (Institutional v5)
        Enforces 200% Margin Coverage invariant (Equity / Used Margin >= 2.0).
        """
        with self.lock:
            if used_margin <= 0:
                return True, "SAFE: No margin used"
    
            # Fixed Definitions
            margin_ratio = SafeCore.div(used_margin, current_equity)
            margin_coverage = SafeCore.div(current_equity, used_margin)
    
            # 1. BLOCK_NEW_TRADES (Ratio > 0.5 or Coverage < 2.0)
            if margin_ratio > 0.5 or margin_coverage < 2.0:
                return False, f"MARGIN BLOCK: Coverage {margin_coverage:.2f} < 2.0 (Used: {margin_ratio:.2%})"
    
            # 2. REDUCE_POSITION_SIZE Warning (Ratio > 0.7)
            if margin_ratio > 0.7:
                errors_logger.warning(f"[RISK] High Margin Usage: {margin_ratio:.2%}. Margin Call risk rising.")
    
            # 3. FORCE_RISK_REDUCTION Alert (Ratio > 0.8)
            if margin_ratio > 0.8:
                errors_logger.critical(f"[CRITICAL] Margin Ratio {margin_ratio:.2%} > 80%. LIQUIDATION RISK IMMINENT.")
            
            return True, ""

    def record_execution_event(self, symbol: str):
        with self.lock:
            self.trade_timestamps.append(time.time())
            self.hourly_trades.setdefault(symbol, []).append(time.time())

    def record_system_failure(self):
        with self.lock:
            self.system_failure_count += 1
            errors_logger.warning(f"System failure recorded. Total: {self.system_failure_count}/{MAX_CONSECUTIVE_SYSTEM_FAILURES}")

    def reset_system_failures(self):
        with self.lock:
            if self.system_failure_count > 0:
                self.system_failure_count = 0
                trades_logger.info("System failure counter reset.")
