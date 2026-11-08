"""
utils/invariants_v2.py — Institutional Safeguard Invariants (Citadel/Jump Style)

This module enforces strict system-level and market-level invariants 
that must be satisfied before ANY state transition or execution.
"""

import time
from typing import Dict, Any
from logger import errors_logger
from utils.safe_ops import safe_float

class InvariantsV2:
    @staticmethod
    def audit_latency(start_time: float, threshold_ms: float = 500.0) -> bool:
        """
        [L1] Latency Guard: Ensure API response time is within institutional limits.
        """
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > threshold_ms:
            errors_logger.warning(f"[INVARIANT VIOLATION] Latency: {elapsed_ms:.2f}ms > {threshold_ms}ms")
            return False
        return True

    @staticmethod
    def audit_spread(orderbook: Dict[str, Any], max_spread_pct: float = 0.0015) -> bool:
        """
        [M1] Spread Guard: Block entries during periods of toxic liquidity/wide spreads.
        """
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return False
            
        best_bid = safe_float(orderbook['bids'][0][0])
        best_ask = safe_float(orderbook['asks'][0][0])
        
        if best_bid <= 0 or best_ask <= 0: return False
        
        spread = (best_ask - best_bid) / best_bid
        if spread > max_spread_pct:
            errors_logger.warning(f"[INVARIANT VIOLATION] Spread: {spread:.4%} > {max_spread_pct:.4%}")
            return False
        return True

    @staticmethod
    def audit_obi(orderbook: Dict[str, Any], threshold: float = 0.8) -> float:
        """
        [M2] OBI Guard: Orderbook Imbalance.
        Returns a score from -1.0 to 1.0. 
        Higher positive = Buy pressure, Lower negative = Sell pressure.
        """
        if not orderbook: return 0.0
        
        # Look at top 5 levels for volume depth
        bids = orderbook.get('bids', [])[:5]
        asks = orderbook.get('asks', [])[:5]
        
        bid_vol = sum([safe_float(level[1]) for level in bids])
        ask_vol = sum([safe_float(level[1]) for level in asks])
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0: return 0.0
        
        obi = (bid_vol - ask_vol) / total_vol
        return obi
