"""
conviction_engine.py — Institutional Signal-to-Conviction Transformation (v3)

Transforms raw signals into weighted Conviction scores.
CONVICTION STAGE of the Pipeline.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from utils.schemas import TradeSignal
from utils.safe_core import SafeCore

@dataclass
class ConvictionScore:
    symbol: str
    side: str
    composite_score: float  # 0.0 to 1.0
    signal_score: int       # Original 0-4 score
    confidence: float       # Original 0.0-1.0 confidence
    invariants_passed: bool = True
    obi_score: float = 0.0
    regime_bias: float = 1.0
    micro_score: float = 1.0

class ConvictionEngine:
    def __init__(self):
        # Weighting configuration
        self.W_SIGNAL = 0.5
        self.W_CONFIDENCE = 0.3
        self.W_OBI = 0.2

    def evaluate_signal(self, 
                        signal: TradeSignal, 
                        obi: float = 0.0) -> ConvictionScore:
        """
        Calculates a composite conviction score (0.0 - 1.0).
        Aggregates Technical Signal, Proprietary Confidence, and Orderbook Imbalance.
        """
        # Normalize original signal score (0-4) to 0.0-1.0
        normalized_signal = SafeCore.div(signal.score, 4.0)
        
        # OBI contribution (-1.0 to 1.0). Transform to 0..1 aligned with side.
        obi_bias = obi if signal.trigger == 'LONG' else -obi
        obi_contribution = (obi_bias + 1.0) / 2.0
        
        # Composite calculation
        composite = (normalized_signal * self.W_SIGNAL) + \
                    (signal.confidence * self.W_CONFIDENCE)
        
        # OBI Multiplier (Phase 5: Size influence, not entry block)
        if obi_bias < 0:
            composite *= 0.7
            
        return ConvictionScore(
            symbol=signal.symbol,
            side=signal.trigger or 'NONE',
            composite_score=min(max(composite, 0.0), 1.0),
            signal_score=signal.score,
            confidence=signal.confidence,
            obi_score=obi,
            regime_bias=1.0,
            micro_score=1.0
        )

    def is_convincing(self, conviction: ConvictionScore) -> bool:
        """
        Phase 3: Soft gate, never blocks entry based on conviction score alone.
        Size multiplier handles risk.
        """
        return True
