from typing import Any
from utils.safe_core import SafeCore

def safe_float(val: Any, default: float = 0.0) -> float:
    """Institutional-grade float conversion."""
    return SafeCore.float(val, default)

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Institutional-grade zero-division protection."""
    return SafeCore.div(numerator, denominator, default)
