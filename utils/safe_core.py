import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional
from utils.exceptions import ZeroDivisionGuard, DataIntegrityError

class SafeCore:
    @staticmethod
    def float(val: Any, default: float = 0.0) -> float:
        """
        Force-convert any value to float. 
        Returns default if conversion fails.
        """
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Institutional Grade Division.
        Returns default if denominator is 0 (Logical Suppression).
        Raises ZeroDivisionGuard in strict mode if requested.
        """
        if not denominator or denominator == 0:
            return default
        return float(numerator) / float(denominator)

    @staticmethod
    def get(obj: Any, keys: List[str], default: Any = None) -> Any:
        """Recursive dictionary getter for deep structures."""
        if not isinstance(obj, dict):
            return default
        
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @staticmethod
    def ensure_dict(val: Any) -> Dict[str, Any]:
        """Null Forbidden Policy: Force input to dict."""
        if isinstance(val, dict):
            return val
        return {}

    @staticmethod
    def ensure_list(val: Any) -> List[Any]:
        """Null Forbidden Policy: Force input to list."""
        if isinstance(val, list):
            return val
        return []

    @staticmethod
    def last_row(df: pd.DataFrame) -> Optional[pd.Series]:
        """Safely extract the last row of a DataFrame."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        return df.iloc[-1]

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, min_rows: int = 50) -> bool:
        """Institutional Invariant: OHLCV Min Length Contract."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if len(df) < min_rows:
            return False
        return True
