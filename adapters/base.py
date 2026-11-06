from typing import Dict, Any, Optional

class AbstractExchangeAdapter:
    def format_order_params(self, symbol: str, side: str, amount: float, price: float, client_order_id: str, reduce_only: bool = False, order_type: str = "limit", params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Formats base parameters for exchange requests."""
        raise NotImplementedError

    def parse_plan_order(self, response: Dict[str, Any]) -> str:
        """Extracts the order ID from a plan/stop order response."""
        raise NotImplementedError

    def get_position_mode(self, client: Any) -> str:
        """Determines if the exchange is in hedge or one-way mode."""
        raise NotImplementedError

    def format_trade_side(self, params: Dict[str, Any], side: str, position_mode: str, is_close: bool = False) -> Dict[str, Any]:
        """Mutates params to include exchange-specific trade side semantics."""
        raise NotImplementedError

    def format_symbol(self, symbol: str) -> str:
        """Formats base symbol structure for exchange requests."""
        return symbol

    def enforce_position_mode(self, client: Any, position_mode: str, symbols: list, format_symbol_fn) -> None:
        """Enforces the configured position mode on the exchange."""
        pass

    def generate_client_order_id(self) -> Optional[str]:
        """Generates an exchange-specific, safe client order ID if needed."""
        return None

    def patch_exchange(self, client: Any) -> None:
        """Applies exchange-specific monkey patches to CCXT instances."""
        pass

