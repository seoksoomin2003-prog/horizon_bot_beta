from typing import Dict, Any, Optional
import uuid
from adapters.base import AbstractExchangeAdapter

class BitgetAdapter(AbstractExchangeAdapter):
    def format_order_params(self, symbol: str, side: str, amount: float, price: float, client_order_id: str, reduce_only: bool = False, order_type: str = "limit", params: Dict[str, Any] = None) -> Dict[str, Any]:
        p = params or {}
        if client_order_id:
            p['clientOid'] = client_order_id
        else:
            p['clientOid'] = f"hb_{uuid.uuid4().hex[:12]}"
        # Bitget handles reduceOnly via posSide, but if someone passes it natively we strip it to prevent API errors
        if 'reduceOnly' in p:
            del p['reduceOnly']
        return p

    def parse_plan_order(self, response: Dict[str, Any]) -> str:
        # Bitget nested data extraction
        if "info" in response and "data" in response["info"]:
            data = response["info"]["data"]
            if "orderId" in data:
                return data["orderId"]
        return response.get('id', str(response))

    def get_position_mode(self, client: Any) -> str:
        try:
            res = client.private_mix_get_v2_mix_account_account({"productType": "USDT-FUMARGIN"})
            if res and isinstance(res, dict) and "data" in res:
                data = res["data"]
                if isinstance(data, list) and len(data) > 0:
                    hm = data[0].get("holdMode")
                    if hm == "double_hold":
                        return 'hedge'
                    elif hm == "single_hold":
                        return 'oneway'
        except Exception:
            pass
        return 'oneway'

    def format_trade_side(self, params: Dict[str, Any], side: str, position_mode: str, is_close: bool = False) -> Dict[str, Any]:
        if position_mode == 'hedge':
            normalized = side.lower()
            if normalized in ['buy', 'long']:
                params['posSide'] = 'long'
            elif normalized in ['sell', 'short']:
                params['posSide'] = 'short'
            else:
                params['posSide'] = normalized
                
            params['tradeSide'] = 'close' if is_close else 'open'
        elif is_close:
            params['reduceOnly'] = True
        return params

    def format_symbol(self, symbol: str) -> str:
        if ':' not in symbol:
            return f"{symbol}:USDT"
        return symbol

    def enforce_position_mode(self, client: Any, position_mode: str, symbols: list, format_symbol_fn) -> None:
        is_hedged = True if position_mode == "hedge" else False
        mode_text = position_mode.upper()

        from logger import trades_logger, errors_logger
        trades_logger.info(f"Enforcing {mode_text} position mode on Bitget...")
        for symbol in symbols:
            formatted_symbol = format_symbol_fn(symbol)
            try:
                client.set_position_mode(is_hedged, formatted_symbol)
                trades_logger.info(f"Position mode confirmed {mode_text} for {symbol}")
            except Exception as e:
                error_str = str(e).lower()
                if "already" in error_str or "no change" in error_str or "30030" in error_str:
                    trades_logger.info(f"{symbol} already in {mode_text} mode.")
                else:
                    errors_logger.warning(f"Could not enforce {mode_text} mode for {symbol}: {e}")

    def generate_client_order_id(self) -> Optional[str]:
        import uuid
        return str(uuid.uuid4()).replace('-', '')[:30]
