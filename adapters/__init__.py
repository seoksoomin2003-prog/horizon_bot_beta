from .base import AbstractExchangeAdapter
from .binance import BinanceAdapter
from .bitget import BitgetAdapter

def get_adapter(exchange_name: str) -> AbstractExchangeAdapter:
    if exchange_name.lower() == 'binance':
        return BinanceAdapter()
    elif exchange_name.lower() == 'bitget':
        return BitgetAdapter()
    return BinanceAdapter() # default
