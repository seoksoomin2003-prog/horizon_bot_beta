import unittest

from execution_engine import ExecutionEngine
from utils.schemas import PositionSchema


class _FakeExchange:
    def __init__(self, *, min_qty=0.0, fill_ratio=1.0):
        self._min_qty = float(min_qty)
        self._fill_ratio = float(fill_ratio)
        self._orders = []
        self._last_order_id = 0

    def market(self, symbol):
        return {"limits": {"amount": {"min": self._min_qty}}}

    def fetch_open_orders(self, symbol):
        return []

    def cancel_order(self, order_id, symbol):
        return True

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self._last_order_id += 1
        oid = f"oid{self._last_order_id}"
        self._orders.append({"id": oid, "symbol": symbol, "type": type, "side": side, "amount": float(amount), "params": params or {}})
        return {"id": oid, "info": {"reduceOnly": bool((params or {}).get("reduceOnly", False))}}

    def fetch_my_trades(self, symbol, limit=5):
        if not self._orders:
            return []
        last = self._orders[-1]
        filled = max(0.0, float(last["amount"]) * self._fill_ratio)
        return [{"order": last["id"], "amount": filled, "side": last["side"], "symbol": symbol}]

    def load_markets(self):
        return True

    def amount_to_precision(self, symbol, amount):
        return str(amount)

    def price_to_precision(self, symbol, price):
        return str(price)


class _FakeDH:
    def __init__(self, exchange):
        self.exchange = exchange
        self.exchange_name = "binance"

    def format_symbol(self, symbol):
        return symbol

    def fetch_active_positions(self, retries=1):
        # This is used by verify_position_direction; return placeholder positions.
        # The side is matched against local state in ExecutionEngine.verify_position_direction.
        return [PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, entry_price=100.0, stop_loss=90.0)]


class TestExecutionPartialClose(unittest.TestCase):
    def test_partial_fill_is_confirmed_and_returned(self):
        ex = _FakeExchange(min_qty=0.0, fill_ratio=0.5)
        dh = _FakeDH(ex)
        eng = ExecutionEngine(dh, simulation_mode=False)

        pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, entry_price=100.0, stop_loss=90.0)
        out = eng.close_partial("BTCUSDT", pos, 4.0, min_qty=0.0)
        self.assertTrue(out["ok"])
        self.assertGreater(out["filled_amount"], 0.0)
        self.assertAlmostEqual(out["filled_amount"], 2.0)

    def test_dust_position_cleanup_full_close(self):
        ex = _FakeExchange(min_qty=1.0, fill_ratio=1.0)
        dh = _FakeDH(ex)
        eng = ExecutionEngine(dh, simulation_mode=False)

        pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=0.5, entry_price=100.0, stop_loss=90.0)
        out = eng.close_partial("BTCUSDT", pos, 0.2, min_qty=1.0)
        # Remaining is below min_qty => should full close (dust cleanup path)
        self.assertTrue(out["dust_closed"])


if __name__ == "__main__":
    unittest.main()

