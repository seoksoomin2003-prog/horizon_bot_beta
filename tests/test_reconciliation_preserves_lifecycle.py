import tempfile
import unittest

import utils.state_manager as sm
from state_engine import StateEngine
from utils.schemas import PositionSchema


class _DH:
    def __init__(self, exchange_positions):
        self._exchange_positions = exchange_positions
        self.exchange = self

    def fetch_active_positions(self, retries=1):
        return list(self._exchange_positions)

    def format_symbol(self, symbol):
        return symbol

    def fetch_my_trades(self, symbol, limit=20):
        return []


class _Exec:
    def place_stop_loss(self, *args, **kwargs):
        return None

    def cancel_all_orders(self, *args, **kwargs):
        return True


class TestReconciliationLifecyclePreservation(unittest.TestCase):
    def test_reconcile_does_not_reset_tp_flags(self):
        with tempfile.TemporaryDirectory() as d:
            sm.POSITIONS_FILE = f"{d}/positions.json"
            sm.BACKUP_FILE = f"{d}/positions_backup.json"
            
            # Redirect journal to temp dir
            from utils.journal import ExecutionJournal
            ExecutionJournal.LOG_PATH = f"{d}/execution_journal.log"
            ExecutionJournal.SNAPSHOT_PATH = f"{d}/lifecycle_state.json"
            
            # Log the authoritative event first
            ExecutionJournal.log_event("BTCUSDT", "TP1_EXECUTED")

            # Persist a local position with TP1 already done
            sm.save_positions([{
                "symbol": "BTCUSDT",
                "side": "LONG",
                "amount": 10.0,
                "entry_price": 100.0,
                "stop_loss": 90.0,
                "tp1_done": True,
                "tp2_done": False,
                "breakeven_active": False,
                "trailing_active": False,
                "status": "open",
                "lifecycle_state": "TP1_DONE",
            }])

            se = StateEngine()
            self.assertEqual(len(se.active_positions), 1)
            self.assertTrue(se.active_positions[0].tp1_done)

            # Exchange reports updated amount (e.g., after partial close), reconcile should keep tp1_done=True
            exchange_pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=6.0, entry_price=100.0, stop_loss=90.0)
            dh = _DH([exchange_pos])

            ok = se.reconcile_all(dh, _Exec())
            self.assertTrue(ok)
            p = se.active_positions[0]
            self.assertEqual(p.amount, 6.0)
            self.assertTrue(p.tp1_done)


if __name__ == "__main__":
    unittest.main()

