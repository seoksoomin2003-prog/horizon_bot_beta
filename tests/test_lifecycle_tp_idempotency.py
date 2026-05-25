import tempfile
import unittest

from utils.exit_engine import ExitEngine, ExitAction
from utils.schemas import PositionSchema
from utils.state_manager import save_positions
from state_engine import StateEngine


class TestTPIdempotency(unittest.TestCase):
    def test_tp1_executes_once_only(self):
        pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, entry_price=100.0, stop_loss=90.0, atr=5.0)
        ex = ExitEngine()

        action, amount, reason, _ = ex.compute_exit(pos, current_price=110.0, current_atr=5.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0)
        self.assertEqual(action, ExitAction.PARTIAL_CLOSE_TP1)
        self.assertGreater(amount, 0)

        pos.tp1_done = True
        action2, _, _, _ = ex.compute_exit(pos, current_price=110.0, current_atr=5.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0)
        self.assertNotEqual(action2, ExitAction.PARTIAL_CLOSE_TP1)

    def test_tp2_executes_once_only_and_after_tp1(self):
        # Use a high ATR baseline to avoid triggering VOL_SPIKE_PROTECTION in ExitEngine.
        pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, entry_price=100.0, stop_loss=90.0, atr=20.0)
        ex = ExitEngine()

        # Profit at 2R but tp1 not done -> MUST NOT emit TP2
        action, _, _, _ = ex.compute_exit(pos, current_price=120.0, current_atr=20.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0)
        self.assertNotEqual(action, ExitAction.PARTIAL_CLOSE_TP2)

        pos.tp1_done = True
        action2, amount2, _, _ = ex.compute_exit(pos, current_price=120.0, current_atr=20.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0)
        self.assertEqual(action2, ExitAction.PARTIAL_CLOSE_TP2)
        self.assertGreater(amount2, 0)

        pos.tp2_done = True
        action3, _, _, _ = ex.compute_exit(pos, current_price=120.0, current_atr=20.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0)
        self.assertNotEqual(action3, ExitAction.PARTIAL_CLOSE_TP2)


if __name__ == "__main__":
    unittest.main()
