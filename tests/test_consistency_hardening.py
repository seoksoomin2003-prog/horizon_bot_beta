import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import threading
from unittest.mock import MagicMock, patch

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.journal import ExecutionJournal
from state_engine import StateEngine
from utils.schemas import PositionSchema, TradeSignal, AccountState, Ticker
from utils.exit_engine import ExitEngine, ExitAction
from utils.lifecycle import LifecycleState, derive_lifecycle_state, apply_tp_fill
from pipeline import Pipeline
from execution_engine import ExecutionEngine
from utils.retry_scheduler import BoundedRetryScheduler
from utils.degraded_manager import DegradedExchangeManager, ExchangeDegradationState

class TestConsistencyHardening(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for safe test environments
        self.test_dir = tempfile.mkdtemp()
        
        # Override paths to temporary test directory
        self.orig_log_path = ExecutionJournal.LOG_PATH
        self.orig_snap_path = ExecutionJournal.SNAPSHOT_PATH
        self.orig_degrade_path = DegradedExchangeManager.STATE_FILE

        ExecutionJournal.LOG_PATH = os.path.join(self.test_dir, "execution_journal.log")
        ExecutionJournal.SNAPSHOT_PATH = os.path.join(self.test_dir, "lifecycle_state.json")
        DegradedExchangeManager.STATE_FILE = os.path.join(self.test_dir, "degradation_state.json")

        # Reset managers
        if hasattr(DegradedExchangeManager, "_instance"):
            DegradedExchangeManager._instance = None
        if hasattr(BoundedRetryScheduler, "_instance"):
            BoundedRetryScheduler._instance = None

        self.degrade_mgr = DegradedExchangeManager()

    def tearDown(self):
        # Restore original paths
        ExecutionJournal.LOG_PATH = self.orig_log_path
        ExecutionJournal.SNAPSHOT_PATH = self.orig_snap_path
        DegradedExchangeManager.STATE_FILE = self.orig_degrade_path
        
        # Clean up temporary test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_tp1_partial_fill_at_1_percent(self):
        print("\n--- [TEST-1] TP1 partial fill at 1% ---")
        pos = PositionSchema(
            symbol="SOLUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open",
            intended_qty=100.0, filled_qty=1.0, remaining_qty=99.0, cumulative_fill_ratio=0.01,
            tp1_pending=True
        )
        state = derive_lifecycle_state(pos)
        self.assertEqual(state, LifecycleState.TP1_PARTIAL_PENDING.value)
        self.assertFalse(pos.tp1_done)
        print("[PASS] Partial fill at 1% kept the state in TP1_PARTIAL_PENDING.")

    def test_02_tp1_partial_fill_at_17_percent(self):
        print("\n--- [TEST-2] TP1 partial fill at 17% ---")
        pos = PositionSchema(
            symbol="SOLUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open",
            intended_qty=100.0, filled_qty=17.0, remaining_qty=83.0, cumulative_fill_ratio=0.17,
            tp1_pending=True
        )
        state = derive_lifecycle_state(pos)
        self.assertEqual(state, LifecycleState.TP1_PARTIAL_PENDING.value)
        self.assertFalse(pos.tp1_done)
        print("[PASS] Partial fill at 17% kept the state in TP1_PARTIAL_PENDING.")

    def test_03_tp1_partial_fill_at_49_percent(self):
        print("\n--- [TEST-3] TP1 partial fill at 49% ---")
        pos = PositionSchema(
            symbol="SOLUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open",
            intended_qty=100.0, filled_qty=49.0, remaining_qty=51.0, cumulative_fill_ratio=0.49,
            tp1_pending=True
        )
        state = derive_lifecycle_state(pos)
        self.assertEqual(state, LifecycleState.TP1_PARTIAL_PENDING.value)
        self.assertFalse(pos.tp1_done)
        print("[PASS] Partial fill at 49% kept the state in TP1_PARTIAL_PENDING.")

    def test_04_restart_during_tp1_partial_fill(self):
        print("\n--- [TEST-4] Restart during TP1 partial fill ---")
        # 1. Initialize StateEngine with local position that has tp1_pending=True
        se = StateEngine()
        pos = PositionSchema(
            symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open",
            intended_qty=100.0, filled_qty=0.0, remaining_qty=100.0, cumulative_fill_ratio=0.0,
            tp1_pending=True
        )
        se.active_positions = [pos]

        # 2. Mock DataHandler trades list showing a partial fill of 17.0
        dh = MagicMock()
        dh.format_symbol.side_effect = lambda s: s
        dh.exchange.fetch_my_trades.return_value = [
            {"amount": 17.0, "price": 100.0, "side": "sell", "timestamp": time.time() * 1000}
        ]

        # 3. Call confirm_pending_lifecycle_events
        se.confirm_pending_lifecycle_events(dh, pos)

        # 4. Verify that recovery did not coerce it to TP1_DONE
        self.assertEqual(pos.filled_qty, 17.0)
        self.assertEqual(pos.intended_qty, 100.0)
        self.assertEqual(pos.cumulative_fill_ratio, 0.17)
        self.assertFalse(pos.tp1_done)
        self.assertTrue(pos.tp1_pending)
        
        state = derive_lifecycle_state(pos)
        self.assertEqual(state, LifecycleState.TP1_PARTIAL_PENDING.value)
        print("[PASS] Restart recovery correctly processed partial fill without binary TP coercion.")

    def test_05_replay_after_fragmented_fills(self):
        print("\n--- [TEST-5] Replay after fragmented fills ---")
        se = StateEngine()
        pos = PositionSchema(
            symbol="ETHUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open",
            intended_qty=100.0, filled_qty=0.0, remaining_qty=100.0, cumulative_fill_ratio=0.0,
            tp1_pending=True
        )
        se.active_positions = [pos]

        # Fragmented trades sum up to 51.0 (51% fill)
        dh = MagicMock()
        dh.format_symbol.side_effect = lambda s: s
        dh.exchange.fetch_my_trades.return_value = [
            {"amount": 10.0, "price": 100.0, "side": "sell", "timestamp": time.time() * 1000},
            {"amount": 25.0, "price": 100.0, "side": "sell", "timestamp": time.time() * 1000},
            {"amount": 16.0, "price": 100.0, "side": "sell", "timestamp": time.time() * 1000}
        ]

        se.confirm_pending_lifecycle_events(dh, pos)

        self.assertEqual(pos.filled_qty, 51.0)
        self.assertEqual(pos.cumulative_fill_ratio, 0.51)
        self.assertFalse(pos.tp1_done)
        
        state = derive_lifecycle_state(pos)
        self.assertEqual(state, LifecycleState.TP1_PARTIAL_PENDING.value)
        print("[PASS] Replay deterministically processed fragmented fills as a partial pending state.")

    def test_06_retry_queue_saturation_during_sl_placement(self):
        print("\n--- [TEST-6] Retry queue saturation during SL placement ---")
        # 1. Initialize BoundedRetryScheduler with 1 slot
        scheduler = BoundedRetryScheduler(max_workers=1, max_queue_size=1)
        
        # 2. Block the scheduler's queue
        scheduler.schedule_retry(lambda: time.sleep(0.5) or True, name="Blocker")
        
        dh = MagicMock()
        dh.format_symbol.side_effect = lambda s: s
        dh.exchange.create_order.side_effect = [
            Exception("CCXT Timeout on first sync attempt"),
            {"id": "sync_sl_order"}
        ]
        
        ee = ExecutionEngine(dh, simulation_mode=False)
        ee.scheduler = scheduler
        ee.state = MagicMock()
        
        pos = PositionSchema(
            symbol="LTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, status="open"
        )
        ee.state.get_position.return_value = pos

        # 3. Trigger stop loss placement that should hit backpressure
        # This will fail scheduling, trigger degraded manager -> CRITICAL_PROTECTION_FAILURE,
        # and attempt synchronous placement which will succeed!
        order = ee.place_stop_loss("LTCUSDT", "LONG", 10.0, 90.0)
        
        # 4. Verify system transitioned to CRITICAL_PROTECTION_FAILURE
        self.assertEqual(self.degrade_mgr.get_state(), ExchangeDegradationState.CRITICAL_PROTECTION_FAILURE)
        # Verify sync fallback placement successfully placed the Stop-Loss
        self.assertEqual(pos.stop_loss, 90.0)
        self.assertEqual(pos.sl_plan_id, "sync_sl_order")
        print("[PASS] Overload safely triggered CRITICAL_PROTECTION_FAILURE, fallback SL placed synchronously.")

    def test_07_exchange_outage_during_retry_overflow(self):
        print("\n--- [TEST-7] Exchange outage during retry overflow ---")
        # 1. Block the scheduler queue
        scheduler = BoundedRetryScheduler(max_workers=1, max_queue_size=1)
        scheduler.schedule_retry(lambda: time.sleep(0.5) or True, name="Blocker")
        
        dh = MagicMock()
        dh.format_symbol.side_effect = lambda s: s
        # Simulates Exchange outage during synchronous fallback
        dh.exchange.create_order.side_effect = Exception("CCXT exchange unavailable")
        
        ee = ExecutionEngine(dh, simulation_mode=False)
        ee.scheduler = scheduler
        ee.state = MagicMock()
        
        pos = PositionSchema(
            symbol="ADAUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=1.0, stop_loss=0.9, atr=0.05, status="open"
        )
        ee.state.get_position.return_value = pos
        ee.close_full = MagicMock(return_value=True)

        # 2. Trigger Stop-Loss which fails async scheduling AND fails fallback sync placement
        ee.place_stop_loss("ADAUSDT", "LONG", 10.0, 0.9)

        # 3. Verify absolute protection invariant: emergency position liquidation was triggered
        ee.close_full.assert_called_once_with("ADAUSDT", pos)
        ee.state.remove_position.assert_called_once_with("ADAUSDT")
        print("[PASS] Fallback sync SL placement failure safely triggered emergency liquidation.")

    def test_08_restart_during_critical_protection_failure(self):
        print("\n--- [TEST-8] Restart during CRITICAL_PROTECTION_FAILURE ---")
        self.degrade_mgr.set_state(ExchangeDegradationState.CRITICAL_PROTECTION_FAILURE, "Naked Exposure Detected")

        # Simulate bot restart by creating a new manager instance
        if hasattr(DegradedExchangeManager, "_instance"):
            DegradedExchangeManager._instance = None
        
        new_mgr = DegradedExchangeManager()
        
        # Verify status is loaded from persistence safely
        self.assertEqual(new_mgr.get_state(), ExchangeDegradationState.CRITICAL_PROTECTION_FAILURE)
        self.assertFalse(new_mgr.is_entry_allowed())
        print("[PASS] Bot successfully booted up in CRITICAL_PROTECTION_FAILURE state, entry rules frozen.")

    def test_09_bitget_delayed_execution_fills(self):
        print("\n--- [TEST-9] Bitget delayed execution fills ---")
        se = StateEngine()
        pos = PositionSchema(
            symbol="DOTUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=10.0, stop_loss=9.0, atr=0.5, status="open",
            intended_qty=10.0, filled_qty=0.0, remaining_qty=10.0, cumulative_fill_ratio=0.0,
            tp1_pending=True
        )
        se.active_positions = [pos]

        dh = MagicMock()
        dh.format_symbol.side_effect = lambda s: s
        
        # 1. Fetch trades initially returns empty (delayed fill on exchange side)
        dh.exchange.fetch_my_trades.return_value = []
        se.confirm_pending_lifecycle_events(dh, pos)
        # Verify no change
        self.assertEqual(pos.filled_qty, 0.0)
        self.assertTrue(pos.tp1_pending)

        # 2. Later fetch_my_trades returns the fill
        dh.exchange.fetch_my_trades.return_value = [
            {"amount": 10.0, "price": 10.0, "side": "sell", "timestamp": time.time() * 1000}
        ]
        se.confirm_pending_lifecycle_events(dh, pos)
        
        # Verify quantity metrics updated and TP1 is finalized as DONE
        self.assertEqual(pos.filled_qty, 10.0)
        self.assertEqual(pos.cumulative_fill_ratio, 1.0)
        self.assertTrue(pos.tp1_done)
        self.assertFalse(pos.tp1_pending)
        print("[PASS] Delayed execution fills reconciled cleanly and updated state dynamically.")

    def test_10_replay_convergence_after_interrupted_persistence(self):
        print("\n--- [TEST-10] Replay convergence after interrupted persistence ---")
        # Log event in Journal (representing executed trade not saved to active_positions due to crash)
        ExecutionJournal.log_event("LINKUSDT", "TP1_TRIGGERED")
        ExecutionJournal.log_event("LINKUSDT", "TP1_PARTIALLY_FILLED")

        # Verify alignment and replay reconstructs state deterministically
        snap = ExecutionJournal.get_snapshot("LINKUSDT")
        self.assertTrue(snap.get("tp1_pending"))
        self.assertFalse(snap.get("tp1_done"))
        print("[PASS] Journal replay successfully converged and alignment remained 100% deterministic.")

if __name__ == '__main__':
    unittest.main()
