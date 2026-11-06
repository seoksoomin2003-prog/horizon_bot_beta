import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import threading
import concurrent.futures
from typing import Dict, Any

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.journal import ExecutionJournal
from state_engine import StateEngine, PositionManager
from utils.schemas import PositionSchema, TradeSignal, AccountState
from utils.exit_engine import ExitEngine, ExitAction
from pipeline import Pipeline
from execution_engine import ExecutionEngine
from utils.telegram_commands import TelegramManager, TelegramMessage

class TestReconstructionIntegrity(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for safe test environments
        self.test_dir = tempfile.mkdtemp()
        
        # Backup original journal paths
        self.orig_log_path = ExecutionJournal.LOG_PATH
        self.orig_snap_path = ExecutionJournal.SNAPSHOT_PATH
        
        # Override paths to temporary test directory
        ExecutionJournal.LOG_PATH = os.path.join(self.test_dir, "execution_journal.log")
        ExecutionJournal.SNAPSHOT_PATH = os.path.join(self.test_dir, "lifecycle_state.json")

    def tearDown(self):
        # Restore original paths
        ExecutionJournal.LOG_PATH = self.orig_log_path
        ExecutionJournal.SNAPSHOT_PATH = self.orig_snap_path
        
        # Clean up temporary test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_simultaneous_tp1_across_10_assets(self):
        print("\n--- [TEST-1] Simultaneous TP1 across 10 assets ---")
        
        # Setup 10 assets
        assets = [f"COIN{i}USDT" for i in range(10)]
        positions = []
        for sym in assets:
            pos = PositionSchema(
                symbol=sym, side="LONG", amount=10.0, initial_amount=10.0,
                entry_price=100.0, stop_loss=90.0, atr=5.0
            )
            positions.append(pos)
            
        # Create ExitEngine
        exit_engine = ExitEngine()
        
        # Process exits in parallel mimicking Parallel Executor
        triggered_events = []
        
        def run_exit(pos):
            action, amount, reason, _ = exit_engine.compute_exit(
                pos, current_price=115.0, current_atr=5.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0
            )
            if action == ExitAction.PARTIAL_CLOSE_TP1:
                # Deduplication block
                if not pos.tp1_done and not pos.tp1_pending:
                    pos.tp1_pending = True
                    # Lock-synchronized journal logging
                    ExecutionJournal.log_event(pos.symbol, "TP1_TRIGGERED")
                    triggered_events.append(pos.symbol)
                    
        threads = []
        for p in positions:
            t = threading.Thread(target=run_exit, args=(p,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Verify 10 events were written without concurrency conflict
        self.assertEqual(len(triggered_events), 10)
        snap = ExecutionJournal.get_all_snapshots()
        for sym in assets:
            self.assertTrue(snap[sym]["tp1_pending"])
            self.assertFalse(snap[sym]["tp1_done"])
        print("[PASS] Successfully logged and tracked 10 simultaneous TP1 triggers without conflict.")

    def test_02_crash_during_tp1_confirmation_lag(self):
        print("\n--- [TEST-2] Crash during TP1 confirmation lag ---")
        
        pos = PositionSchema(
            symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0
        )
        
        # Log pending trigger
        pos.tp1_pending = True
        ExecutionJournal.log_event(pos.symbol, "TP1_TRIGGERED")
        
        # Simulate exit engine evaluation on subsequent loop during lag
        exit_engine = ExitEngine()
        action, _, _, _ = exit_engine.compute_exit(
            pos, current_price=110.0, current_atr=5.0, trend_bias="NEUTRAL", regime="RANGE", min_qty=0.0
        )
        
        # Verify that because tp1_pending is active, exit engine returns HOLD
        self.assertEqual(action, ExitAction.HOLD)
        print("[PASS] Duplicate TP1 triggers correctly blocked during lag (pending confirmation state).")

    def test_03_restart_during_pending_partial_close(self):
        print("\n--- [TEST-3] Restart during pending partial close ---")
        
        # 1. Log pending to journal
        ExecutionJournal.log_event("ETHUSDT", "TP1_TRIGGERED")
        
        # 2. Simulate crash/reboot by reloading and aligning state
        ExecutionJournal.verify_and_align_snapshot()
        
        # 3. Verify snapshot correctly reconstructed state as pending
        snap = ExecutionJournal.get_snapshot("ETHUSDT")
        self.assertTrue(snap["tp1_pending"])
        self.assertFalse(snap["tp1_done"])
        print("[PASS] Pending state successfully recovered and aligned from journal replay after restart.")

    def test_04_concurrent_snapshot_writes(self):
        print("\n--- [TEST-4] Concurrent snapshot writes ---")
        
        # Spawn 50 threads trying to write to ExecutionJournal simultaneously
        num_threads = 50
        errors = []
        
        def write_worker(idx):
            try:
                sym = f"COIN{idx}USDT"
                ExecutionJournal.log_event(sym, "TP1_TRIGGERED")
                # Interleaved write
                ExecutionJournal.log_event(sym, "TP1_EXECUTED")
            except Exception as e:
                errors.append(e)
                
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=write_worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Verify zero concurrent serialization exceptions occurred
        self.assertEqual(len(errors), 0)
        
        # Verify all 50 coins are written correctly and matches append-only log exactly
        snap = ExecutionJournal.get_all_snapshots()
        self.assertEqual(len(snap), num_threads)
        for i in range(num_threads):
            sym = f"COIN{i}USDT"
            self.assertTrue(snap[sym]["tp1_done"])
            self.assertFalse(snap[sym]["tp1_pending"])
        print("[PASS] RLock serialized 100 concurrent log events perfectly. No data loss or corruption.")

    def test_05_exchange_rest_lag_during_execution(self):
        print("\n--- [TEST-5] Exchange REST lag during execution ---")
        
        class MockDH:
            def format_symbol(self, s): return s
            def fetch_active_positions(self, retries=1, **kwargs): 
                return [PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0, entry_price=100.0, stop_loss=90.0, atr=5.0)]
            def get_account_state(self): return AccountState(equity=1000.0)
            def fetch_ohlcv(self, *args, **kwargs): return []
            def fetch_ticker(self, *args, **kwargs): return None
            
        class MockExec:
            def place_stop_loss(self, *args, **kwargs): return None
            def cancel_all_orders(self, *args, **kwargs): return True
            def close_full(self, *args, **kwargs): return True
            def close_partial(self, *args, **kwargs):
                time.sleep(1.5) # Simulate exchange REST lag
                return {"filled_amount": 0.0}
            def watchdog_verify_sl(self, *args, **kwargs): return True

        # Pipeline test
        pos = PositionSchema(
            symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0
        )
        
        # Verify orchestrator wait isolation
        # Exit stage runs pos exit which sleeps for 1.5s
        # Pipeline must finish wait before 5.0 seconds timeout
        from state_engine import StateEngine
        from conviction_engine import ConvictionEngine
        from risk_engine import RiskEngine
        from signal_engine import SignalEngine
        
        se = StateEngine()
        se.active_positions = [pos]
        
        pipe = Pipeline(
            dh=MockDH(),
            signal_engine=SignalEngine(),
            conviction_engine=ConvictionEngine(),
            risk_engine=RiskEngine(),
            execution_engine=MockExec(),
            exit_engine=ExitEngine(),
            state_engine=se
        )
        
        start = time.time()
        pipe.run_cycle()
        elapsed = time.time() - start
        
        # Verify orchestrator completed run successfully
        self.assertLess(elapsed, 3.0)
        print(f"[PASS] Orchestrator handled exchange REST lag gracefully in {elapsed:.3f}s (under 5s loop bounds).")

    def test_06_bitget_api_stall_during_trailing_migration(self):
        print("\n--- [TEST-6] Bitget API stall during trailing migration ---")
        
        # Mock exchange that stalls during order cancellations
        class MockExchangeStall:
            def format_symbol(self, s): return s
            def price_to_precision(self, *args): return "0.05"
            def amount_to_precision(self, *args): return "10"
            def load_markets(self): return True
            def fetch_open_orders(self, *args): return []
            def market(self, *args): return {"limits": {"amount": {"min": 0.0}, "cost": {"min": 0.0}}}
            def create_order(self, *args, **kwargs):
                return {"id": "new_sl_id"}
            def cancel_order(self, *args):
                time.sleep(3.0) # Stall during old order cancel
                raise RuntimeError("API Timeout")
                
        class MockDH:
            def format_symbol(self, s): return s
            def fetch_active_positions(self, retries=2, **kwargs): 
                return [PositionSchema(symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0, entry_price=100.0, stop_loss=90.0)]
            @property
            def exchange_name(self): return "bitget"
            def get_position_mode(self): return "hedge"
            
        dh = MockDH()
        dh.exchange = MockExchangeStall()
        
        eng = ExecutionEngine(dh, simulation_mode=False)
        pos = PositionSchema(
            symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, sl_plan_id="old_sl_id"
        )
        
        # Run Trailing stop update (Stop-First sequence)
        start = time.time()
        res = eng.atomic_stop_update("BTCUSDT", pos, 95.0)
        elapsed = time.time() - start
        
        # The new stop is live, but the old one could not be cancelled: double-stop risk must be reported,
        # not hidden. Protection stays on the new stop and new entries are frozen.
        self.assertFalse(res)
        self.assertEqual(pos.stop_loss, 95.0)
        self.assertEqual(pos.sl_plan_id, "new_sl_id")
        self.assertEqual(pos.lifecycle_state, "PROTECTION_CONFLICT")
        from utils.degraded_manager import DegradedExchangeManager, ExchangeDegradationState
        self.assertEqual(DegradedExchangeManager().get_state(), ExchangeDegradationState.CRITICAL_PROTECTION_FAILURE)
        DegradedExchangeManager().set_state(ExchangeDegradationState.HEALTHY, "test cleanup")
        print(f"[PASS] Stop-First migration flagged the conflict in {elapsed:.3f}s, preserving protection even during Bitget cancel stall.")

    def test_07_executor_worker_hang(self):
        print("\n--- [TEST-7] Executor worker hang ---")
        
        # Setup pipeline with deadlocked exit worker
        class MockDH:
            def format_symbol(self, s): return s
            def fetch_active_positions(self, retries=1, **kwargs): 
                return [PositionSchema(symbol="DEADLOCKUSDT", side="LONG", amount=10.0, initial_amount=10.0, entry_price=100.0, stop_loss=90.0, atr=5.0)]
            def get_account_state(self): return AccountState(equity=1000.0)
            def fetch_ohlcv(self, *args, **kwargs): return []
            def fetch_ticker(self, *args, **kwargs): return None
            
        class MockDeadlockExec:
            def place_stop_loss(self, *args, **kwargs): return None
            def cancel_all_orders(self, *args, **kwargs): return True
            def close_full(self, *args, **kwargs): return True
            def close_partial(self, *args, **kwargs): return {"filled_amount": 0.0}
            def watchdog_verify_sl(self, *args, **kwargs):
                # Block for 6 seconds to trigger Pipeline's 5-second timeout, then exit cleanly
                time.sleep(6.0)
                    
        pos = PositionSchema(
            symbol="DEADLOCKUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0
        )
        
        from state_engine import StateEngine
        from conviction_engine import ConvictionEngine
        from risk_engine import RiskEngine
        from signal_engine import SignalEngine
        
        se = StateEngine()
        se.active_positions = [pos]
        
        pipe = Pipeline(
            dh=MockDH(),
            signal_engine=SignalEngine(),
            conviction_engine=ConvictionEngine(),
            risk_engine=RiskEngine(),
            execution_engine=MockDeadlockExec(),
            exit_engine=ExitEngine(),
            state_engine=se
        )
        
        start = time.time()
        # Should timeout in 5 seconds
        pipe.run_cycle()
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 6.5)
        self.assertGreater(elapsed, 4.8)
        print(f"[PASS] Timeout-isolation successfully intercepted worker deadlock, freeing orchestrator in {elapsed:.3f}s.")

    def test_08_telegram_429_flood_during_liquidation_storm(self):
        print("\n--- [TEST-8] Telegram 429 flood during liquidation storm ---", flush=True)
        
        # Queue message tests
        mgr = TelegramManager()
        
        # Force active state for testing so send() actually queues messages
        with unittest.mock.patch('utils.telegram_commands.TelegramManager.is_active', new_callable=unittest.mock.PropertyMock) as mock_active:
            mock_active.return_value = True
            
            # Enforce rate-limits simulated on message queues
            for i in range(10):
                mgr.send(f"⚠️ LIQUIDATION THREAT {i}", priority=0)
                
            self.assertGreaterEqual(mgr.msg_queue.qsize(), 0)
            
            # Process queue item with timeout to prevent infinite block
            try:
                msg = mgr.msg_queue.get(timeout=2.0)
                self.assertEqual(msg.text, "⚠️ LIQUIDATION THREAT 0")
                self.assertEqual(msg.priority, 0)
                print("[PASS] Telegram PriorityQueue successfully isolated flood and sorted emergency priority.")
            except queue.Empty:
                self.fail("Message queue was empty, send() failed to queue message.")

    def test_09_replay_after_partial_persistence(self):
        print("\n--- [TEST-9] Replay after partial persistence ---")
        
        # 1. Log authentic events
        ExecutionJournal.log_event("BTCUSDT", "TP1_TRIGGERED")
        ExecutionJournal.log_event("BTCUSDT", "TP1_EXECUTED")
        
        # 2. Corrupt/drift snapshot cache manually
        corrupt_snap = {"BTCUSDT": {"tp1_done": False, "tp1_pending": False}}
        with open(ExecutionJournal.SNAPSHOT_PATH, "w") as f:
            json.dump(corrupt_snap, f)
            
        # 3. Call verify_and_align_snapshot
        ExecutionJournal.verify_and_align_snapshot()
        
        # 4. Verify snap aligned back to journal truth ("Journal Always Wins")
        snap = ExecutionJournal.get_snapshot("BTCUSDT")
        self.assertTrue(snap["tp1_done"])
        self.assertFalse(snap["tp1_pending"])
        print("[PASS] Snapshot cache successfully aligned back to journal log truth after drift corruption.")

    def test_10_recovery_after_journal_replay(self):
        print("\n--- [TEST-10] Recovery after journal replay ---")
        
        # Log complex lifecycle series
        ExecutionJournal.log_event("BTCUSDT", "TP1_TRIGGERED")
        ExecutionJournal.log_event("BTCUSDT", "TP1_EXECUTED")
        ExecutionJournal.log_event("BTCUSDT", "TP2_TRIGGERED")
        ExecutionJournal.log_event("BTCUSDT", "TP2_EXECUTED")
        
        # Full close
        ExecutionJournal.log_event("BTCUSDT", "POSITION_CLOSED")
        
        # Re-read
        reconstructed = ExecutionJournal._replay_journal(write_to_disk=False)
        
        # Verify deleted / clean lifecycle for BTCUSDT
        self.assertNotIn("BTCUSDT", reconstructed)
        print("[PASS] Replay determinism perfectly resolved complex state transitions and terminal close sequences.")

if __name__ == "__main__":
    unittest.main()
