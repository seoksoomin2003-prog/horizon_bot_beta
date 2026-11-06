import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import threading
import ccxt
from typing import Dict, Any

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.journal import ExecutionJournal, JournalEvent
from state_engine import StateEngine
from utils.schemas import PositionSchema, Ticker, TradeSignal, AccountState, ACCOUNT_STATE_UNAVAILABLE
from utils.persistence import get_absolute_path, atomic_write_json
from execution_engine import ExecutionEngine
from data_handler import DataHandler
from utils.telegram_commands import TelegramManager, TelegramMessage, _manager
import config
from utils.runtime_coordinator import RuntimeCoordinator

class TestReliabilityReconstruction(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for isolated test environment
        self.test_dir = tempfile.mkdtemp()
        
        # Override data dir for absolute path authority
        self.orig_log_path = ExecutionJournal.LOG_PATH
        self.orig_snap_path = ExecutionJournal.SNAPSHOT_PATH
        
        ExecutionJournal.LOG_PATH = os.path.join(self.test_dir, "execution_journal.log")
        ExecutionJournal.SNAPSHOT_PATH = os.path.join(self.test_dir, "lifecycle_state.json")
        
        # Backup original stop flag and config states
        self.orig_is_running = RuntimeCoordinator().get_state("is_running", True)
        self.orig_system_state = RuntimeCoordinator().get_state("system_state", "NORMAL")
        self.orig_trading_enabled = RuntimeCoordinator().get_state("trading_enabled", True)
        self.orig_allow_data_polling = RuntimeCoordinator().get_state("allow_data_polling", True)
        
        self.orig_telegram_enabled = config.TELEGRAM_ENABLED
        self.orig_telegram_bot_token = config.TELEGRAM_BOT_TOKEN
        self.orig_telegram_chat_id = config.TELEGRAM_CHAT_ID
        
        RuntimeCoordinator().set_state("is_running", True)
        RuntimeCoordinator().set_state("system_state", "NORMAL")
        RuntimeCoordinator().set_state("trading_enabled", True)
        RuntimeCoordinator().set_state("allow_data_polling", True)

    def tearDown(self):
        # Restore original paths
        ExecutionJournal.LOG_PATH = self.orig_log_path
        ExecutionJournal.SNAPSHOT_PATH = self.orig_snap_path
        
        # Restore config states
        RuntimeCoordinator().set_state("is_running", self.orig_is_running)
        RuntimeCoordinator().set_state("system_state", self.orig_system_state)
        RuntimeCoordinator().set_state("trading_enabled", self.orig_trading_enabled)
        RuntimeCoordinator().set_state("allow_data_polling", self.orig_allow_data_polling)
        
        config.TELEGRAM_ENABLED = self.orig_telegram_enabled
        config.TELEGRAM_BOT_TOKEN = self.orig_telegram_bot_token
        config.TELEGRAM_CHAT_ID = self.orig_telegram_chat_id
        
        # Clean up temporary test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_successful_entry_under_nested_persistence_calls(self):
        print("\n--- [TEST-1] Successful entry under nested persistence calls ---")
        
        state_engine = StateEngine()
        # Mock load initial state
        state_engine.active_positions = []
        
        # Verify lock is an RLock (Reentrant)
        self.assertTrue(isinstance(state_engine.lock, type(threading.RLock())))
        
        # Simulate the pipeline thread acquiring the lock and then nesting calls
        with state_engine.lock:
            # Reentrant locking simulation: persist() acquires state_engine.lock again
            # In standard Locks this causes self-deadlock. With RLock it succeeds immediately.
            state_engine.persist()
            
        print("[PASS] Successful reentrant locking without self-deadlock.")

    def test_02_concurrent_symbol_reconciliation_across_formats(self):
        print("\n--- [TEST-2] Concurrent symbol reconciliation across formats ---")
        
        # Formats to normalize: BTC/USDT, BTC/USDT:USDT, BTCUSDT
        from utils.validator import normalize_symbol
        self.assertEqual(normalize_symbol("BTC/USDT"), "BTCUSDT")
        self.assertEqual(normalize_symbol("BTC/USDT:USDT"), "BTCUSDT")
        self.assertEqual(normalize_symbol("BTCUSDT"), "BTCUSDT")
        self.assertEqual(normalize_symbol("btc/usdt:usdt"), "BTCUSDT")
        
        # Instantiate position schemas
        pos1 = PositionSchema(symbol="BTC/USDT", side="LONG", amount=1.0)
        pos2 = PositionSchema(symbol="BTC/USDT:USDT", side="LONG", amount=1.0)
        pos3 = PositionSchema(symbol="BTCUSDT", side="LONG", amount=1.0)
        
        # Verify dataclass __post_init__ normalized all symbols automatically
        self.assertEqual(pos1.symbol, "BTCUSDT")
        self.assertEqual(pos2.symbol, "BTCUSDT")
        self.assertEqual(pos3.symbol, "BTCUSDT")
        
        print("[PASS] Centralized symbol normalization is successfully enforced on instantiation.")

    def test_03_sl_placement_during_fetch_positions_outage(self):
        print("\n--- [TEST-3] SL placement during fetch_positions outage ---")
        
        class MockExchange:
            def fetch_open_orders(self, *args, **kwargs):
                return []
            def load_markets(self):
                pass
            def amount_to_precision(self, *args, **kwargs):
                return "10"
            def price_to_precision(self, *args, **kwargs):
                return "90"
            def create_order(self, *args, **kwargs):
                return {"id": "12345"}
                
        class MockDataHandler:
            exchange_name = "binance"
            def format_symbol(self, s):
                return s
            def fetch_active_positions(self, *args, **kwargs):
                raise ccxt.NetworkError("Simulated exchange timeout/outage")
            exchange = MockExchange()
            
        dh = MockDataHandler()
        execution = ExecutionEngine(dh, simulation_mode=False)
        execution.state = StateEngine()
        
        # Verify that verify_position_direction returns True during outage
        # so that stop loss placement is allowed to proceed (fail-safe protection)
        self.assertTrue(execution.verify_position_direction("BTCUSDT", "LONG"))
        
        # Call place_stop_loss which depends on verify_position_direction
        order = execution.place_stop_loss("BTCUSDT", "LONG", 10.0, 90.0, force=True)
        self.assertIsNotNone(order)
        self.assertEqual(order["id"], "12345")
        print("[PASS] Outage degradation correctly allows stop-loss placement to proceed.")

    def test_04_emergency_liquidation_during_degraded_exchange_visibility(self):
        print("\n--- [TEST-4] Emergency liquidation during degraded exchange visibility ---")
        
        class MockDataHandler:
            def __init__(self):
                self.exchange_name = 'binance'
                from adapters import get_adapter
                self.adapter = get_adapter(self.exchange_name)
                self.exchange = type('MockExch', (object,), {
                    'fetch_open_orders': lambda s: [],
                    'create_order': lambda *a, **k: {'id': 'mocked_id'},
                    'cancel_order': lambda *a, **k: True,
                    'amount_to_precision': lambda s, a: str(a),
                    'price_to_precision': lambda s, p: str(p),
                    'load_markets': lambda: None,
                })()
                
            def get_position_mode(self, symbol=None):
                return 'one_way'
                
            def format_symbol(self, s):
                return s
            def fetch_active_positions(self, *args, **kwargs):
                raise ccxt.ExchangeNotAvailable("Simulated API degradation")
            
        dh = MockDataHandler()
        execution = ExecutionEngine(dh, simulation_mode=False)
        pos = PositionSchema(symbol="BTCUSDT", side="LONG", amount=1.0)
        
        # verify_position_direction returns True, allowing close_full to run
        ok = execution.close_full("BTCUSDT", pos)
        self.assertTrue(ok)
        print("[PASS] Emergency liquidation successfully proceeds during degraded visibility.")

    def test_05_journal_replay_after_dust_quarantine_restart(self):
        print("\n--- [TEST-5] Journal replay after dust quarantine restart ---")
        
        symbol = "SOLUSDT"
        
        # 1. Log a DUST_QUARANTINED event
        ExecutionJournal.log_event(symbol, JournalEvent.DUST_QUARANTINED)
        
        # 2. Log POSITION_CLOSED event
        ExecutionJournal.log_event(symbol, JournalEvent.POSITION_CLOSED)
        
        # 3. Read snapshot. The SOLUSDT snapshot should preserve the dust_quarantined = True state!
        snap = ExecutionJournal.get_snapshot(symbol)
        self.assertTrue(snap.get("dust_quarantined"))
        self.assertEqual(snap.get("last_event"), JournalEvent.POSITION_CLOSED)
        
        # 4. Wipe snapshots in-memory and replay from append-only log file
        replayed = ExecutionJournal._replay_journal(write_to_disk=False)
        self.assertTrue(replayed[symbol]["dust_quarantined"])
        self.assertEqual(replayed[symbol]["last_event"], JournalEvent.POSITION_CLOSED)
        
        print("[PASS] Dust quarantine state is successfully preserved across position closures and log replays.")

    def test_06_disk_full_persistence_failure_escalation(self):
        print("\n--- [TEST-6] Disk-full persistence failure escalation ---")
        
        # Verify that state_manager's save_positions propagates the write exception and sets is_running = False
        from utils.state_manager import save_positions
        
        # We mock POSITIONS_FILE in state_manager module
        import utils.state_manager
        utils.state_manager.POSITIONS_FILE = os.path.join(self.test_dir, "nonexistent_dir", "positions.json")
        
        # Make the parent path a directory itself so write fails with IsADirectoryError / OSError
        utils_manager_file = utils.state_manager.POSITIONS_FILE
        os.makedirs(utils_manager_file, exist_ok=True)
        
        with self.assertRaises(Exception):
            save_positions([{"symbol": "BTCUSDT"}])
            
        # Verify that bot is halted
        self.assertFalse(RuntimeCoordinator().get_state("is_running"))
        print("[PASS] Disk-full/persistence write failure successfully halts trading and propagates error.")

    def test_07_startup_exchange_outage_recovery(self):
        print("\n--- [TEST-7] Startup exchange outage recovery ---")
        
        # Test that validate_api_keys and _safe_load_markets enter DEGRADED mode
        # rather than looping infinitely or crashing unrecoverably
        
        class OutageExchange:
            def fetch_balance(self):
                raise ccxt.RequestTimeout("Exchange server timeout")
            def load_markets(self):
                raise ccxt.NetworkError("Connection refused")
                
        class MockDataHandler(DataHandler):
            def __init__(self):
                self.exchange_name = "binance"
                self._local = threading.local()
                self._exchange_config = {}
                self.last_equity = 0.0
                self.last_used_margin = 0.0
                
                # Validation and loading calls: must complete and enter DEGRADED state
                self.validate_api_keys()
                self._safe_load_markets()
                
            @property
            def exchange(self):
                return OutageExchange()
                
        # Run startup loading
        dh = MockDataHandler()
        
        # Verify system enters DEGRADED and paused mode instead of hanging or crashing
        self.assertEqual(RuntimeCoordinator().get_state("system_state"), "DEGRADED")
        self.assertFalse(RuntimeCoordinator().get_state("trading_enabled"))
        self.assertFalse(RuntimeCoordinator().get_state("allow_data_polling"))
        print("[PASS] Offline startup completed safely in DEGRADED mode without process lock wedges.")

    def test_08_tp_replay_attribution(self):
        print("\n--- [TEST-8] TP replay attribution under manual close & stop-out ---")
        
        pos = PositionSchema(
            symbol="BTCUSDT", side="LONG", amount=10.0, initial_amount=10.0,
            entry_price=100.0, stop_loss=90.0, atr=5.0, tp1_pending=True, tp1_order_id="tp_order_456"
        )
        
        class MockExchange:
            def fetch_my_trades(self, symbol, limit=20):
                # 3 simulated trade fills
                return [
                    # 1. An unrelated manual exit or stop-out (price at 90.0)
                    {"side": "sell", "price": 90.0, "amount": 10.0, "order": "stop_999", "timestamp": time.time()*1000},
                    # 2. A profitable fill but wrong order ID
                    {"side": "sell", "price": 110.0, "amount": 4.0, "order": "wrong_111", "timestamp": time.time()*1000},
                    # 3. Profitable fill with exact matching order ID (TP1)
                    {"side": "sell", "price": 110.0, "amount": 4.0, "order": "tp_order_456", "timestamp": time.time()*1000}
                ]
                
        class MockDataHandler:
            def format_symbol(self, s):
                return s
            exchange = MockExchange()
            
        dh = MockDataHandler()
        state_engine = StateEngine()
        
        # Run attribution check
        state_engine.confirm_pending_lifecycle_events(dh, pos)
        
        # Verify exact matching order ID was attributed (4.0 filled)
        self.assertEqual(pos.filled_qty, 4.0)
        # Verify wrong order and stop-out prices were excluded
        self.assertNotEqual(pos.filled_qty, 14.0)
        self.assertNotEqual(pos.filled_qty, 18.0)
        print("[PASS] Take-profit attribution is strictly execution-authoritative, excluding unrelated fills.")

    def test_09_daemon_cwd_changes_under_linux(self):
        print("\n--- [TEST-9] Daemon cwd changes under Linux/systemd ---")
        
        # 1. Resolve path
        path_in_orig_cwd = get_absolute_path("test_state.json")
        self.assertTrue(os.path.isabs(path_in_orig_cwd))
        
        # 2. Change current working directory to simulate daemon behavior
        orig_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        
        try:
            path_in_new_cwd = get_absolute_path("test_state.json")
            self.assertEqual(path_in_orig_cwd, path_in_new_cwd)
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        print("[PASS] Centralized path authority successfully prevents directory split-brain across daemon CWD changes.")

    def test_10_telegram_outage_flood_stress(self):
        print("\n--- [TEST-10] Telegram outage flood stress ---")
        
        # Configure config variables to force is_active = True
        config.TELEGRAM_ENABLED = True
        config.TELEGRAM_BOT_TOKEN = "mock_token"
        config.TELEGRAM_CHAT_ID = "mock_chat_id"
        
        manager = TelegramManager()
        
        # Simulate Telegram flood. Bounded PriorityQueue has size 100.
        # Push 119 normal priority (1) messages
        for i in range(119):
            manager.send(f"Normal notification {i}", priority=1)
            
        # Queue should be saturated (100 items max size)
        self.assertTrue(manager.msg_queue.full())
        
        # Queue emergency message: must drop normal-priority message to make room
        manager.send("EMERGENCY: SYSTEM AT PEAK LIMIT", priority=0)
        
        # Check queue has 100 items (not 120) and doesn't crash or block
        self.assertEqual(manager.msg_queue.qsize(), 100)
        
        # Verify emergency message successfully inserted
        items = []
        while not manager.msg_queue.empty():
            items.append(manager.msg_queue.get_nowait())
            
        # Lowest priority (0) is sorted first
        self.assertEqual(items[0].priority, 0)
        self.assertEqual(items[0].text, "EMERGENCY: SYSTEM AT PEAK LIMIT")
        
        print("[PASS] Telegram messaging backlog does not stall system, respects bounded queue and priority shedding.")

if __name__ == "__main__":
    unittest.main()
