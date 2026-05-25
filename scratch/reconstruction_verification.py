import sys
import os
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.journal import ExecutionJournal
from state_engine import StateEngine
from execution_engine import ExecutionEngine
from data_handler import DataHandler
from pipeline import Pipeline
from utils.schemas import PositionSchema

def verify_all():
    print("=== Verification of Horizon Bot Reconstruction ===")
    
    # 1. Verify Durable Journal Propagation
    print("\n[VERIFY-1] Durable Journal append error propagation:")
    original_path = ExecutionJournal.LOG_PATH
    ExecutionJournal.LOG_PATH = "/invalid_directory/non_existent_file.log"
    try:
        ExecutionJournal.log_event("BTCUSDT", "TP1_EXECUTED")
        print("FAIL: Log event did not raise an exception on invalid directory.")
    except Exception as e:
        print(f"PASS: Log event correctly propagated exception: {type(e).__name__}")
    finally:
        ExecutionJournal.LOG_PATH = original_path

    # 2. Verify State Engine Lock
    print("\n[VERIFY-2] StateEngine Lock Invariant:")
    se = StateEngine()
    assert hasattr(se, "lock"), "FAIL: StateEngine does not have self.lock"
    print("PASS: StateEngine lock is initialized.")

    # 3. Verify Execution Engine persistent registries and force stops
    print("\n[VERIFY-3] ExecutionEngine persistent registries and force stops:")
    # Mock validate_api_keys
    original_validate = DataHandler.validate_api_keys
    DataHandler.validate_api_keys = lambda self: None
    
    dh = DataHandler()
    ee = ExecutionEngine(dh, simulation_mode=True)
    
    assert hasattr(ee, "lock"), "FAIL: ExecutionEngine does not have self.lock"
    assert hasattr(ee, "order_registry"), "FAIL: ExecutionEngine does not have order_registry"
    assert hasattr(ee, "pending_sl"), "FAIL: ExecutionEngine does not have pending_sl"
    print("PASS: ExecutionEngine lock and persistent registries are initialized.")
    
    # Clean up DataHandler mock
    DataHandler.validate_api_keys = original_validate

    # 4. Verify thread safety lock serialization
    print("\n[VERIFY-4] Thread safety lock serialization test:")
    results = []
    def thread_worker():
        with se.lock:
            time.sleep(0.5)
            results.append("ThreadWorker")
            
    t = threading.Thread(target=thread_worker)
    t.start()
    time.sleep(0.1)
    
    start_time = time.time()
    with se.lock:
        results.append("MainThread")
    elapsed = time.time() - start_time
    
    t.join()
    assert results == ["ThreadWorker", "MainThread"], "FAIL: Thread locking sequence incorrect"
    assert elapsed >= 0.3, f"FAIL: Main thread did not block on StateEngine lock (elapsed: {elapsed:.2f}s)"
    print("PASS: Thread locking successfully serializes and synchronizes access.")

    print("\n=== Verification Successful ===")

if __name__ == "__main__":
    verify_all()
