import sys
import os
import time

# Ensure imports work from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handler import DataHandler
from signal_engine import SignalEngine
from conviction_engine import ConvictionEngine
from risk_engine import RiskEngine
from execution_engine import ExecutionEngine
from utils.exit_engine import ExitEngine
from state_engine import StateEngine
from pipeline import Pipeline
import config

def run_simulation():
    print("Initializing components for Simulation...")
    # Force simulation mode
    config.DEMO_MODE = True
    
    try:
        # Mock API validation for simulation
        original_validate = DataHandler.validate_api_keys
        DataHandler.validate_api_keys = lambda self: print("[MOCK] Bypassing API Validation for Simulation.")
        dh = DataHandler()
        DataHandler.validate_api_keys = original_validate
        signal = SignalEngine()
        conviction = ConvictionEngine()
        risk = RiskEngine()
        execution = ExecutionEngine(dh, simulation_mode=True)
        exit_e = ExitEngine()
        state = StateEngine()

        bot_pipeline = Pipeline(
            dh=dh,
            signal_engine=signal,
            conviction_engine=conviction,
            risk_engine=risk,
            execution_engine=execution,
            exit_engine=exit_e,
            state_engine=state
        )

        print(f"Running simulation on symbols: {config.SYMBOLS}")
        
        # Run 3 cycles
        for i in range(3):
            print(f"--- Cycle {i+1} ---")
            bot_pipeline.run_cycle()
            time.sleep(1)
            
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        print(f"[1] Signals Generated: {bot_pipeline.signal_generated_count}")
        print(f"[2] Accepted Trades  : {bot_pipeline.execution_attempt_count}")
        print(f"[3] Rejections       : ")
        print(f"    - By Filter/Invariants: {bot_pipeline.signal_rejected_count}")
        print(f"    - By Risk Engine      : {bot_pipeline.risk_rejection_count}")
        print("="*50)
        
    except Exception as e:
        print(f"Simulation failed: {e}")

if __name__ == "__main__":
    run_simulation()
