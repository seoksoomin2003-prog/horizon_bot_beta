import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.validator import Validator
from utils.exceptions import DataIntegrityError
from utils.exit_engine import ExitEngine, ExitAction
from utils.schemas import PositionSchema

def test_short_position_reconciliation():
    print("\n--- [TEST 1] SHORT Position Reconciliation ---")
    
    # 1. CCXT style Short position (unsigned size)
    raw_ccxt_short = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'short',
        'contracts': 1.5,
        'entryPrice': 60000.0,
        'unrealizedPnl': 100.0,
        'leverage': 10
    }
    
    pos1 = Validator.position(raw_ccxt_short)
    assert pos1.side == 'SHORT', f"Expected SHORT, got {pos1.side}"
    print("[PASS] CCXT style explicit side correctly mapped to SHORT.")
    
    # 2. One-Way Info struct fallback
    raw_info_short = {
        'symbol': 'ETH/USDT:USDT',
        'contracts': 5.0,
        'info': {
            'positionSide': 'SHORT'
        }
    }
    pos2 = Validator.position(raw_info_short)
    assert pos2.side == 'SHORT', f"Expected SHORT, got {pos2.side}"
    print("[PASS] Info struct fallback correctly mapped to SHORT.")
    
    # 3. Ambiguous position test (should fail)
    raw_ambiguous = {
        'symbol': 'SOL/USDT:USDT',
        'contracts': 10.0,
    }
    
    try:
        Validator.position(raw_ambiguous)
        print("[FAIL] Ambiguous position should have failed.")
    except DataIntegrityError:
        print("[PASS] Ambiguous position safely rejected.")

def test_zombie_position_recovery():
    print("\n--- [TEST 2 & 5] Zombie Position Recovery / Exit Action ---")
    
    exit_engine = ExitEngine()
    
    # Simulate a reconstructed orphan position (stop_loss = 0.0)
    pos = PositionSchema(
        symbol='BTC/USDT:USDT',
        side='SHORT',
        amount=1.0,
        entry_price=65000.0,
        stop_loss=0.0
    )
    
    action, amount, reason, new_stop = exit_engine.compute_exit(
        pos=pos,
        current_price=64000.0,
        current_atr=500.0,
        trend_bias='SHORT',
        regime='TREND'
    )
    
    assert action == ExitAction.EMERGENCY_RECOVERY, f"Expected EMERGENCY_RECOVERY, got {action}"
    assert reason == "INVALID_STATE"
    print("[PASS] Exit Engine returns EMERGENCY_RECOVERY for zombie positions.")

if __name__ == "__main__":
    test_short_position_reconciliation()
    test_zombie_position_recovery()
    print("\n--- All Unit Tests Passed ---")
