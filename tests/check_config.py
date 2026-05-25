import sys
import os

# Add parent directory to sys.path to allow importing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from utils.config_validator import validate_env, validate_config_values, get_config_summary

def run_diagnostic():
    print("--- [HORIZON BOT] CONFIGURATION DIAGNOSTIC ---")
    
    # 1. Test Environment
    print("\n[PHASE 1] Validating Environment...")
    try:
        telegram_enabled = validate_env()
        print(f"SUCCESS: Environment variables validated.")
        print(f"STATUS: Telegram is {'ENABLED' if telegram_enabled else 'DISABLED'}")
    except Exception as e:
        print(f"FAILED: {e}")

    # 2. Test Class Constants
    print("\n[PHASE 2] Validating Configuration Constants...")
    try:
        validate_config_values(config)
        print("SUCCESS: Essential constants (Stop Loss, RR, Cooldown) are correctly defined.")
    except Exception as e:
        print(f"FAILED: {e}")

    # 3. Print Summary
    print("\n[PHASE 3] Current Config Summary (Redacted):")
    summary = get_config_summary(config)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n--- DIAGNOSTIC COMPLETE ---")

if __name__ == "__main__":
    run_diagnostic()
