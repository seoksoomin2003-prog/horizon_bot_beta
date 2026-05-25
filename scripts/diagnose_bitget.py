import os
import sys
import json
import ccxt

# --- ABSOLUTE EARLY-BOOT ENCODING ENFORCEMENT (WINDOWS RELIABILITY) ---
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def safe_print(msg):
    """
    Standard print wrapper for Windows reliability.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback to pure ASCII if cp1252 fails
        print(str(msg).encode("ascii", "ignore").decode())

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE
from logger import errors_logger

def diagnose():
    safe_print("[INIT] Starting Bitget API Forensic Diagnosis...")
    
    if not BITGET_API_KEY or not BITGET_API_SECRET or not BITGET_PASSPHRASE:
        safe_print("[ERROR] Missing Bitget credentials in .env file.")
        return

    # Initialize Bitget (Futures/Swap)
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY,
        'secret': BITGET_API_SECRET,
        'password': BITGET_PASSPHRASE,
        'options': {'defaultType': 'swap'}
    })

    try:
        # 1. Fetch Balance (Basic Connectivity)
        safe_print("[INFO] Step 1: Fetching Balance...")
        balance = exchange.fetch_balance()
        safe_print("[OK] Balance Fetched successfully.")
        
        # 2. Inspect Permissions (Metadata)
        safe_print("[INFO] Step 2: Inspecting Account Metadata...")
        info = balance.get('info', {})
        safe_print(f"[DATA] Raw Info Keys: {list(info.keys())}")
        
        # 3. Check for permissions explicitly
        try:
            safe_print("[INFO] Step 3: Checking API Permission metadata...")
            safe_print(f"[DATA] Account Type/Info from Bitget: {json.dumps(info, indent=2)}")
        except Exception:
            safe_print("[WARN] Could not fetch explicit permission list. Checking account state instead.")

        # 4. Final Verdict
        if 'USDT' not in balance:
            safe_print("[WARN] USDT wallet not found in 'swap' account. Is Futures enabled on the web UI?")
        
        safe_print("\n[OK] --- DIAGNOSIS COMPLETE ---")
        safe_print("If you see 'Incorrect permissions' (40014) in actual trades, check:")
        safe_print("1. [Web UI] API Key Settings -> Permission: 'Futures Trading' must be CHECKED.")
        safe_print("2. [Web UI] API Key Settings -> Type: Must be 'Write' (not Read-only).")
        safe_print("3. [Account] Ensure Futures Account was actually opened and funded.")

    except Exception as e:
        safe_print(f"[FATAL] ERROR during diagnosis: {e}")
        if "40014" in str(e):
            safe_print("\n[CRITICAL] CONFIRMED ERROR 40014: Missing 'Future Order Write' Permissions.")
            safe_print("SOLUTION: Go to Bitget API Management, find this key, and ensure 'Trade' (Futures) and 'Write' permissions are enabled.")
        elif "40001" in str(e):
            safe_print("[ERROR] Invalid API Key or Secret.")

if __name__ == "__main__":
    diagnose()
