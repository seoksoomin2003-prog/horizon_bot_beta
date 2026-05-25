import sys
modules = [
    'config', 'logger', 'utils.schemas', 'utils.safe_core', 'utils.validator', 
    'data_handler', 'signal_engine', 'conviction_engine', 'risk_engine', 
    'execution_engine', 'state_engine', 'utils.exit_engine', 'utils.lifecycle', 'pipeline', 'main'
]
failed = False
for m in modules:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        failed = True

if failed:
    sys.exit(1)
print("ALL CORE IMPORTS SUCCESSFUL.")
