from utils.schemas import TradeSignal
import uuid
import sys

print("Python Version:", sys.version)

signal = TradeSignal(symbol="BTCUSDT", trigger="LONG", confidence=0.8)
print("Signal created:", signal)

try:
    signal.intent_id = uuid.uuid4().hex
    print("MUTATION SUCCESS - WARNING: Immutability broken.")
except Exception as e:
    print(f"MUTATION FAILED: {type(e).__name__} - {e}")
