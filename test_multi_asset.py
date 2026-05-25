import config
from data_handler import DataHandler
from execution_engine import ExecutionEngine
from risk_engine import RiskEngine
from utils.schemas import PositionSchema
from conviction_engine import ConvictionScore

def simulate_tests():
    print("=== Multi-Asset Expansion Validation ===")
    
    # Init
    # Mock validate_api_keys for test environment without real keys
    original_validate = DataHandler.validate_api_keys
    DataHandler.validate_api_keys = lambda self: None
    
    dh = DataHandler()
    exec_engine = ExecutionEngine(dh, simulation_mode=True)
    risk = RiskEngine()
    
    DataHandler.validate_api_keys = original_validate
    
    print("\n[TEST-1] DOGE Breakout Entry (Override Check)")
    doge_slip = risk.validate_trade("DOGEUSDT", 1000, [], 0.05, 0.9, price=0.150, side="LONG", stop_loss=0.140)
    print(f"DOGE Validated: {doge_slip['approved']} | Reason: {doge_slip['reason']}")

    print("\n[TEST-2] SUI Volatility Spike Check")
    sui_slip = risk.validate_trade("SUIUSDT", 1000, [], 0.02, 0.9, price=1.200, side="LONG", stop_loss=1.150)
    print(f"SUI Validated: {sui_slip['approved']} | Reason: {sui_slip['reason']}")
    
    print("\n[TEST-3] XRP Trailing Stop Precision")
    pos_xrp = PositionSchema(symbol="XRPUSDT", side="LONG", amount=100.0, entry_price=0.500, stop_loss=0.480, take_profit=0.600)
    # Simulation doesn't talk to exchange but we verify types
    print(f"XRP Mock Position: {pos_xrp}")
    
    print("\n[TEST-4] ADA Partial Exit Precision")
    exec_engine.close_partial("ADAUSDT", pos_xrp, 50.0) # Will just print logs since simulation_mode=True
    
    print("\n[TEST-5] Binance / Bitget Precision Loading")
    try:
        dh.exchange.load_markets()
        market_xrp = dh.exchange.market(dh.format_symbol("XRPUSDT"))
        print(f"XRP Min Qty: {market_xrp.get('limits', {}).get('amount', {}).get('min')}")
        print(f"XRP Min Cost: {market_xrp.get('limits', {}).get('cost', {}).get('min')}")
    except Exception as e:
        print(f"Error fetching precision: {e}")
        
    print("\n[TEST-6] System Operational Scaling (Rate Limits)")
    print(f"Active Symbols after config override: {config.SYMBOLS}")

if __name__ == "__main__":
    simulate_tests()
