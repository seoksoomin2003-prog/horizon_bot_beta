import ccxt
import pandas as pd
import numpy as np

def get_stats():
    exchange = ccxt.binance({'options': {'defaultType': 'future'}})
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT", "DOGEUSDT", "BNBUSDT", "SUIUSDT", "ADAUSDT"]
    
    report = []
    
    try:
        exchange.load_markets()
        for sym in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, '1h', limit=24)
                df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                
                # Volatility (Daily range %)
                df['range'] = (df['h'] - df['l']) / df['l']
                avg_volat = df['range'].mean() * 100
                
                # Liquidity proxy (24h Vol in USD)
                tot_vol_ccy = (df['v'] * df['c']).sum()
                
                report.append({
                    'Symbol': sym,
                    'Liquidity': f"${tot_vol_ccy/1e6:.1f}M (24h)",
                    'Volatility': f"{avg_volat:.2f}% (Avg Hr)",
                    'Strategy Fit': 'HIGH' if avg_volat > 0.15 else 'MED'
                })
            except:
                report.append({'Symbol': sym, 'Liquidity': 'N/A', 'Volatility': 'N/A', 'Strategy Fit': 'UNKNOWN'})
    except Exception as e:
        print(f"Error loading: {e}")
        
    print("| Symbol | Liquidity | Volatility | Strategy Fit |")
    print("|--------|------------|-------------|---------------|")
    for r in report:
        print(f"| {r['Symbol']} | {r['Liquidity']} | {r['Volatility']} | {r['Strategy Fit']} |")

if __name__ == "__main__":
    get_stats()
