# test_install.py
print("Testing imports...")

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import riskfolio as rp
    import quantstats_lumi as qs
    from pypfopt import EfficientFrontier
    import ffn
    print("✓ Core dependencies OK")
except ImportError as e:
    print(f"✗ Import error: {e}")

try:
    from myPortfolioManagement.myData import get_stock_prices
    from myPortfolioManagement.myReturns import calculate_returns
    from myPortfolioManagement.myPortfolioOptimisation import HRP
    from myPortfolioManagement.myPerformanceMetrics import get_main_stats
    print("✓ myPortfolioManagement OK")
except ImportError as e:
    print(f"✗ Package import error: {e}")

# Quick functional test
print("\nRunning quick test...")
tickers = ['AAPL', 'MSFT']
prices = get_stock_prices(tickers, start_date='2024-01-01', wide_format=True)
print(f"✓ Fetched {len(prices)} days of price data")

returns = calculate_returns(prices)
print(f"✓ Calculated returns: {returns.shape}")

print("\n✅ All tests passed!")