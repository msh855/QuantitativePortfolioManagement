#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Fanchart Generator
=============================
This script demonstrates the complete workflow for:
1. Downloading stock prices using myPortfolioManagement
2. Calculating returns
3. Constructing a portfolio based on custom weights
4. Simulating the portfolio using bootstrap methods
5. Producing a fanchart visualization

Uses the myPortfolioManagement library following tutorial patterns.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.4f}'.format)

# ============================================================================
# IMPORTS FROM myPortfolioManagement
# ============================================================================

from myPortfolioManagement.myData import get_stock_prices, get_stock_info
from myPortfolioManagement.myReturns import calculate_returns, calculate_portfolio_returns
from myPortfolioManagement.myBacktesting import fan_chart

# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# List of stock and crypto tickers to download (Yahoo Finance USD format)
TICKERS = ["IGLN.L", "MSFT", "ABBV", "NVDA", "CRWD", "V", "JPM", "SOL-USD", "BTC-USD", "ETH-USD"]

# Portfolio weights calculated from total combined value (~$8,100 USD)
# Weights are based on the GBP values from your provided images
WEIGHTS = {
    'IGLN.L': 0.1419,   # iShares Physical Gold
    'MSFT': 0.1172,     # Microsoft
    'ABBV': 0.1070,     # AbbVie
    'NVDA': 0.0967,     # Nvidia
    'CRWD': 0.0762,     # Crowdstrike
    'V': 0.0527,        # Visa
    'JPM': 0.0522,      # JPMorgan Chase & Co
    'SOL-USD': 0.1719,  # Solana
    'BTC-USD': 0.1365,  # Bitcoin
    'ETH-USD': 0.0477   # Ethereum
}

# Verification: sum(WEIGHTS.values()) should be approx 1.0
# Date range for historical data
START_DATE = '2016-01-01'
END_DATE = None  # None = today's date

# Fanchart simulation parameters
OUT_OF_SAMPLE_DATE = '2022-11-30'  # Date from which simulation begins (ChatGPT launch)
N_SIMULATIONS = 5000  # Number of bootstrap simulations
STARTING_VALUE = 1.0  # Starting portfolio value

# Portfolio name
PORTFOLIO_NAME = 'My Custom Portfolio'


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PORTFOLIO FANCHART GENERATOR")
    print("Using myPortfolioManagement Library")
    print("=" * 70)

    # =========================================================================
    # STEP 1: DOWNLOAD STOCK PRICES
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Fetching Stock Prices")
    print("=" * 70)

    # Fetch historical prices using myPortfolioManagement
    prices = get_stock_prices(
        yahoo_tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        freq='daily',
        wide_format=True
    )

    print(f"\nPrice data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print("\nFirst 5 rows of prices:")
    print(prices.head())

    # Get stock information
    print("\nStock Information:")
    stock_info = get_stock_info(TICKERS)
    print(stock_info[['yahooTicker', 'longName', 'currency']].to_string())

    # =========================================================================
    # STEP 2: CALCULATE RETURNS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Calculating Returns")
    print("=" * 70)

    # Calculate daily returns using myPortfolioManagement
    returns = calculate_returns(prices, log_returns=False)

    print(f"\nReturns shape: {returns.shape}")
    print("\nDaily Returns Statistics:")
    print(returns.describe().T[['mean', 'std', 'min', 'max']])

    # =========================================================================
    # STEP 3: CONSTRUCT PORTFOLIO RETURNS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Constructing Portfolio")
    print("=" * 70)

    # Get the mapping from tickers to long names
    # The prices DataFrame uses long names as columns when wide_format=True
    ticker_to_name = dict(zip(stock_info['yahooTicker'], stock_info['longName']))
    
    # Convert weights to use long names (matching returns columns)
    WEIGHTS_LONGNAME = {ticker_to_name[ticker]: weight 
                        for ticker, weight in WEIGHTS.items()}

    # Validate weights sum to 1
    total_weight = sum(WEIGHTS_LONGNAME.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
        WEIGHTS_LONGNAME = {k: v / total_weight for k, v in WEIGHTS_LONGNAME.items()}

    print(f"\nPortfolio '{PORTFOLIO_NAME}' weights:")
    for name, weight in WEIGHTS_LONGNAME.items():
        print(f"  {name}: {weight:.2%}")

    # Create weights DataFrame in the format expected by calculate_portfolio_returns
    # The function expects a DataFrame with asset names in first column and weights in second
    weights_df = pd.DataFrame({
        'asset': list(WEIGHTS_LONGNAME.keys()),
        'weight': list(WEIGHTS_LONGNAME.values())
    })

    # Calculate portfolio returns using the library's robust function
    portfolio_returns = calculate_portfolio_returns(
        returns=returns,
        myweights=weights_df,
        portfolio_name=PORTFOLIO_NAME
    )

    # Print portfolio statistics
    print(f"\nPortfolio Statistics:")
    print(f"  Mean daily return: {portfolio_returns[PORTFOLIO_NAME].mean():.4%}")
    print(f"  Daily volatility: {portfolio_returns[PORTFOLIO_NAME].std():.4%}")
    print(f"  Annualized return: {portfolio_returns[PORTFOLIO_NAME].mean() * 252:.2%}")
    print(f"  Annualized volatility: {portfolio_returns[PORTFOLIO_NAME].std() * np.sqrt(252):.2%}")

    # =========================================================================
    # STEP 4: CREATE FANCHART
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Creating Fanchart")
    print("=" * 70)

    print(f"\nGenerating fanchart with {N_SIMULATIONS} bootstrap simulations...")
    print(f"Out-of-sample date: {OUT_OF_SAMPLE_DATE}")

    # Create fanchart using myPortfolioManagement
    # This shows bootstrap-simulated possible paths for the portfolio
    fan_chart(
        returns=portfolio_returns,
        out_of_sample_date=OUT_OF_SAMPLE_DATE,
        n_sample=N_SIMULATIONS,
        starting_value=STARTING_VALUE,
        chart_title=f'{PORTFOLIO_NAME} - Bootstrap Fan Chart'
    )

    fan_chart(
        returns=portfolio_returns,
        weight_period=['2021-01-01', '2022-11-29'],
        out_of_sample_date=OUT_OF_SAMPLE_DATE,
        n_sample=N_SIMULATIONS,
        starting_value=STARTING_VALUE,
        chart_title=f'{PORTFOLIO_NAME} - Bootstrap Fan Chart [Rregime Weight]'
    )

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
