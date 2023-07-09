# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from myPortfolioManagement.myData import get_fred_data, get_stock_prices
from myPortfolioManagement.myReturns import convert_returns_freq
import statsmodels.api as sm

# Load returns data for the S&P500
Nasdaq_prices = get_fred_data(fred_sumbol=['NASDAQCOM'], freq='a')

# Load returns data for the S&P500
returns = Nasdaq_prices.pct_change()
returns = returns.dropna()
returns = returns['NASDAQCOM']

# Set parameters
mean = returns.mean()  # Annualized average return from historical data
std = returns.std()  # Annualized standard deviation from historical data

n_sim = 10000  # Number of simulations
n_years = [1, 2, 5, 10]  # Time horizons in years

# exceed this value
threshold = 0.30  # Threshold for exceeding initial value

# Initialize empty list for probabilities
probabilities = []

# Loop over time horizons
for n in n_years:
    # Generate n_sim random annual returns from normal distribution
    sample = np.random.normal(mean, std, size=(n_sim, n))
    # Compound returns over time horizon
    final_values = (1 + sample).prod(axis=1)
    # Count how many times final value exceeds threshold
    count = np.sum(final_values > (1 + threshold))
    # Calculate probability as ratio of count to n_sim
    prob = count / n_sim
    # Append probability to list
    probabilities.append(prob)

# Create a pandas series with probabilities and time horizons as index
prob_series = pd.Series(probabilities, index=n_years)

# Plot probabilities as bar chart
prob_series.plot.bar()
plt.xlabel('Time horizon (years)')
plt.ylabel('Probability')
plt.title('Probability that S&P500 will exceed 10%')
plt.show()
