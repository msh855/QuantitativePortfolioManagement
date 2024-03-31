import pandas as pd
from scipy.stats import norm
from yahoo_fin.options import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy
import yfinance as yf
import numpy as np

"""
    Calculate the value of an option using the Black-Scholes model

    :param option_type: "call"/"c" or "put"/"p"
    :type option_type: str
    :param S: price of the underlying
    :type S: float
    :param K: strike price of option
    :type K: float
    :param sigma: input implied volatility
    :type sigma: float
    :param t: time to expiration
    :type t: float, optional
    :param r: risk-free rate
    :type r: float, optional
    """

calls = get_calls('nflx', '01/16/2026')
calls.columns = calls.columns.str.lower()
calls["midprice"] = (calls.bid + calls.ask) / 2
calls = calls[calls.midprice > 0]

calls['iv'] = [float(x.replace("%", "")) for x in calls['implied volatility'] ]

calls_sub = calls[(calls.strike > min(calls.strike)) & (calls.strike < max(calls.strike))]

plt.figure(figsize=(12, 6))
plt.plot(calls.strike, calls.iv, ".");
plt.xlabel("strike")
plt.ylabel("price")
plt.savefig("call_prices.png", dpi=400)
plt.show()


def call_value(S, K, sigma, t=0, r=0):
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide='ignore'):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma ** 2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def call_vega(S, K, sigma, t=0, r=0):
    with np.errstate(divide='ignore'):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma ** 2 / 2) * t)
    return np.multiply(S, norm.pdf(d1)) * np.sqrt(t)


def bs_iv(price, S, K, t=0, r=0, precision=1e-4, initial_guess=0.2, max_iter=1000, verbose=False):
    iv = initial_guess
    for _ in range(max_iter):
        P = call_value(S, K, iv, t, r)
        diff = price - P
        if abs(diff) < precision:
            return iv
        grad = call_vega(S, K, iv, t, r)
        iv += diff / grad
    if verbose:
        print(f"Did not converge after {max_iter} iterations")
    return iv


c_test = call_value(100, 110, 0.2, t=1)
print(c_test)

# Check that it works
bs_iv(c_test, 100, 110, t=1)
bs_iv(c_test, 100, 110, t=1)

prices = yf.download('nflx')['Adj Close']
risk_free_rate = yf.download("^TNX", start="2022-11-01", end="2023-11-26")["Close"].iloc[-1] / 100

S = prices[-1]  # price of the underline
t = 3 / 52
calls.midprice
calls.strike

bs_iv(S, calls['midprice'].iloc[-1], calls['strike'].iloc[-1])

calls["iv"] = calls.apply(lambda row: bs_iv(row.midprice, S, row.strike, t=0, max_iter=500), axis=1)


def plot_vol_smile(calls, savefig=False):
    plt.figure(figsize=(9, 6))
    plt.plot(calls.strike, calls.iv, ".")
    plt.xlabel("Strike")
    plt.ylabel("IV")
    if savefig:
        plt.savefig("vol_smile.png", dpi=300)
    plt.show()


calls_no_na = calls.dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
ax1.plot(calls_no_na.strike, calls_no_na.midprice, "r.")
ax1.set_ylabel("Call price")
ax2.plot(calls_no_na.strike, calls_no_na.iv, ".")
ax2.set_ylabel("IV")
ax2.set_xlabel("Strike")
plt.tight_layout()
# plt.savefig("calls_to_iv.png", dpi=400)
plt.show()

calls_clean = calls.dropna().copy()
calls_clean["iv"] = gaussian_filter1d(calls_clean.iv, 3)

calls_no_na = calls.dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
ax1.plot(calls_no_na.strike, calls_no_na.midprice, "r.")
ax1.set_ylabel("Call price")
ax2.plot(calls_no_na.strike, calls_no_na.iv, ".")
ax2.set_ylabel("IV")
ax2.set_xlabel("Strike")
plt.tight_layout()
# plt.savefig("calls_to_iv.png", dpi=400)
plt.show()

calls_clean = calls.dropna().copy()
calls_clean["iv"] = gaussian_filter1d(calls_clean.iv, 3)

plot_vol_smile(calls_clean)
calls_clean = calls_clean[(calls_clean.strike > 300) & (calls_clean.strike < 375)]
plot_vol_smile(calls_clean, savefig=False)

plt.figure(figsize=(9, 6))
vol_surface = scipy.interpolate.interp1d(calls_clean.strike, calls_clean.iv, kind="cubic",
                                         fill_value="extrapolate")
x_new = np.arange(calls_clean.strike.min(), calls_clean.strike.max(), 0.1)
plt.plot(calls_clean.strike, calls_clean.iv, "bx", x_new, vol_surface(x_new), "k-");
plt.legend(["smoothed IV", "fitted smile"], loc="best")
plt.xlabel("Strike")
plt.ylabel("IV")
plt.tight_layout()
# plt.savefig("SPY_smile.png", dpi=300)
plt.show()

import numpy as np
import pandas as pd
from scipy.stats import norm

def generate_sample_option_data():
    # Define the risk-free interest rate (specified as a decimal number between 0 and 1).
    rf = 0.005
    # Define the underlying asset price (specified in currency units).
    S = 100
    # Define the option expiry times (specified in years).
    T0 = np.arange(0.25, 4.01, 0.5)
    # Replicate the option expiry times.
    N = 6
    T = np.repeat(T0, N)
    # Create a vector of option strike prices (specified in currency units).
    K0 = np.linspace(98, 102, N)
    # Define the option strike prices (specified in currency units).
    K = np.tile(K0, len(T) // N)
    # Define the asset volatility (specified as a decimal number between 0 and 1).
    sigma = np.nan * np.ones_like(K)
    for k in range(len(T) // N):
        sigma0 = 1e-4 * (K0 - np.mean(K0)) ** 2 + K0 + k
        sigma[k * N : (k + 1) * N] = sigma0
    # Compute the call and put prices from the Black-Scholes pricing model.
    d1 = (np.log(S / K) + (rf + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2)
    P = K * np.exp(-rf * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    # Add some (reproducible) noise to both the call and put options to simulate real-life market data.
    noise_vol = 0.001
    np.random.seed(0)
    C = C + noise_vol * np.abs(np.random.randn(len(C)))
    P = P + noise_vol * np.abs(np.random.randn(len(P)))
    # Assemble the data in a table.
    D = pd.DataFrame({"K": K, "C": C, "P": P, "T": T})
    # Include the risk-free interest rate and the underlying asset price in the table data.
    U = np.ones(len(D))
    D["rf"] = rf * U
    D["S"] = S * U
    return D


data = generate_sample_option_data()