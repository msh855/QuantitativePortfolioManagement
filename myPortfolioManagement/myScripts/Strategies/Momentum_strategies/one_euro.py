# https://jaantollander.com/post/noise-filtering-using-one-euro-filter/#fn:1

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from matplotlib.animation import FuncAnimation

from myScripts.functions.download_and_prepare_data import get_data

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat



########################################################################################################################
# EURUSD Example
########################################################################################################################
currency = 'EURUSD'
df_ccy = get_data(currency='EURUSD', get_transformed_data=False)
df_ccy = df_ccy[df_ccy['Currency'] == currency]
df_ccy = df_ccy[['Spot']]

usd_spots = pd.DataFrame(df_ccy[-100:])

start = 0
end = len(usd_spots)

# The noisy signal
t = np.arange(start, end, 1)
x = usd_spots['Spot'].values
x_noisy = x

# The filtered signal
min_cutoff = 0.004
beta = 0.7
x_hat = np.zeros_like(x_noisy)
x_hat[0] = x_noisy[0]
one_euro_filter = OneEuroFilter(
    t[0], x_noisy[0],
    min_cutoff=min_cutoff,
    beta=beta
)
for i in range(1, len(t)):
    x_hat[i] = one_euro_filter(t[i], x_noisy[i])

usd_spots['filtered'] = x_hat

usd_spots['diff']
usd_spots['diff'].mean()