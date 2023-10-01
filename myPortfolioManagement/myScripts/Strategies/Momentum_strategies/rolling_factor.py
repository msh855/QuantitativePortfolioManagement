
import numpy as np
from myScripts.functions.download_and_prepare_data import make_sma, get_historical_spots, \
    normalise_data, mom_production
import statsmodels.api as sm
from myScripts.functions.performance_stats import  get_PnL
from medh.momentum import BaseMomentum

# plotting
import matplotlib

# matplotlib.use('QtAgg')
import matplotlib.dates as mdates

# plt.interactive(True)
myfmt = mdates.DateFormatter('%Y')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 10})


def vol_adj_spots(df_returns, vol_window=60):
    raw_vol = BaseMomentum.std_vol(df_returns, window=vol_window)
    # Vol normalise returns
    adj_return = df_returns / raw_vol * np.sqrt(252)
    adj_spot = adj_return.cumsum()

    return adj_spot, raw_vol


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]

    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


def get_DFM(data, nfactors=6, lags=12):
    model = sm.tsa.DynamicFactorMQ(data, factors=nfactors, factor_orders=lags, idiosyncratic_ar1=False, iterations=2000)
    results = model.fit(maxiter=5000, disp=500)

    # get factors
    df_factors = results.factors.filtered
    df_factors.columns = ['factor' + str(x) for x in range(1, len(df_factors.columns) + 1)]
    df_factors.index = data.index

    return df_factors[['factor1']]


currencies = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD'][0]

nfactors = 6
lags = 12
window = 5
vol_scale = 0.075
start_date_sample = '2015-01-01'
datasets = []

for ccy in [currencies]:
    print(ccy)
    # download data
    # =============
    df_spots = get_historical_spots(ccy=ccy).dropna()  # get_bbq_ccy(ccy) #
    df_spots = df_spots[df_spots.index >= start_date_sample]

    for x in range(255 * window, len(df_spots[ccy]) - 1):
        print(f'loop{x}')
        df = df_spots[x - 255 * window:x]
        df['return'] = np.log(df[ccy]).diff()
        df['return_cum'] = df['return'].cumsum()
        df.dropna(inplace=True)

        # isolate currency
        # ================
        spot = df[[ccy]]
        spot.columns = ['Spot']

        # performance of production signal
        signals = mom_production(df[['return']])

        # smooth returns
        ret_cum_adj, raw_vol = vol_adj_spots(df[['return']])
        df_spot_sm = make_sma(df=ret_cum_adj.dropna(), spans=[16, 32, 64, 128, 256])
        ret_cum_adj.columns = ['return_cum_risk_adj']
        raw_vol.columns = ['ret_vol']

        #   prepare data for DFMs
        # ======================
        # data_normalised = normalise_data(df_spot_sm)
        data_normalised = remove_outliers(df_spot_sm)  # df_smoth is scale invariant
        factors = get_DFM(data_normalised, nfactors=6, lags=12)

        # plot factors
        # =============
        df_spot_sm = df_spot_sm.join(factors[['factor1']])

        # construct dataset for signals and trading
        # =========================================
        ret_cum_adj_norm = normalise_data(ret_cum_adj)
        ret_cum_adj_norm.columns = ['return_cum_risk_adj_norm']
        signals.columns = ['Prod_mom']

        # collect all
        # ======================================================================================================================
        df_mom = df[['return', 'return_cum']].join(ret_cum_adj)
        df_mom = df_mom.join(ret_cum_adj_norm)
        df_mom = df_mom.join(df_spot_sm)
        df_mom = df_mom.join(signals)
        df_mom = df_mom.join(spot)
        df_mom.dropna(inplace=True)

        # strategies and signals
        df_factor_sm = make_sma(df=factors[['factor1']], spans=[16, 32, 64, 128, 256])
        df_mom = df_mom.join(df_factor_sm)

        for fact in df_factor_sm.columns.tolist():
            if any(df_spot_sm.corr().tail(1).iloc[0] < 0):
                df_mom[fact] = np.where(df_mom[fact] > df_mom['factor1'], 1, -1)
            else:
                df_mom[fact] = np.where(df_mom[fact] > df_mom['factor1'], -1, 1)
            df_mom[fact + '_adj'] = (df_mom[fact] / raw_vol['ret_vol']) * vol_scale
            df_performance = get_PnL(df_mom, signals_list=[fact + '_adj', 'Prod_mom'])

