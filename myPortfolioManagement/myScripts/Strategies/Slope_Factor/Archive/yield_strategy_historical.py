import pandas as pd
import numpy as np
import plotly.express as px
from myScripts.functions.performance_stats import mainStats, performance
from myScripts.functions.download_and_prepare_data import make_sma, risk_adjust, calculate_carry,  get_policy_rates, get_bond_yield, get_historical_spots, get_macrobond_data, prices_from_returns
from medh.option_replicator import BaseOptionReplicator
import monthly_returns_heatmap as mrh

'''
Anything  else equal:  
    1.Positive (negative) Spread from policy rate and yield implies that interest rates is expected to be falling (rising)
    2.Positive (negative) spread between policy rates  implies  that US monetary policy tighter (looser) than  foreign, 
      so USD appreciates  (depreciates)    
'''
#
# # get bbq data
# bbq_tickers = ['IOE1 A:00_0_R comdty', 'GC1 A:00_0_R comdty', 'VIX index', 'CL2 A:00_0_R comdty', 'HG1 A:00_0_R comdty']
# df_bbq = get_bbq_data(bbq_tickers)
# df_bbq.columns = ['IronOre_adj_fut', 'Gold_adj_fut', 'Vix', 'WTI_adj_fut', 'Copper_adj_fut']

ccy = ['CHFUSD', 'EURUSD', 'JPYUSD', 'CADUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD', 'SEKUSD'][2]
country = ccy[0:3]
yields = int_name = '2Y'
spot_mb = get_historical_spots(ccy=ccy).dropna()
spot_mb = spot_mb[spot_mb.index >= '1985-01-01']

# spots = get_historical_spots(all_spots=True)[['CHFUSD', 'EURUSD']]
# spots1 = pd.Series(spots.iloc[:, 0])
# spots2 = pd.Series(spots.iloc[:, 1])
# spots1.dropna().rolling(30).corr(spots2.dropna()).hist()

# policy rate
# =============
df_policy_rates = get_policy_rates(country=['US', country])
df_policy_rates['policy_rate_spread'] = df_policy_rates['US'] - df_policy_rates[country]

# load USD
# =========================
df_US_bonds = get_bond_yield(country='US')
df_US_bonds_risk_adj = risk_adjust(df_US_bonds, window=120)

df_bonds = get_bond_yield(country=country)
df_bonds_risk_adj = risk_adjust(df_bonds, window=120)

# calculate carry
df_carry = calculate_carry(df_US_bonds, df_bonds, yield_int='2Y', country=country)
df_carry_risk_adj = calculate_carry(df_US_bonds_risk_adj, df_bonds_risk_adj, yield_int='2Y', country=country)
df_carry_risk_adj.columns = ['carry_risk_adj']

df_carry = df_carry.reset_index().merge(df_carry_risk_adj.reset_index())
df_carry = df_carry.set_index('Date')

# calculate US slope
# ==================
df_US_spread = df_US_bonds.copy()
df_US_spread = df_US_spread.reset_index().merge(df_policy_rates['US'].reset_index())
df_US_spread['spread_US'] = df_US_spread['US'] - df_US_spread[yields]
df_US_spread = df_US_spread[['Date', 'spread_US']]

# Load Non-USD bonds
# =========================
df_spread = df_bonds.copy()
df_spread = df_spread.reset_index().merge(df_policy_rates[country].reset_index())
df_spread['spread_' + country] = df_spread[country] - df_spread[yields]
df_spread = df_spread[['Date', 'spread_' + country]]

df_spreads = df_US_spread.merge(df_spread)
df_spreads = df_spreads.merge(df_policy_rates['policy_rate_spread'].reset_index())
df_spreads = df_spreads.merge(df_carry.reset_index('Date'))
df_spreads = df_spreads.set_index('Date')

# df_spreads_norm = normalise_data(df_spreads, feature_range=(-1, 1)).dropna()
# df_spreads[['carry', 'policy_rate_spread']].plot(secondary_y=['policy_rate_spread'])

# t-stats
# =========
tst = BaseOptionReplicator()
tstats = tst.tstat_deltaRR(df_spreads, decay=0.97)[0]
tstats.columns = tstats.columns + '_tst'

# Make smoothing average
# ==================================================================================================================
df_ema = make_sma(df=df_spreads[df_spreads.index >= '1985-01-01'], spans=[50, 60, 90, 200])

# ==================================================================================================================
df_all = df_spreads.reset_index().merge(df_ema.reset_index())
df_all = df_all.merge(spot_mb.reset_index())
df_all = df_all.merge(tstats.reset_index())
df_all = df_all.set_index('Date')

# df_all[['carry', 'policy_rate_spread']].dropna().plot(secondary_y=['policy_rate_spread'])

# create strategy
# ================
factors = ['factor1', 'factor2', 'factor3', 'factor4']
name_sig = 'spread_' + country + '_tst'

# signals
df_all[factors[0]] = np.where(df_all['carry_risk_adj_sm50'] > df_all['carry_risk_adj_sm50'], -1, 1)
df_all[factors[1]] = np.where(df_all['spread_US'] > df_all['spread_US_sm50'], 1, -1)
df_all[factors[2]] = np.where(df_all['carry_sm50'] > df_all['carry_sm200'], -1, 1)
df_all[factors[3]] = np.where(df_all['carry_risk_adj_sm50'] > df_all['carry_risk_adj_sm200'], -1, 1)

#  composite signal
df_all['Composite'] = 1 / 2 * df_all[factors[1]] + 1 / 2 * df_all[factors[2]]
df_all['Composite_risk_adj'] = 1 / 2 * df_all[factors[1]] + 1 / 2 * df_all[factors[3]]

# strategy performance
df_performance = pd.DataFrame()
df_performance['Composite'] = performance(df_all['Composite'], total_returns=np.log(df_all[ccy]).diff()).cumsum()
df_performance['Composite_risk_adj'] = performance(df_all['Composite_risk_adj'],
                                                   total_returns=np.log(df_all[ccy]).diff()).cumsum()
df_performance['Dollar_Factor'] = performance(df_all[factors[1]], total_returns=np.log(df_all[ccy]).diff()).cumsum()

df_stats = list()
for i in range(0, df_performance.shape[1]):
    df_stats.append(mainStats(df_performance.iloc[:, i].diff()))

df_stats = pd.concat(df_stats, axis=1)
df_stats.columns = df_performance.columns

# plot performance
df_performance.plot()

# monthly performance
returns = performance(df_all[factors[1]], total_returns=np.log(df_all[ccy]).diff())
mrh.plot(returns, eoy=True)
mrh.get(returns)

prices_from_returns(df_performance.dropna().diff()).plot()

nasdaq = get_macrobond_data(['useqin0002'])
nasdaq.columns = ['Nasdaq']

mrh.plot(nasdaq.pct_change(), eoy=True)

nasdaq.index.name = 'Date'

temp = df_performance.reset_index().merge(nasdaq.reset_index()).dropna()
tem_cum = prices_from_returns(temp.set_index('Date').pct_change())

mainStats(tem_cum.iloc[:, 0].diff())

# ======================================================================================================================

performance(df_all['Composite'], total_returns=np.log(df_all[ccy]).diff()).cumsum().plot()
performance(df_all['Composite_risk_adj'], total_returns=np.log(df_all[ccy]).diff()).cumsum().plot()
performance(df_all[factors[1]], total_returns=np.log(df_all[ccy]).diff()).cumsum().plot()
performance(df_all[factors[2]], total_returns=np.log(df_all[ccy]).diff()).cumsum().plot()

fed_factor1 = pd.Series(fed_factor1, name=factors[0])
fed_factor2 = pd.Series(fed_factor2, name=factors[0])
fed_factor3 = pd.Series(fed_factor3, name=factors[0])

df_factors = pd.DataFrame({factors[0]: fed_factor1, factors[1]: fed_factor2, factors[2]: fed_factor3})
df_factors[ccy] = df_all[ccy]
df_pnl = prices_from_returns(df_factors.diff().dropna())

df_pnl['Currency'] = ccy
df_pnl['Signal_' + factors[0]] = df_all[factors[0]]
df_pnl['Signal_' + factors[1]] = df_all[factors[1]]
df_pnl['Signal_' + factors[2]] = df_all[factors[2]]

df_pnl_all = df_pnl.copy()
drop_factors = ['Signal_' + factors[0], 'Signal_' + factors[1], 'Signal_' + factors[2]]
results = pd.melt(df_pnl_all.drop(drop_factors, axis=1).reset_index(), id_vars=['Date', 'Currency'])

# plot
fig = px.line(results, x='Date', y='value', color='variable',
              facet_col='Currency', facet_col_wrap=5)
fig.update_yaxes(matches=None)
fig.show(renderer="browser")

temp = performance(df_all['Composite'], total_returns=np.log(df_all[ccy]).diff()).cumsum()
mainStats(temp.diff())
