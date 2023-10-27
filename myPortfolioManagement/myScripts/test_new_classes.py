import pandas as pd
from myPortfolioManagement.Macroeconomics.getdata import get_FX_spots
from myPortfolioManagement.myDataPreparation import Trends, decomposeTS

df_fx = get_FX_spots(currencies=['EUR'], wide_format=True)

trends = Trends(df_fx)
df1 = trends.make_sma(spans=[50], add_prod=False)
df2 = trends.smooth_series_LowessSmoother(0.1)
df3 = trends.denoise_series_fft()
df4 = trends.denoise_series_kf()
df5 = trends.denoise_series_welvet()


pd.concat([df_fx, df1, df2, df3, df4.iloc[100:, :], df5], axis=1).plot()

df_decp = decomposeTS(df_fx)
df_decp.CFfilter().iloc[:, [0, 2]].plot()
df_decp.HPfilter().iloc[:, [0, 2]].plot()
df_decp.HPfilter()
df_decp.CFfilter()['EURUSD_cycle'].hist()
df_decp.HPfilter()['EURUSD_cycle'].hist()

from openbb_terminal.sdk import openbb
f = openbb.funds.load("Vanguard", "US")
df_fund = openbb.funds.historical(f, "2000-01-01", "2023-10-01")
openbb.funds.historical(f, "2020-01-01", "2020-12-31")

df_fund.plot()
