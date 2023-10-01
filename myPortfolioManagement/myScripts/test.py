import pandas as pd
from myPortfolioManagement.myData import get_FX_spots
from myPortfolioManagement.myDataPreparation import Trends, decomposeTS
from openbb_terminal.sdk import openbb

df_fx = get_FX_spots(currencies=['EUR'], wide_format=True)

trends = Trends(df_fx)

df1 = trends.make_sma(spans=[50], add_prod=False)
df2 = trends.smooth_series_LowessSmoother(0.1)
df3 = trends.denoise_series_fft()
df4 = trends.denoise_series_kf()
df5 = trends.denoise_series_welvet()

df_decp = decomposeTS(df_fx)
df_decp.CFfilter().iloc[:, [0, 2]].plot()
df_decp.HPfilter().iloc[:, [0, 2]].plot()
df_decp.HPfilter()

pd.concat([df_fx, df1, df2, df3, df4.iloc[100:, :], df5], axis=1).plot()

temp_decomp = openbb.qa.decompose(data=df_fx, multiplicative=True)

df_decompose = pd.concat([df_fx, temp_decomp[1], temp_decomp[2]], axis=1).dropna()
df_decompose.columns = [df_fx.columns[0], df_fx.columns[0] + '_cycle', df_fx.columns[0] + '_trend']

temp_decomp[0].trend[temp_decomp[0].trend.notna().values]



type(df_fx.index)
if df_fx.index.inferred_type != "datetime64":
    print('not a da')

import pandas as pd


def check_date_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a date index")


# Example usage:
df = pd.DataFrame({'A': [1, 2, 3]})
check_date_index(df)  # This will raise a ValueError
