
import pandas as pd
from openbb import obb

df_daily = obb.equity.price.historical(symbol="spy", start_date='1980-01-01', provider="yfinance")
df_daily.to_df()

from openbb import obb

obb.crypto.price.historical(symbol="BTCUSD")

from openbb import obb

obb.user.credentials.fmp_api_key = 'eb50221eaef20292fe4b57f675be8b23'
data = obb.equity.price.historical(symbol="9984.T", interval="1d", start_date='1980-01-01').to_df()

data['adj_close'].plot()
