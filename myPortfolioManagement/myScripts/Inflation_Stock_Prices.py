from myPortfolioManagement.myData import get_fred_data
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np

df = get_fred_data(['CPIAUCSL', 'NASDAQCOM'], freq='a')

# Stationarity in yearly percentage change
for c in list(df.columns.values):
    df[c + "_diff"] = (df[c] - df[c].shift(12)) / df[c].shift(12) * 100

df = df.reset_index()
df['Decade'] = df['Date'].dt.year // 10 * 10
df = df.dropna()
df['StockMarket'] = np.log(df['NASDAQCOM'])

# select only the numerical columns from the dataframe
df_num = df.select_dtypes(include=['float64'])

# create the scaler object
scaler = MinMaxScaler(feature_range=(-1, 1))

# fit and transform the numerical columns
df_num_normalized = scaler.fit_transform(df_num)

# convert the normalized data back to a dataframe
df_num_normalized = pd.DataFrame(df_num_normalized, columns=df_num.columns)

# join the normalized numerical columns back to the original dataframe
df_normalized = df.join(df_num_normalized, rsuffix='_normalized')

# view the first 5 rows of the normalized dataframe
print(df_normalized.head())

# per decade
fig = px.scatter(df[['Decade', 'CPIAUCSL_diff', 'NASDAQCOM', 'StockMarket']], x="CPIAUCSL_diff", y="StockMarket",
                 facet_col="Decade", trendline='lowess', trendline_color_override="black")
fig.show(renderer="browser")

fig = go.Figure()


# per decade
fig = px.scatter(df[['CPIAUCSL_diff', 'StockMarket']], x="CPIAUCSL_diff", y="StockMarket",trendline='lowess', trendline_color_override="black")
fig.show(renderer="browser")

fig = go.Figure()

# when annual data
df['inf'] =  df['CPIAUCSL'].pct_change()
# per decade
fig = px.scatter(df[['inf', 'StockMarket']], x="inf", y="StockMarket",trendline='lowess', trendline_color_override="black")
fig.show(renderer="browser")
fig = go.Figure()