from aeon.forecasting.trend import TrendForecaster
import pandas as pd

def forecast_trend(data: pd.DataFrame or pd.Series = None, steps: list = [1, 2, 3], feq: str = 'd'):
    y = data.to_period(feq)
    forecaster = TrendForecaster()
    forecaster.fit(y)  # fit the forecaster
    pred = forecaster.predict(fh=steps)  # predict the next value
    return pred

