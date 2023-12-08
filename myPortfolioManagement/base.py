import pandas as pd
import yfinance as yf
from myPortfolioManagement.myUtils import convert_date_index
from operator import itemgetter
from openbb import obb

def _add_stock_main_info(yahoo_ticker: str = None) -> dict:
    stock_info = yf.Ticker(yahoo_ticker)
    d = stock_info.info

    mykeys_l = ['type', 'marketCap', 'longName', 'exchange', 'currency']
    mykeys_exp = mykeys_l + ['industry', 'sector', 'country']

    if not d.keys() & {'industry', 'sector', 'country'}:

        dv = itemgetter('quoteType', 'marketCap', 'longName', 'exchange', 'currency')(d)
        df_dv = pd.DataFrame([dv], columns=mykeys_l)
        df_dv['industry'] = 'Other'
        df_dv['sector'] = 'Other'
        df_dv['country'] = 'Other'
    else:
        dv = itemgetter('quoteType', 'marketCap', 'longName', 'exchange', 'currency', 'industry', 'sector', 'country')(
            d)
        df_dv = pd.DataFrame([dv], columns=mykeys_exp)

    df_dv['currency'] = [x.upper() for x in df_dv['currency']]
    df_dv['yahooTicker'] = yahoo_ticker
    order = ['yahooTicker'] + mykeys_exp
    return df_dv[order]


def _load_stock(yahoo_ticker: str, period: str = 'max', start_date: str = None,
                end_date: str = None, time_interval: str = 'daily', fix_data: bool = False, **kwarg):
    time_interval_mapping = {
        'daily': '1d',
        'monthly': '1M',
        'quarterly': '3M'
    }

    time_interval_temp = time_interval_mapping[time_interval]

    stock = yf.Ticker(yahoo_ticker)
    prices = stock.history(period=period, interval=time_interval_temp,
                           start=start_date, end=end_date, repair=fix_data, **kwarg)
    prices = convert_date_index(prices)
    prices['yahooTicker'] = yahoo_ticker
    return prices


def _load_fx(cross: str = "EURUSD", start_date: str = '1950-01-01', end_date: str = None):
    df_fx = obb.currency.price.historical(symbol=cross, start_date=start_date, end_date=end_date,
                                          provider='yfinance').to_df()
    df_fx = df_fx[['close']]
    df_fx.index.name = 'Date'
    df_fx.columns = [cross]
    #df_fx['fx'] = cross
    #df_fx['currency'] = cross[0:3]
    return df_fx


