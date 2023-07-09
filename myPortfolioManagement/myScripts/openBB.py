from openbb_terminal.sdk import openbb

# put my keys
openbb.keys.quandl(key="fhbmNKX6oNP7PpFuZJNo")
openbb.keys.fred(key="cc628b51e21828ae6b98c06f4eef6714")

# load data
mystocks = ['SMT.L', 'MSFT', 'AAPL', 'NIO', 'V']

data = openbb.stocks.load('V', start_date="2016-01-01")

sentiment = openbb.stocks.ba.bullbear(mystocks)
openbb.stocks.ba.trending().sort_values('Watchlist Count', ascending=False)

openbb.stocks.ca.balance(['MSFT', 'AAPL'])

openbb.economy.available_indices()
openbb.economy.future(future_type="Indices", sortby="ticker", ascend=False)
openbb.economy.futures(source="WSJ", future_type="Indices")
openbb.economy.glbonds()
openbb.economy.indices()
openbb.economy.overview()
openbb.economy.bigmac().pct_change().plot()
openbb.economy.perfmap(period="1d", map_filter="sp500")

events = openbb.economy.events(countries='United States', start_date='2022-11-01')
events = events[events.consensus != "-"]
events = events.sort_values('Event', ascending=False)

openbb.economy.get_groups()
openbb.economy.spectrum(group="sector", export="")

openbb.economy.macro_parameters()
openbb.economy.macro_countries()

openbb.economy.macro(parameters=['GDEBT'], countries=['Austria'], start_date="1900-01-01", symbol='EUR')
openbb.economy.macro(parameters=['GDEBT'], countries=['Austria'], start_date="1900-01-01", symbol='USD')

openbb.economy.rtps()

openbb.economy.spectrum(group="sector", export="csv")

# https://docs.openbb.co/sdk/reference/economy/treasury
openbb.economy.treasury_maturities()
openbb.economy.treasury(instruments=['inflation'],
                        maturities=['1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y'],
                        frequency="daily", start_date="1900-01-01")

openbb.economy.usbonds()

openbb.economy.valuation(group="sector", sortby="Name", ascend=True)
openbb.economy.ycrv_chart()
