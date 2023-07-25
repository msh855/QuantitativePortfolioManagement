import pandas as pd

from myPortfolioManagement.myDataPreparation import denoise_series_kf, HPfilter, CFfilter, smooth_series_LowessSmoother
from myPortfolioManagement.myData import get_stock_prices

sp = get_stock_prices(['^GSPC'], wide_format=True)
sp_index = sp['^GSPC']
sp_index.name = 'SP500'

# Smooth Series
SP_sm_kalman = denoise_series_kf(sp_index)
SP_sm_HP = HPfilter(sp_index)
SP_sm_CF = CFfilter(sp_index, 2, 4)
SP_sm_lowess = smooth_series_LowessSmoother(sp_index)

# combine smoothed series
df_sm = pd.DataFrame({'SP_original': sp_index, 'KF': SP_sm_kalman['SP500_smooth_KF'], 'HP': SP_sm_HP['SP500_trend'],
                      'Lowess': SP_sm_lowess['SP500_smooth_Lowess']})

df_sm.plot()
