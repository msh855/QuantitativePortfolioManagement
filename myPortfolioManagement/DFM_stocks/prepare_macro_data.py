import types
import numpy as np
import pandas as pd

import os


def transform(column, transforms):
    transformation = transforms[column.name]
    # For quarterly data like GDP, we will compute
    # annualized percent changes
    mult = 4 if column.index.freqstr[0] == 'Q' else 1

    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference, multiplied by 100
    #      (i.e. approximate percent change)
    #      with optional multiplier for annualization
    elif transformation == 5:
        column = np.log(column).diff() * 100 * mult
    # 6 => Log second difference, multiplied by 100
    #      with optional multiplier for annualization
    elif transformation == 6:
        column = np.log(column).diff().diff() * 100 * mult
    # 7 => Exact percent change, multiplied by 100
    #      with optional annualization
    elif transformation == 7:
        column = ((column / column.shift(1)) ** mult - 1.0) * 100

    return column


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


def load_fredmd_data(vintage):
    base_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/'

    # - FRED-MD --------------------------------------------------------------
    # 1. Download data
    orig_m = (pd.read_csv(f'{base_url}/monthly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract transformation information
    transform_m = orig_m.iloc[0, 1:]
    orig_m = orig_m.iloc[1:]

    # 3. Extract the date as an index
    orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
    orig_m.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_m = orig_m.apply(transform, axis=0,
                         transforms=transform_m)

    # 5. Remove outliers (but not in 2020)
    dta_m.loc[:'2019-12'] = remove_outliers(dta_m.loc[:'2019-12'])

    # - FRED-QD --------------------------------------------------------------
    # 1. Download data
    orig_q = (pd.read_csv(f'{base_url}/quarterly/{vintage}.csv')
              .dropna(how='all'))

    # 2. Extract factors and transformation information
    factors_q = orig_q.iloc[0, 1:]
    transform_q = orig_q.iloc[1, 1:]
    orig_q = orig_q.iloc[2:]

    # 3. Extract the date as an index
    orig_q.index = pd.PeriodIndex(orig_q.sasdate.tolist(), freq='Q')
    orig_q.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_q = orig_q.apply(transform, axis=0,
                         transforms=transform_q)

    # 5. Remove outliers (but not in 2020)
    dta_q.loc[:'2019Q4'] = remove_outliers(dta_q.loc[:'2019Q4'])

    # - Output datasets ------------------------------------------------------
    return types.SimpleNamespace(
        orig_m=orig_m, orig_q=orig_q,
        dta_m=dta_m, transform_m=transform_m,
        dta_q=dta_q, transform_q=transform_q, factors_q=factors_q)


def get_macro_data(csv_dates=['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']):

    # Load the vintages of data from FRED
    dta = {date: load_fredmd_data(date)
           for date in csv_dates}

    data_path = "S:\Investment Solutions Group\Quant_research\Moustafa\Github\QuantitativePortfolioManagement\myPortfolioManagement\Data"

    data_des_month = os.path.join(data_path, 'fredmd_definitions.csv')
    data_des_quart = os.path.join(data_path, 'fredqd_definitions.csv')

    # Definitions from the Appendix for FRED-MD variables
    defn_m = pd.read_csv(data_des_month)
    defn_m.index = defn_m.fred

    # Definitions from the Appendix for FRED-QD variables
    defn_q = pd.read_csv(data_des_quart)
    defn_q.index = defn_q.fred

    # Replace the names of the columns in each monthly and quarterly dataset
    map_m = defn_m['description'].to_dict()
    map_q = defn_q['description'].to_dict()
    for date, value in dta.items():
        value.orig_m.columns = value.orig_m.columns.map(map_m)
        value.dta_m.columns = value.dta_m.columns.map(map_m)
        value.orig_q.columns = value.orig_q.columns.map(map_q)
        value.dta_q.columns = value.dta_q.columns.map(map_q)

    # Get the mapping of variable id to group name, for monthly variables
    groups = defn_m[['description', 'group']].copy()

    # Re-order the variables according to the definition CSV file
    # (which is ordered by group)
    columns = [name for name in defn_m['description']
               if name in dta['2020-02'].dta_m.columns]
    for date in dta.keys():
        dta[date].dta_m = dta[date].dta_m.reindex(columns, axis=1)

    # Add real GDP (our quarterly variable) into the "Output and Income" group
    gdp_description = defn_q.loc['GDPC1', 'description']
    groups.loc['GDPC1'] = {'description': gdp_description, 'group': 'Output and Income'}

    # Construct the variable => list of factors dictionary
    factors = {row['description']: ['Global', row['group']]
               for ix, row in groups.iterrows()}

    # Get the baseline monthly and quarterly datasets
    start = '2000'
    endog_m = dta['2020-02'].dta_m.loc[start:, :]
    gdp_description = defn_q.loc['GDPC1', 'description']
    endog_q = dta['2020-02'].dta_q.loc[start:, [gdp_description]]

    return endog_m, endog_q, factors

