import pandas as pd
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, IIDBootstrap, MovingBlockBootstrap, \
    optimal_block_length

from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import *
from tsmoothie.utils_func import _id_nb_bootstrap, _id_mb_bootstrap, _id_cb_bootstrap, _id_s_bootstrap


# cleaning
def _helper(df: pd.DataFrame = None, series: pd.Series = None, n_samples: int = None) -> pd.DataFrame:
    string_name = series.name
    cols = [string_name + '_path' + str(x) for x in range(1, n_samples + 1)]
    df.columns = cols
    df.index = series.index

    return df


def BootstrapIDD(series: pd.Series = None, n_samples: int = 1000, seed: int = None):
    nobs = len(series)
    bsidd = IIDBootstrap(series, seed=seed)
    ps_list = []
    row_range = range(0, nobs)
    for pos_data in bsidd.bootstrap(reps=n_samples):
        new_sample = pos_data[0][0]
        new_sample.index = row_range
        ps_list.append(new_sample)

    df = pd.concat(ps_list, axis=1)

    # cleaning
    df = _helper(df, series, n_samples)
    return df


def BootstrapStationary(series: pd.Series = None, block_size: int or float = 12, n_samples: int = 1000, seed=None,
                        optimal_block: bool = False):
    # # Initialize with entropy from random.org
    # if seed is not None:
    #     entropy = [877788388, 418255226, 989657335, 69307515]
    #     seed = np.random.default_rng(entropy)

    if optimal_block:
        block_size = optimal_block_length(series)
        block_size = block_size.loc[series.name, 'stationary']

    nobs = len(series)
    bs = StationaryBootstrap(block_size, series, seed=seed)
    ps_list = []
    row_range = range(0, nobs)
    for pos_data in bs.bootstrap(reps=n_samples):
        new_sample = pos_data[0][0]
        new_sample.index = row_range
        ps_list.append(new_sample)

    df = pd.concat(ps_list, axis=1)

    # cleaning
    df = _helper(df, series, n_samples)

    return df


def BootstrapCircular(series: pd.Series = None, block_size: int or float = 12, n_samples: int = 1000, seed=None,
                      optimal_block: bool = False):
    # # Initialize with entropy from random.org
    # if seed is not None:
    #     entropy = [877788388, 418255226, 989657335, 69307515]
    #     seed = np.random.default_rng(entropy)

    if optimal_block:
        opt = optimal_block_length(series)
        block_size = max(opt.loc[series.name, 'circular'],1)

    nobs = len(series)
    bs = CircularBlockBootstrap(block_size, series, seed=seed)
    ps_list = []
    row_range = range(0, nobs)
    for pos_data in bs.bootstrap(reps=n_samples):
        new_sample = pos_data[0][0]
        new_sample.index = row_range
        ps_list.append(new_sample)

    df = pd.concat(ps_list, axis=1)

    # cleaning
    df = _helper(df, series, n_samples)

    return df


def BootstrapMovingBlock(series: pd.Series = None, block_size: int or float = 12, n_samples: int = 1000, seed=None):
    nobs = len(series)
    bs = MovingBlockBootstrap(block_size, series, seed=seed)
    ps_list = []
    row_range = range(0, nobs)
    for pos_data in bs.bootstrap(reps=n_samples):
        new_sample = pos_data[0][0]
        new_sample.index = row_range
        ps_list.append(new_sample)

    df = pd.concat(ps_list, axis=1)

    # cleaning
    df = _helper(df, series, n_samples)

    return df


# ts smoothie bootsrapping
# =========================

def tsmoothie_boostrapping(series: pd.Series = None, bootstrap_type: str = 'mbb', block_size: int = 12,
                           n_samples: int = None,
                           residual_method: bool = False, optimal_block: bool = False):
    # btype = ['nbb', 'mbb', 'cbb', 'sb'][1]

    if isinstance(series, pd.DataFrame):
        series = pd.Series(series.iloc[:, 0])

    if optimal_block:
        opt = optimal_block_length(series.dropna())
        if bootstrap_type == 'cbb':
            block_size = opt['circular'][0]
        elif bootstrap_type == 'stationary':
            block_size = opt['stationary'][0]

    if residual_method:
        spc = SpectralSmoother(smooth_fraction=0.18, pad_len=12)
        bts = BootstrappingWrapper(spc, bootstrap_type=bootstrap_type, block_length=block_size)
        bts_samples = bts.sample(data=series.dropna(), n_samples=n_samples)
        df = pd.DataFrame(bts_samples.transpose())
        df = _helper(df, series, n_samples)
        return df
    else:

        bootstrap_functions = {
            'nbb': _id_nb_bootstrap,
            'mbb': _id_mb_bootstrap,
            'cbb': _id_cb_bootstrap,
            'sbb': _id_s_bootstrap
        }

        bootstrap_func = bootstrap_functions.get(bootstrap_type, _id_s_bootstrap)

        nobs = len(series)
        bootstrap_data = np.empty((n_samples, nobs))

        for i in np.arange(n_samples):
            bootstrap_id = bootstrap_func(nobs, block_length=block_size)
            bootstrap_data[i] = np.squeeze(series.iloc[bootstrap_id].values)

        df = pd.DataFrame(bootstrap_data.transpose())
        df = _helper(df, series, n_samples)

    return df
