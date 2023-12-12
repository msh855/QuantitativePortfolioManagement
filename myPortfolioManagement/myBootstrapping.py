import pandas as pd
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, IIDBootstrap, MovingBlockBootstrap, \
    optimal_block_length

from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import *
from tsmoothie.utils_func import _id_nb_bootstrap, _id_mb_bootstrap, _id_cb_bootstrap, _id_s_bootstrap
from timebudget import timebudget
from myPortfolioManagement.myUtils import _helper


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
        block_size = int(max(opt.loc[series.name, 'circular'], 1))

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

def bootstrappingTS_smoothie(series: pd.Series = None, bootstrap_type: str = 'mbb', block_size: int = 12,
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


@timebudget
def bootstrappingTS(series: pd.Series = None, block_size: int = None, optimal_block: bool = False,
                    n_samples: int = 1000,
                    bootstrap_type: bool = 'mbb', seed: int or None = None) -> pd.DataFrame:
    func_map = {
        'nbb': BootstrapIDD,
        'sb': BootstrapStationary,
        'mbb': BootstrapMovingBlock,
        'cbb': BootstrapCircular
    }

    BootstrapFunc = func_map.get(bootstrap_type)

    if BootstrapFunc is None:
        raise ValueError(f"Invalid bootstrap type: {bootstrap_type}")

    kwargs = {}
    if BootstrapFunc is not BootstrapIDD:
        kwargs['block_size'] = block_size
        kwargs['optimal_block'] = optimal_block

    if BootstrapFunc is not BootstrapMovingBlock:
        results = BootstrapFunc(series, n_samples=n_samples, seed=seed, **kwargs)
    else:
        results = BootstrapFunc(series, n_samples=n_samples, seed=seed, block_size=block_size)

    # if BootstrapFunc in [BootstrapCircular, BootstrapStationary]:
    #     print('I am here')
    #     results = BootstrapFunc(series, n_samples=n_samples, seed=seed, block_size=block_size,
    #                             optimal_block=optimal_block)
    # elif BootstrapFunc is BootstrapMovingBlock:
    #     results = BootstrapFunc(series, n_samples=n_samples, seed=seed, block_size=block_size)
    # else:
    #     results = BootstrapFunc(series, n_samples=n_samples, seed=seed)

    return results
