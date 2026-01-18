"""
Bootstrapping Module with Automatic GPU Detection
Optimized for both CPU and GPU environments
"""

import warnings

import numpy as np
import pandas as pd
from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    optimal_block_length,
)
from timebudget import timebudget
from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import *
from tsmoothie.utils_func import _id_cb_bootstrap, _id_mb_bootstrap, _id_nb_bootstrap, _id_s_bootstrap

from myPortfolioManagement.myUtils import _helper

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    # Test if GPU is actually available and functional
    try:
        _ = cp.array([1, 2, 3])  # Try a simple operation
        GPU_AVAILABLE = True
        print("✓ GPU acceleration available via CuPy")
    except Exception as e:
        # CuPy imported but GPU not functional
        cp = np
        GPU_AVAILABLE = False
        warnings.warn(f"CuPy installed but GPU not functional ({type(e).__name__}). Using CPU-only mode")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. Using CPU-only mode")


class GPUBootstrap:
    """GPU-accelerated bootstrap resampling (auto-falls back to CPU if GPU unavailable)"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def _to_numpy(self, array):
        """Convert CuPy array to NumPy if needed"""
        if self.use_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    def _to_gpu(self, array):
        """Convert NumPy array to CuPy if GPU is available"""
        if self.use_gpu and isinstance(array, np.ndarray):
            return cp.asarray(array)
        return array

    @timebudget
    def bootstrap_iid_gpu(self, series: pd.Series, n_samples: int = 1000, seed: int = None) -> pd.DataFrame:
        """
        GPU-accelerated IID bootstrap (auto-detects GPU availability)

        Args:
            series: Time series data
            n_samples: Number of bootstrap samples
            seed: Random seed

        Returns:
            DataFrame with bootstrap samples
        """
        if seed is not None:
            if self.use_gpu:
                cp.random.seed(seed)
            else:
                np.random.seed(seed)

        nobs = len(series)
        data = self._to_gpu(series.values)

        # Generate all random indices at once (vectorized)
        random_indices = self.xp.random.randint(0, nobs, size=(nobs, n_samples))

        # Sample all at once using advanced indexing
        bootstrap_samples = data[random_indices]

        # Convert back to CPU if needed
        bootstrap_samples = self._to_numpy(bootstrap_samples)

        # Create DataFrame
        df = pd.DataFrame(bootstrap_samples, columns=[f"path{i+1}" for i in range(n_samples)])
        df.index = series.index

        return df

    @timebudget
    def bootstrap_block_gpu(
        self, series: pd.Series, block_size: int = 12, n_samples: int = 1000, method: str = "moving", seed: int = None
    ) -> pd.DataFrame:
        """
        GPU-accelerated block bootstrap (auto-detects GPU availability)

        Args:
            series: Time series data
            block_size: Size of blocks
            n_samples: Number of bootstrap samples
            method: 'moving', 'circular', or 'stationary'
            seed: Random seed

        Returns:
            DataFrame with bootstrap samples
        """
        if seed is not None:
            if self.use_gpu:
                cp.random.seed(seed)
            else:
                np.random.seed(seed)

        nobs = len(series)
        data = self._to_gpu(series.values)

        # Pre-allocate output array
        bootstrap_samples = self.xp.zeros((nobs, n_samples), dtype=data.dtype)

        if method == "circular":
            # Circular block bootstrap - wrap around
            n_blocks = int(np.ceil(nobs / block_size))

            for i in range(n_samples):
                sample = self.xp.zeros(nobs, dtype=data.dtype)
                pos = 0

                for _ in range(n_blocks):
                    if pos >= nobs:
                        break
                    start_idx = self.xp.random.randint(0, nobs)

                    # Handle wrapping
                    if start_idx + block_size <= nobs:
                        block = data[start_idx : start_idx + block_size]
                    else:
                        # Wrap around
                        part1 = data[start_idx:]
                        part2 = data[: (start_idx + block_size) % nobs]
                        block = self.xp.concatenate([part1, part2])

                    end_pos = min(pos + len(block), nobs)
                    sample[pos:end_pos] = block[: end_pos - pos]
                    pos = end_pos

                bootstrap_samples[:, i] = sample

        elif method == "moving":
            # Moving block bootstrap
            max_start = nobs - block_size
            n_blocks = int(np.ceil(nobs / block_size))

            for i in range(n_samples):
                sample = self.xp.zeros(nobs, dtype=data.dtype)
                pos = 0

                for _ in range(n_blocks):
                    if pos >= nobs:
                        break
                    start_idx = self.xp.random.randint(0, max_start + 1)
                    block = data[start_idx : start_idx + block_size]

                    end_pos = min(pos + len(block), nobs)
                    sample[pos:end_pos] = block[: end_pos - pos]
                    pos = end_pos

                bootstrap_samples[:, i] = sample

        # Convert back to CPU
        bootstrap_samples = self._to_numpy(bootstrap_samples)

        # Create DataFrame
        df = pd.DataFrame(bootstrap_samples, columns=[f"path{i+1}" for i in range(n_samples)])
        df.index = series.index

        return df


def BootstrapIID(series: pd.Series = None, n_samples: int = 1000, seed: int = None, use_gpu: bool = None):
    """
    IID Bootstrap with automatic GPU detection

    Args:
        series: Time series data
        n_samples: Number of bootstrap samples
        seed: Random seed
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)

    Returns:
        DataFrame with bootstrap samples
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE

    # Use GPU-accelerated version if available and requested
    if use_gpu and GPU_AVAILABLE:
        gpu_bs = GPUBootstrap(use_gpu=True)
        return gpu_bs.bootstrap_iid_gpu(series, n_samples, seed)

    # Fall back to CPU version
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


# Backwards compatibility alias (was misspelled as BootstrapIDD)
BootstrapIDD = BootstrapIID


def BootstrapStationary(
    series: pd.Series = None,
    block_size: int or float = 12,
    n_samples: int = 1000,
    seed=None,
    optimal_block: bool = False,
):
    # # Initialize with entropy from random.org
    # if seed is not None:
    #     entropy = [877788388, 418255226, 989657335, 69307515]
    #     seed = np.random.default_rng(entropy)

    if optimal_block:
        block_size = optimal_block_length(series)
        block_size = block_size.loc[series.name, "stationary"]

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


def BootstrapCircular(
    series: pd.Series = None,
    block_size: int or float = 12,
    n_samples: int = 1000,
    seed=None,
    optimal_block: bool = False,
    use_gpu: bool = None,
):
    """
    Circular Block Bootstrap with automatic GPU detection

    Args:
        series: Time series data
        block_size: Size of blocks
        n_samples: Number of bootstrap samples
        seed: Random seed
        optimal_block: Whether to use optimal block size
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)

    Returns:
        DataFrame with bootstrap samples
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE

    if optimal_block:
        opt = optimal_block_length(series)
        block_size = int(max(opt.loc[series.name, "circular"], 1))

    # Use GPU-accelerated version if available and requested
    if use_gpu and GPU_AVAILABLE:
        gpu_bs = GPUBootstrap(use_gpu=True)
        return gpu_bs.bootstrap_block_gpu(series, block_size, n_samples, "circular", seed)

    # Fall back to CPU version
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


def BootstrapMovingBlock(
    series: pd.Series = None, block_size: int or float = 12, n_samples: int = 1000, seed=None, use_gpu: bool = None
):
    """
    Moving Block Bootstrap with automatic GPU detection

    Args:
        series: Time series data
        block_size: Size of blocks
        n_samples: Number of bootstrap samples
        seed: Random seed
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None)

    Returns:
        DataFrame with bootstrap samples
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE

    # Use GPU-accelerated version if available and requested
    if use_gpu and GPU_AVAILABLE:
        gpu_bs = GPUBootstrap(use_gpu=True)
        return gpu_bs.bootstrap_block_gpu(series, block_size, n_samples, "moving", seed)

    # Fall back to CPU version
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


def bootstrappingTS_smoothie(
    series: pd.Series = None,
    bootstrap_type: str = "mbb",
    block_size: int = 12,
    n_samples: int = None,
    residual_method: bool = False,
    optimal_block: bool = False,
):
    # btype = ['nbb', 'mbb', 'cbb', 'sb'][1]

    if isinstance(series, pd.DataFrame):
        series = pd.Series(series.iloc[:, 0])

    if optimal_block:
        opt = optimal_block_length(series.dropna())
        if bootstrap_type == "cbb":
            block_size = opt["circular"][0]
        elif bootstrap_type == "stationary":
            block_size = opt["stationary"][0]

    if residual_method:
        spc = SpectralSmoother(smooth_fraction=0.18, pad_len=12)
        bts = BootstrappingWrapper(spc, bootstrap_type=bootstrap_type, block_length=block_size)
        bts_samples = bts.sample(data=series.dropna(), n_samples=n_samples)
        df = pd.DataFrame(bts_samples.transpose())
        df = _helper(df, series, n_samples)
        return df
    else:

        bootstrap_functions = {
            "nbb": _id_nb_bootstrap,
            "mbb": _id_mb_bootstrap,
            "cbb": _id_cb_bootstrap,
            "sbb": _id_s_bootstrap,
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
def bootstrappingTS(
    series: pd.Series or pd.DataFrame = None,
    block_size: int = None,
    optimal_block: bool = False,
    n_samples: int = 1000,
    bootstrap_type: bool = "mbb",
    seed: int or None = None,
) -> pd.DataFrame:
    # convert dataframe to series. If a multi-column df is passed would only consider the first column.
    if isinstance(series, pd.DataFrame):
        series = pd.Series(series.iloc[:, 0], name=series.columns[0])

    func_map = {"nbb": BootstrapIID, "sb": BootstrapStationary, "mbb": BootstrapMovingBlock, "cbb": BootstrapCircular}

    BootstrapFunc = func_map.get(bootstrap_type)

    if BootstrapFunc is None:
        raise ValueError(f"Invalid bootstrap type: {bootstrap_type}")

    kwargs = {}
    if BootstrapFunc is not BootstrapIID:
        kwargs["block_size"] = block_size
        kwargs["optimal_block"] = optimal_block

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
