"""
GPU-Accelerated Bootstrapping Module
Optimized for NVIDIA GPUs (Kaggle T4x2)
"""
import numpy as np
import pandas as pd
from typing import Tuple
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU acceleration available via CuPy")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. Falling back to NumPy (CPU)")

from timebudget import timebudget


class GPUBootstrap:
    """GPU-accelerated bootstrap resampling"""
    
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
    def bootstrap_iid_gpu(self, series: pd.Series, n_samples: int = 1000, 
                          seed: int = None) -> pd.DataFrame:
        """
        GPU-accelerated IID bootstrap
        
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
        df = pd.DataFrame(bootstrap_samples, columns=[f'path{i+1}' for i in range(n_samples)])
        df.index = series.index
        
        return df
    
    @timebudget
    def bootstrap_block_gpu(self, series: pd.Series, block_size: int = 12,
                           n_samples: int = 1000, method: str = 'moving',
                           seed: int = None) -> pd.DataFrame:
        """
        GPU-accelerated block bootstrap
        
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
        
        if method == 'circular':
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
                        block = data[start_idx:start_idx + block_size]
                    else:
                        # Wrap around
                        part1 = data[start_idx:]
                        part2 = data[:(start_idx + block_size) % nobs]
                        block = self.xp.concatenate([part1, part2])
                    
                    end_pos = min(pos + len(block), nobs)
                    sample[pos:end_pos] = block[:end_pos - pos]
                    pos = end_pos
                
                bootstrap_samples[:, i] = sample
                
        elif method == 'moving':
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
                    block = data[start_idx:start_idx + block_size]
                    
                    end_pos = min(pos + len(block), nobs)
                    sample[pos:end_pos] = block[:end_pos - pos]
                    pos = end_pos
                
                bootstrap_samples[:, i] = sample
        
        # Convert back to CPU
        bootstrap_samples = self._to_numpy(bootstrap_samples)
        
        # Create DataFrame
        df = pd.DataFrame(bootstrap_samples, columns=[f'path{i+1}' for i in range(n_samples)])
        df.index = series.index
        
        return df


# Convenience functions matching original API
def BootstrapIDD_GPU(series: pd.Series, n_samples: int = 1000, seed: int = None) -> pd.DataFrame:
    """GPU-accelerated IID bootstrap"""
    gpu_bs = GPUBootstrap(use_gpu=True)
    return gpu_bs.bootstrap_iid_gpu(series, n_samples, seed)


def BootstrapCircular_GPU(series: pd.Series, block_size: int = 12, 
                          n_samples: int = 1000, seed: int = None) -> pd.DataFrame:
    """GPU-accelerated circular block bootstrap"""
    gpu_bs = GPUBootstrap(use_gpu=True)
    return gpu_bs.bootstrap_block_gpu(series, block_size, n_samples, 'circular', seed)


def BootstrapMovingBlock_GPU(series: pd.Series, block_size: int = 12,
                             n_samples: int = 1000, seed: int = None) -> pd.DataFrame:
    """GPU-accelerated moving block bootstrap"""
    gpu_bs = GPUBootstrap(use_gpu=True)
    return gpu_bs.bootstrap_block_gpu(series, block_size, n_samples, 'moving', seed)
