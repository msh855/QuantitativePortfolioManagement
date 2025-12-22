# Performance Metrics Standardization

## Overview

This document explains the standardization of performance metrics calculations in the myPortfolioManagement library, specifically addressing the inconsistency between different packages (QuantStats vs FFN).

## Problem Statement

Previously, the library used different methods to calculate CAGR (Compound Annual Growth Rate):

1. **QuantStats method** (`qs.stats.cagr`): Uses `num_periods / periods_per_year`
   - Example: 1461 days / 365 = 4.003 years
   - Formula: `(compound_return) ^ (periods_per_year / num_periods) - 1`

2. **FFN method** (`ffn.core.calc_cagr`): Uses `year_frac()` for actual calendar time
   - Example: 3.997 years (accounting for leap years and exact dates)
   - Formula: `(end_value / start_value) ^ (1 / year_frac) - 1`

This inconsistency led to different CAGR values (typically 0.1-0.5% difference) for the same data, depending on which function was used.

## Solution

The library now uses **FFN's year_frac method** as the standard across all performance metric calculations.

### Why year_frac?

1. **Financial Industry Standard**: Reflects actual calendar time, which is what investors experience
2. **Leap Year Accuracy**: Properly accounts for leap years and varying month lengths
3. **Frequency Agnostic**: Works correctly with daily, weekly, monthly, or irregular data
4. **Consistency**: Provides the same result regardless of data frequency

## Updated Functions

### 1. `cagr(df_prices)` - *No change*
Already used FFN's `calc_cagr` internally. Now has enhanced documentation.

```python
from myPortfolioManagement.myPerformanceMetrics import cagr

# Calculate CAGR from price series or DataFrame
cagr_values = cagr(prices)
```

### 2. `cagr_from_returns(returns)` - *New function*
Standardized CAGR calculation from returns data.

```python
from myPortfolioManagement.myPerformanceMetrics import cagr_from_returns

# Calculate CAGR from return series or DataFrame
cagr_values = cagr_from_returns(returns)
```

### 3. `get_main_stats(returns)` - *Updated*
Now uses `cagr_from_returns()` instead of `qs.stats.cagr(returns, periods=365)`.

```python
from myPortfolioManagement.myPerformanceMetrics import get_main_stats

# Get comprehensive performance stats with standardized CAGR
stats = get_main_stats(returns, rf=0.04)
# stats['cagr'] now uses year_frac methodology
```

### 4. `bootstrap_stats(returns)` - *Updated*
Bootstrap simulations now use standardized CAGR.

```python
from myPortfolioManagement.myBacktesting import bootstrap_stats

# Bootstrap performance statistics with consistent CAGR
bootstrap_results = bootstrap_stats(returns, n_sim=1000)
```

### 5. `performance_overview(df)` - *No change*
Already used FFN internally. Now has enhanced documentation.

```python
from myPortfolioManagement.myPerformanceMetrics import performance_overview

# Generate performance overview with FFN's CAGR
perf = performance_overview(returns, prices=False, short=True)
```

## Expected Behavior

### Date Range Alignment

When comparing CAGR calculations:

```python
# These may differ slightly:
cagr_from_prices = cagr(prices)  # Uses full date range of prices
cagr_from_rets = cagr_from_returns(prices.pct_change())  # Missing first date

# This is EXPECTED because pct_change() drops the first row
```

**Why?** Each function calculates CAGR for the actual date range of its input data. If you pass returns that start on day 2, the CAGR is calculated from day 2 to the end.

### Comparison with QuantStats

```python
import quantstats_lumi as qs
from myPortfolioManagement.myPerformanceMetrics import cagr_from_returns

# These will differ (typically by 0.1-0.5%)
cagr_qs = qs.stats.cagr(returns, periods=365)  # Period counting method
cagr_std = cagr_from_returns(returns)  # Year_frac method (standardized)
```

The standardized method (year_frac) is more accurate for financial reporting.

## Migration Guide

### If you were using `qs.stats.cagr` directly:

**Before:**
```python
import quantstats_lumi as qs
cagr = qs.stats.cagr(returns, periods=365)
```

**After:**
```python
from myPortfolioManagement.myPerformanceMetrics import cagr_from_returns
cagr = cagr_from_returns(returns)  # More accurate
```

### If you were using `get_main_stats`:

No code changes needed! The function now automatically uses standardized CAGR:

```python
from myPortfolioManagement.myPerformanceMetrics import get_main_stats

stats = get_main_stats(returns, rf=0.04)
# stats['cagr'] now uses the standardized method
```

## Testing

Comprehensive unit tests verify:

1. ✓ CAGR consistency across functions
2. ✓ Correct use of year_frac
3. ✓ DataFrame and Series inputs work correctly
4. ✓ Backward compatibility maintained
5. ✓ Expected differences from QuantStats

## Technical Details

### year_frac Calculation

FFN's `year_frac` uses the Actual/Actual day count convention:

```
year_frac = (end_date - start_date).days / 365.25
```

This accounts for leap years by using 365.25 as the average days per year.

### CAGR Formula

The standardized CAGR formula:

```
CAGR = (End_Value / Start_Value) ^ (1 / year_frac) - 1
```

Where:
- `End_Value` = Final price (or cumulative return product)
- `Start_Value` = Initial price (or 1.0 for returns)
- `year_frac` = Actual years between start and end dates

## Benefits

1. **Consistency**: All CAGR calculations use the same methodology
2. **Accuracy**: More accurate reflection of calendar-time returns
3. **Simplicity**: Single source of truth for performance calculations
4. **Compatibility**: Aligns with financial industry standards
5. **Maintainability**: Easier to maintain and debug performance metrics

## References

- FFN library: https://github.com/pmorissette/ffn
- QuantStats library: https://github.com/ranaroussi/quantstats
- Day Count Conventions: https://en.wikipedia.org/wiki/Day_count_convention
