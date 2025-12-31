# Testing Guide for myPortfolioManagement

This document provides comprehensive guidance for testing the myPortfolioManagement library.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [CI/CD Pipelines](#cicd-pipelines)
- [Writing Tests](#writing-tests)
- [Coverage](#coverage)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Install test dependencies
pip install -r requirements-test.txt
pip install -e .

# 2. Run all tests
pytest

# 3. Run fast tests only (no API calls, no GPU, no slow tests)
pytest -m "unit and not slow and not data_fetch and not gpu"

# 4. Run with coverage
pytest --cov=myPortfolioManagement --cov-report=html

# 5. Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## Test Organization

```
tests/
├── __init__.py
├── conftest.py                           # Shared fixtures
├── test_myData.py                        # Data fetching tests (~45 tests)
├── test_myReturns.py                     # Returns calculation tests (~50 tests)
├── test_myPerformance_and_Optimization.py # Metrics & optimization (~40 tests)
└── test_myBootstrap_GPU.py               # GPU bootstrap tests (~30 tests)
```

### Test Modules

| Module | Description | Test Count |
|--------|-------------|------------|
| `test_myData.py` | Data fetching, validation, FX adjustment | ~45 |
| `test_myReturns.py` | Simple/log returns, frequency conversion, portfolios | ~50 |
| `test_myPerformance_and_Optimization.py` | Sharpe, VaR, drawdown, HRP, optimization | ~40 |
| `test_myBootstrap_GPU.py` | GPU bootstrap, IID, block, consistency | ~30 |

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_myReturns.py

# Run specific test class
pytest tests/test_myReturns.py::TestCalculateReturns

# Run specific test function
pytest tests/test_myReturns.py::TestCalculateReturns::test_simple_returns_basic

# Run tests matching a pattern
pytest -k "sharpe"

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto           # Use all available cores
pytest -n 4              # Use 4 cores
```

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=myPortfolioManagement

# Generate HTML report
pytest --cov=myPortfolioManagement --cov-report=html

# Generate XML report (for CI)
pytest --cov=myPortfolioManagement --cov-report=xml

# Fail if coverage below threshold
pytest --cov=myPortfolioManagement --cov-fail-under=70
```

## Test Markers

Tests are organized using pytest markers for selective execution:

| Marker | Description | Example |
|--------|-------------|---------|
| `@pytest.mark.unit` | Standard unit test | `pytest -m unit` |
| `@pytest.mark.integration` | Multi-module tests | `pytest -m integration` |
| `@pytest.mark.gpu` | Requires GPU (CuPy) | `pytest -m gpu` |
| `@pytest.mark.slow` | Long-running (>5s) | `pytest -m slow` |
| `@pytest.mark.data_fetch` | Needs internet/API | `pytest -m data_fetch` |

### Marker Combinations

```bash
# Fast tests (no slow, no GPU, no API)
pytest -m "unit and not slow and not data_fetch and not gpu"

# All non-GPU tests
pytest -m "not gpu"

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## CI/CD Pipelines

### Pull Request Workflow (`pr-tests.yml`)

Runs on every PR to main/master branches:
- **Duration**: 5-10 minutes
- **Jobs**:
  - Lint & syntax check (Black, isort, flake8)
  - Fast unit tests
  - Security scan (Bandit)

### Main Branch Workflow (`main-tests.yml`)

Runs on push to main/master:
- **Duration**: 20-40 minutes
- **Jobs**:
  - Code quality checks
  - Unit tests (Python 3.10, 3.11, 3.12)
  - Slow tests
  - Data fetching tests
  - Integration tests
  - Coverage report

### Manual Trigger

```bash
# Trigger main workflow manually from GitHub Actions tab
# Or via GitHub CLI:
gh workflow run main-tests.yml
```

## Writing Tests

### Test Structure

```python
import pytest
import pandas as pd
import numpy as np

@pytest.mark.unit
class TestMyFeature:
    """Tests for MyFeature functionality."""

    def test_basic_functionality(self, sample_returns):
        """Test basic feature behavior."""
        result = my_function(sample_returns)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        with pytest.raises(ValueError):
            my_function(pd.DataFrame())

    @pytest.mark.parametrize("param", [1, 2, 3])
    def test_with_parameters(self, param):
        """Test with different parameters."""
        result = my_function(param)
        assert result > 0
```

### Using Fixtures

Available fixtures in `conftest.py`:

```python
# Price data fixtures
sample_prices          # 5 assets, ~4 years of daily data
single_asset_prices    # Single asset prices
prices_with_nans       # Prices with NaN values
constant_prices        # Zero volatility prices

# Returns fixtures
sample_returns         # Simple returns from sample_prices
sample_log_returns     # Log returns
short_returns          # Short series for quick tests
correlated_returns     # Known correlation structure

# Portfolio fixtures
equal_weights          # Equal weights for 5 assets
concentrated_weights   # Non-equal weights
multiple_portfolios_weights  # Multiple portfolio weights

# Benchmark fixtures
market_returns         # S&P 500-like returns
bond_returns           # Bond benchmark returns

# Option fixtures
option_params          # Standard BS parameters
itm_call_params        # In-the-money call
otm_call_params        # Out-of-the-money call

# GPU fixtures
gpu_available          # Boolean: GPU available
large_returns_for_gpu  # Large dataset for GPU testing
bootstrap_params       # Bootstrap parameters

# Edge case fixtures
empty_dataframe        # Empty DataFrame
single_row_returns     # Single row of data
all_nan_returns        # All NaN returns
```

### Test Best Practices

1. **Isolation**: Tests should not depend on each other
2. **Repeatability**: Use fixed random seeds
3. **Speed**: Unit tests should run in <1 second
4. **Clarity**: Clear docstrings explaining what's tested
5. **Coverage**: Test both happy paths and edge cases

## Coverage

### Current Coverage Targets

| Module | Target | Critical Paths |
|--------|--------|----------------|
| Overall | 70% | - |
| myData | 60% | Data fetching |
| myReturns | 80% | Return calculations |
| myPerformanceMetrics | 75% | Key metrics |
| myPortfolioOptimisation | 70% | HRP, optimization |
| myBootstrapping | 75% | Bootstrap methods |

### Checking Coverage

```bash
# Quick coverage check
pytest --cov=myPortfolioManagement --cov-report=term-missing

# Generate detailed HTML report
pytest --cov=myPortfolioManagement --cov-report=html
open htmlcov/index.html

# Check specific module coverage
pytest --cov=myPortfolioManagement.myReturns tests/test_myReturns.py
```

## Pre-commit Hooks

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Hooks Included

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **bandit**: Security scanning
- **Trailing whitespace**: Remove trailing spaces
- **End of file fixer**: Ensure newline at EOF
- **YAML/JSON validation**: Config file validation
- **Large file check**: Prevent large file commits
- **Debug statement check**: Remove debug code
- **Private key detection**: Security check

### Skipping Hooks

```bash
# Skip all hooks for a commit
git commit --no-verify -m "Emergency fix"

# Skip specific hook
SKIP=black git commit -m "Skip formatting"
```

## Troubleshooting

### Common Issues

#### Tests fail with import errors

```bash
# Ensure package is installed in editable mode
pip install -e .
```

#### GPU tests fail

```bash
# Skip GPU tests if no GPU available
pytest -m "not gpu"

# Check GPU availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

#### Data fetch tests timeout

```bash
# Skip data fetching tests
pytest -m "not data_fetch"

# Increase timeout
pytest --timeout=300
```

#### Slow test suite

```bash
# Run only fast tests
pytest -m "unit and not slow"

# Use parallel execution
pytest -n auto
```

#### Coverage below threshold

```bash
# Identify untested code
pytest --cov=myPortfolioManagement --cov-report=term-missing

# Focus on specific module
pytest --cov=myPortfolioManagement.myReturns tests/test_myReturns.py
```

### Getting Help

```bash
# Show available markers
pytest --markers

# Show fixtures
pytest --fixtures

# Verbose collection (see which tests will run)
pytest --collect-only

# Debug test discovery
pytest -v --collect-only tests/
```

## Performance Benchmarking

```bash
# Run benchmarks (requires pytest-benchmark)
pytest --benchmark-only

# Compare with baseline
pytest --benchmark-compare

# Save benchmark results
pytest --benchmark-save=baseline
```

## Contributing Tests

When adding new tests:

1. Place in appropriate test file
2. Use descriptive test names (`test_<what>_<expected_behavior>`)
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Include docstring explaining the test
5. Use fixtures for test data
6. Test both success and failure cases
7. Run full test suite before submitting PR

```bash
# Before submitting PR
pytest -m "not gpu"  # Run non-GPU tests
pre-commit run --all-files  # Run all hooks
```
