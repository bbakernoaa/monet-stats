---
type: explanation
category: explanation
tags: [architecture, integration, ci-cd]
---

# Monet Stats System Integration Documentation

## Overview

This document provides a comprehensive overview of the Monet Stats system integration, detailing the architecture, components, interfaces, and validation status for the Earth system modeling statistics package.

## System Architecture

### Package Structure

```
monet-stats/
├── src/monet_stats/              # Core statistical modules
│   ├── __init__.py               # Package initialization and API exports
│   ├── accessor.py               # xarray DataArray/Dataset accessors
│   ├── analysis.py               # Climatology, KZ filter, MDA8, FFT, anomalies
│   ├── benchmarks.py             # Performance and accuracy benchmarking
│   ├── contingency_metrics.py    # Categorical / 2x2 table metrics
│   ├── correlation_metrics.py    # Correlation and skill relationships
│   ├── data_processing.py        # Alignment, normalization, missing values
│   ├── distribution_metrics.py   # PDF/distribution comparison
│   ├── efficiency_metrics.py     # NSE family and percent-correct
│   ├── error_metrics.py          # Error and bias measures
│   ├── interfaces.py             # Abstract base classes
│   ├── performance.py            # Chunking and lazy-evaluation helpers
│   ├── plugin_system.py          # Custom metric plugin registry
│   ├── relative_metrics.py       # Normalized/relative error measures
│   ├── spatial_ensemble_metrics.py # EDS, CRPS, SAL, BSS, rank histograms
│   ├── spatial_skill_metrics.py  # FSS and VETS
│   ├── temporal_metrics.py       # DTW, cross-wavelet, phase error
│   ├── track_metrics.py          # Tropical-cyclone track statistics
│   ├── uncertainty.py            # Block bootstrapping
│   └── utils_stats.py            # Shared statistical utilities
├── tests/                        # Test suite
├── docs/                         # MkDocs documentation
├── data/                         # Example-dataset generation script
├── scripts/                      # Demo/utility scripts
├── .github/workflows/            # CI/CD pipeline configuration
├── mkdocs.yml                    # Documentation build configuration
├── codecov.yml                   # Coverage reporting configuration
├── pyproject.toml                # Project configuration and dependencies
└── README.md                     # Project overview
```

### Component Dependencies

#### Core Dependencies

- **numpy**: Numerical computing foundation
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing and statistical functions
- **xarray**: Multi-dimensional array data structures (required)

Optional backends are provided as extras: **dask** (`dask`, `distributed`) and
**cubed** (`cubed`, `cubed-xarray`).

#### Development Dependencies

- **pre-commit**: Git hooks
- **ruff**: Linting and formatting (replaces flake8, isort, black)
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance benchmarking
- **hypothesis**: Property-based testing (in the `test` extra)

### Statistical Modules Integration

Coverage figures below are from a full `pytest --cov=src/monet_stats` run on
this branch (898 tests passing, 90% total).

#### 1. Error Metrics Module (`error_metrics.py`)

- **Purpose**: Measures of error and deviation between model and observations
- **Key Functions**: MAE, RMSE, MB, MSE, MAPE, MASE, NRMSE, STDO, STDP, MedAE, IOA, FAC2, RMSLE, COE, etc.
- **Integration Points**: Used by all other modules for error analysis
- **Test Coverage**: 94%

#### 2. Correlation Metrics Module (`correlation_metrics.py`)

- **Purpose**: Statistical relationship measures
- **Key Functions**: R2, pearsonr, spearmanr, kendalltau, KGE, CCC, AC, taylor_skill
- **Integration Points**: Cross-references with error metrics
- **Test Coverage**: 91%

#### 3. Efficiency Metrics Module (`efficiency_metrics.py`)

- **Purpose**: Model performance and efficiency measures
- **Key Functions**: NSE, NSEm, NSElog, rNSE, mNSE, PC (MAPE/MASE/KGE are re-exported)
- **Integration Points**: Uses error and correlation metrics
- **Test Coverage**: 84%

#### 4. Relative Metrics Module (`relative_metrics.py`)

- **Purpose**: Normalized difference measures
- **Key Functions**: NMB, NMB_ABS, NMdnB, FB, FE, MG, VG, MPE (plus weighted/median variants)
- **Integration Points**: Complementary to error metrics
- **Test Coverage**: 98%

#### 5. Contingency Metrics Module (`contingency_metrics.py`)

- **Purpose**: Categorical verification and contingency analysis
- **Key Functions**: POD, FAR, CSI, HSS, ETS, TSS, FBI, BSS_binary, BS, scores
- **Integration Points**: Specialized for categorical data
- **Test Coverage**: 94%

#### 6. Spatial & Ensemble Metrics Module (`spatial_ensemble_metrics.py`)

- **Purpose**: Ensemble verification and probabilistic analysis
- **Key Functions**: EDS, CRPS, SAL, BSS, spread_error, rank_histogram, reliability_diagram
- **Integration Points**: Uses spatial data structures
- **Test Coverage**: 96%

#### 7. Spatial Skill Metrics Module (`spatial_skill_metrics.py`)

- **Purpose**: Scale-dependent spatial skill scores
- **Key Functions**: FSS, VETS
- **Test Coverage**: 98%

#### 8. Track Metrics Module (`track_metrics.py`)

- **Purpose**: Tropical-cyclone / storm-track verification
- **Key Functions**: track_error, along_track_error, cross_track_error, haversine_distance, bearing, translation_speed, find_storm_center(s)
- **Test Coverage**: 82%

#### 9. Distribution Metrics Module (`distribution_metrics.py`)

- **Purpose**: PDF and distribution comparison
- **Key Functions**: WassersteinDistance, KLDivergence, JensenShannonDivergence, EnergyDistance, SinkhornDistance, MutualInformation
- **Test Coverage**: 91%

#### 10. Temporal Metrics Module (`temporal_metrics.py`)

- **Purpose**: Time-series alignment and phase analysis
- **Key Functions**: DynamicTimeWarping, CrossWaveletTransform, PhaseError
- **Test Coverage**: 94%

#### 11. Analysis Module (`analysis.py`)

- **Purpose**: Resampling, climatology, filtering, spectral analysis
- **Key Functions**: climatology, kz_filter, mda8, diurnal_cycle, fft_analysis, power_spectrum, anomalies, detrend
- **Test Coverage**: 91%

#### 12. Uncertainty Module (`uncertainty.py`)

- **Purpose**: Confidence intervals via resampling
- **Key Functions**: block_bootstrap
- **Test Coverage**: 97%

#### 13. Performance / Data Processing / Plugin Modules

- `performance.py` (65%): chunking, lazy thresholds, parallel compute
- `data_processing.py` (100%): alignment, normalization, missing-value handling
- `plugin_system.py` (96%): custom metric registry
- `interfaces.py` (96%): abstract base classes
- `accessor.py` (76%): the `.monet_stats` xarray accessors
- `benchmarks.py` (81%): benchmark harness

#### 14. Utility Functions Module (`utils_stats.py`)

- **Purpose**: Common statistical utilities and helper functions
- **Key Functions**: matchedcompressed, matchmasks, circlebias, angular_difference, rmse, mae, correlation
- **Integration Points**: Used across all metric modules
- **Test Coverage**: 83%

## Interface Compatibility Analysis

### Cross-Module Compatibility

#### ✅ Compatible Interfaces

- **Error ↔ Correlation**: All functions can operate on the same input data types
- **Error ↔ Efficiency**: Error metrics provide foundation for efficiency calculations
- **Utility Functions**: Consistent interface across modules for data processing
- **Type Consistency**: All functions return consistent numeric types

#### ⚠️ Interface Issues Identified

1. **Parameter Inconsistencies**: Some functions require different parameter sets
2. **Return Type Variations**: Functions may return tuples vs. single values
3. **Missing Default Parameters**: Several functions lack sensible defaults
4. **Xarray Integration**: Partial support for xarray DataArray objects

#### Issues to Watch

1. **Coverage**: 90% overall — above the 60% CI gate, but `performance.py`
   (65%) and `accessor.py` (76%) are the weakest modules
2. **Import/Export Gaps**: Many functions are importable from their submodule
   but not re-exported at the top level (e.g. `NME`, `sMAPE`, `NSC`, `MP`,
   `NO`, `RM`, `RMSPE`, `MASEm` live in `relative_metrics`/`error_metrics`
   but are absent from `monet_stats.__all__`)
3. **Return Type Variations**: Functions may return tuples vs. single values
   (e.g. `scores` returns a 4-tuple, `HSS_max_threshold` returns `(best_hss,
   best_threshold)`)
4. **Edge Cases**: Degenerate inputs can produce NaN (e.g. `R2` on constant
   observation arrays)

### Data Flow Architecture

```
Input Data (numpy arrays, xarray DataArrays)
    ↓
Utility Functions (data cleaning, mask handling)
    ↓
Error Metrics → Correlation Metrics → Efficiency Metrics
    ↓
Relative Metrics ← Contingency Metrics ← Spatial Metrics
    ↓
Integrated Analysis & Reporting
```

## CI/CD Pipeline Integration

### GitHub Actions Configuration

#### Test Matrix

- **Python Versions**: 3.11, 3.12
- **Test Coverage**: Minimum 60% required (`--cov-fail-under=60`)
- **Code Quality** (via pre-commit):
  - Ruff formatting (`ruff-format`)
  - Ruff linting (`ruff-check`, replaces flake8/isort/black)
  - YAML validation and file hygiene hooks

#### Pipeline Stages

1. **Testing**: Unit tests, integration tests, coverage validation
2. **Build Package**: Create distribution artifacts
3. **Documentation**: Build and deploy documentation
4. **Artifact Upload**: Upload test results, coverage reports, and builds

### Quality Gates

#### Coverage Requirements

- **Overall Coverage**: 60% minimum CI gate (currently 90%)
- **Weakest Modules**: `performance.py` (65%) and `accessor.py` (76%)
- **Test Types**: Unit, integration, performance, and edge case tests

#### Code Quality Standards

- **Line Length**: 120 characters maximum
- **Naming Conventions**: Consistent function and parameter naming
- **Documentation**: Comprehensive docstrings with examples
- **Type Hints**: Full type annotation coverage

## Test Integration Status

### Test Coverage Analysis

| Module                      | Coverage | Status          | Priority |
| --------------------------- | -------- | --------------- | -------- |
| data_processing.py          | 100%     | ✅ Complete     | Low      |
| relative_metrics.py         | 98%      | ✅ Excellent    | Low      |
| spatial_skill_metrics.py    | 98%      | ✅ Excellent    | Low      |
| uncertainty.py              | 97%      | ✅ Excellent    | Low      |
| interfaces.py               | 96%      | ✅ Excellent    | Low      |
| plugin_system.py            | 96%      | ✅ Excellent    | Low      |
| spatial_ensemble_metrics.py | 96%      | ✅ Excellent    | Low      |
| contingency_metrics.py      | 94%      | ✅ Excellent    | Low      |
| error_metrics.py            | 94%      | ✅ Excellent    | Low      |
| temporal_metrics.py         | 94%      | ✅ Excellent    | Low      |
| analysis.py                 | 91%      | ✅ Excellent    | Low      |
| correlation_metrics.py      | 91%      | ✅ Excellent    | Low      |
| distribution_metrics.py     | 91%      | ✅ Excellent    | Low      |
| benchmarks.py               | 81%      | ✅ Good         | Low      |
| track_metrics.py            | 82%      | ✅ Good         | Low      |
| utils_stats.py              | 83%      | ✅ Good         | Low      |
| efficiency_metrics.py       | 84%      | ✅ Good         | Low      |
| __init__.py                 | 77%      | ✅ Good         | Low      |
| accessor.py                 | 76%      | ⚠️ Moderate     | Medium   |
| performance.py              | 65%      | ⚠️ Needs Work   | Medium   |
| **Total**                   | **90%**  | ✅ Above Gate   | —        |

### Integration Test Suite

#### Test Categories

1. **Module Compatibility**: Cross-module function interactions
2. **Data Type Support**: NumPy arrays, xarray DataArrays
3. **Edge Cases**: Zero arrays, constant values, NaN handling
4. **Performance**: Large dataset processing
5. **API Completeness**: All exported functions accessible

#### Test Results

- **Test Suite**: 898 passed (full `pytest` run on this branch)
- **Coverage**: 90% total, above the 60% CI gate

## Known Issues and Limitations

### 1. Test Coverage Gaps

- **Performance**: 35% of `performance.py` untested
- **Accessor**: 24% of `accessor.py` untested (mostly the lazy/dask branches)

### 2. Mathematical Edge Cases

- **R2 Calculation**: Returns NaN on constant observation arrays (zero variance)
- **Degenerate Categorical Inputs**: Scores like HSS/ETS can be NaN when the
  contingency table is fully degenerate

### 3. Interface Issues

- **Parameter Variability**: Inconsistent parameter naming and defaults
- **Return Type Ambiguity**: Some functions return tuples, others single values
- **Top-Level Exports**: `R2`/`RMSE`/`MAE`/`IOA` etc. work on xarray DataArray
  inputs, but many submodule functions must be imported from their module
  rather than the top-level package

### 4. Performance Considerations

- **Large Arrays**: Some functions may be slow on very large datasets
- **Memory Usage**: Potential memory overhead with masked arrays
- **Vectorization**: Some operations could be more efficient

## Production Readiness Assessment

### ✅ Ready Components

- **Error / Correlation / Contingency / Relative Metrics**: 91–98% coverage,
  stable interfaces
- **Spatial & Distribution / Temporal Metrics**: 91–98% coverage
- **Plugin System**: `plugin_system.py` provides a working custom-metric
  registry (96% coverage)
- **Package Structure**: Clean, modular design with 20 submodules
- **CI/CD Pipeline**: Test matrix, coverage gate, docs build, and Codecov

### ⚠️ Watch Components

- **Accessor**: 76% coverage — the lazy/dask code paths are the least tested
- **Performance Utilities**: 65% coverage in `performance.py`
- **Track Metrics**: 82% coverage; newer module with less operational use

### Documented Gaps

- **Top-Level Exports**: Not every submodule function is re-exported in
  `monet_stats.__all__`; import from the submodule when a name is missing
- **Type Hints**: Annotations are present on most public functions but are not
  enforced by a type checker in CI

## Recommendations for Production Release

### Immediate Actions

1. **Raise Coverage on Weakest Modules**: `performance.py` (65%) and
   `accessor.py` (76%)
2. **Close Export Gaps**: promote frequently used names (`NME`, `sMAPE`, `NSC`,
   `MP`, `NO`, `RM`, `RMSPE`) to the top-level namespace or document the
   submodule import path
3. **Standardize Interfaces**: consistent parameter naming and return types

### Medium Priority Actions

1. **Performance Benchmarking**: extend `benchmarks.py` coverage (81%)
2. **Enhanced Documentation**: add more worked examples and use cases
3. **Edge Case Testing**: document NaN behavior for degenerate inputs

### Long-term Improvements

1. **Type Checking**: consider adding a type checker to CI
2. **Lazy Backends**: broaden Cubed coverage alongside Dask

## Conclusion

The monet-stats package is in a healthy state: 898 tests pass and overall
coverage is 90%, comfortably above the 60% CI gate. The remaining work is
concentrated in `performance.py` and `accessor.py`, plus closing the gap
between submodule definitions and top-level re-exports. The modular design,
lazy-evaluation support, and plugin architecture provide a strong foundation
for future enhancements.

---

_Integration Status: Production Ready with Caveats_
_Last Updated: 2026-10-15_
