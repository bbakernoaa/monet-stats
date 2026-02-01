"""
Tests specifically designed to boost coverage for the Aero Protocol refactor.
Targets numpy fallbacks with axis, missing dependency handling, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import monet_stats.analysis as analysis
import monet_stats.interfaces as interfaces
import monet_stats.performance as perf
import monet_stats.utils_stats as utils
from monet_stats import stats
from monet_stats.contingency_metrics import (
    CSI,
    ETS,
    FAR,
    FBI,
    HSS,
    POD,
    TSS,
    BSS_binary,
    ETS_max_threshold,
    FAR_min_threshold,
    HSS_max_threshold,
    POD_max_threshold,
)
from monet_stats.correlation_metrics import (
    AC,
    CCC,
    E1,
    KGE,
    R2,
    WDAC,
    E1_prime,
    IOA_m,
    IOA_prime,
    RMSEs,
    RMSEu,
    d1,
    kendalltau,
    pearsonr,
    spearmanr,
    taylor_skill,
)
from monet_stats.efficiency_metrics import MSE, NSE, PC, NSElog, mNSE, rNSE
from monet_stats.error_metrics import (
    COE,
    CORR_INDEX,
    CRMSE,
    IOA,
    LOG_ERROR,
    MAE,
    MAPE,
    MASE,
    MB,
    MNB,
    MNE,
    MO,
    MP,
    NMSE,
    NO,
    NOP,
    NP,
    NRMSE,
    NSC,
    RM,
    RMSPE,
    VOLUMETRIC_ERROR,
    WDMB,
    MAE_m,
    MAE_norm,
    MAPE_mod,
    MASE_mod,
    MASEm,
    MdnB,
    MdnNB,
    MdnNE,
    MdnO,
    MdnP,
    MedAE,
    MedAE_m,
    NMdnGE,
    NSE_alpha,
    NSE_beta,
    RMdn,
    RMSE_m,
    RMSE_norm,
    WDMB_m,
    WDMdnB,
    bias_fraction,
    sMAPE,
)
from monet_stats.error_metrics import (
    IOA_m as IOA_m_err,
)
from monet_stats.relative_metrics import (
    FB,
    FE,
    ME,
    MNPB,
    MNPE,
    MPE,
    NMB,
    NMB_ABS,
    NME,
    NMPB,
    NMPE,
    PSUTMNPB,
    PSUTMNPE,
    PSUTNMPB,
    PSUTNMPE,
    USUTPB,
    USUTPE,
    WDME,
    MdnE,
    MdnNPB,
    MdnNPE,
    MdnPE,
    NMdnB,
    NMdnE,
    NMdnPB,
    NMdnPE,
    NME_m,
    NME_m_ABS,
    PSUTMdnNPB,
    PSUTMdnNPE,
    PSUTNMdnPB,
    PSUTNMdnPE,
    WDMdnE,
    WDME_m,
    WDNMB_m,
)
from monet_stats.spatial_ensemble_metrics import (
    BSS,
    CRPS,
    EDS,
    SAL,
    ensemble_mean,
    ensemble_std,
    rank_histogram,
    spread_error,
)
from monet_stats.spatial_skill_metrics import FSS, VETS

try:
    import dask.array as da  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    lon = np.linspace(-130, -60, 5)
    lat = np.linspace(20, 50, 5)
    data = np.random.rand(5, 5)
    da = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test_data",
        attrs={"units": "test_units", "history": "initial history"},
    )
    return da


@pytest.fixture
def sample_pair_da():
    """Create a pair of aligned DataArrays."""
    lon = np.linspace(-130, -60, 5)
    lat = np.linspace(20, 50, 5)
    obs_data = np.random.rand(5, 5) + 1.0
    mod_data = np.random.rand(5, 5) + 1.0
    obs = xr.DataArray(obs_data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="Obs")
    mod = xr.DataArray(mod_data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="Mod")
    return obs, mod


def test_correlation_numpy_axis():
    """Boost coverage: Target numpy fallback with axis in correlation_metrics."""
    obs = np.random.rand(10, 5)
    mod = np.random.rand(10, 5)

    # R2
    assert R2(obs, mod, axis=0).shape == (5,)

    # RMSEs, RMSEu
    assert RMSEs(obs, mod, axis=0).shape == (5,)
    assert RMSEu(obs, mod, axis=0).shape == (5,)

    # d1, E1, IOA_m, IOA
    assert d1(obs, mod, axis=1).shape == (10,)
    assert E1(obs, mod, axis=0).shape == (5,)
    assert IOA_m(obs, mod, axis=1).shape == (10,)

    # AC, WDAC
    assert AC(obs, mod, axis=0).shape == (5,)
    assert WDAC(obs, mod, axis=1).shape == (10,)

    # taylor_skill, KGE
    assert taylor_skill(obs, mod, axis=0).shape == (5,)
    assert KGE(obs, mod, axis=1).shape == (10,)

    # pearsonr, spearmanr, kendalltau
    assert pearsonr(obs, mod, axis=0).shape == (5,)
    assert spearmanr(obs, mod, axis=1).shape == (10,)
    assert kendalltau(obs, mod, axis=0).shape == (5,)

    # CCC, E1_prime, IOA_prime
    assert CCC(obs, mod, axis=1).shape == (10,)
    assert E1_prime(obs, mod, axis=0).shape == (5,)
    assert IOA_prime(obs, mod, axis=1).shape == (10,)


def test_error_numpy_axis():
    """Boost coverage: Target numpy fallback with axis in error_metrics."""
    obs = np.random.rand(10, 5) + 1.0
    mod = np.random.rand(10, 5) + 1.0

    metrics = [
        MNB,
        MNE,
        MdnNB,
        MdnNE,
        NMdnGE,
        NO,
        NOP,
        NP,
        MO,
        MP,
        MdnO,
        MdnP,
        RM,
        RMdn,
        MB,
        MdnB,
        WDMB_m,
        WDMB,
        WDMdnB,
        MAE,
        MedAE,
        CRMSE,
        MAPE,
        sMAPE,
        NRMSE,
        RMSPE,
        NSC,
        NSE_alpha,
        NSE_beta,
        MAE_m,
        MedAE_m,
        RMSE_m,
        IOA_m_err,
        MAPE_mod,
        MASE_mod,
        RMSE_norm,
        MAE_norm,
        bias_fraction,
        NMSE,
        LOG_ERROR,
        COE,
        VOLUMETRIC_ERROR,
    ]

    for func in metrics:
        res = func(obs, mod, axis=0)
        assert np.shape(res) == (5,)


def test_error_MASE_numpy():
    """Boost coverage: MASE and MASE_mod numpy paths."""
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    assert MASE(obs, mod) is not None
    assert MASE_mod(obs, mod) is not None

    obs2 = np.random.rand(10, 5)
    mod2 = np.random.rand(10, 5)
    assert MASE(obs2, mod2, axis=0).shape == (5,)
    assert MASEm(obs2, mod2, axis=1).shape == (10,)


def test_efficiency_numpy_axis():
    """Boost coverage: Target numpy fallback with axis in efficiency_metrics."""
    obs = np.random.rand(10, 5) + 1.0
    mod = np.random.rand(10, 5) + 1.0

    assert NSE(obs, mod, axis=0).shape == (5,)
    assert NSElog(obs, mod, axis=1).shape == (10,)
    assert rNSE(obs, mod, axis=0).shape == (5,)
    assert mNSE(obs, mod, axis=1).shape == (10,)
    assert PC(obs, mod, axis=0).shape == (5,)
    assert MSE(obs, mod, axis=1).shape == (10,)


def test_contingency_numpy_axis():
    """Boost coverage: Target numpy fallback with axis in contingency_metrics."""
    obs = np.random.rand(10, 5)
    mod = np.random.rand(10, 5)

    assert HSS(obs, mod, 0.5, axis=0).shape == (5,)
    assert ETS(obs, mod, 0.5, axis=1).shape == (10,)
    assert CSI(obs, mod, 0.5, axis=0).shape == (5,)
    assert POD(obs, mod, 0.5, axis=1).shape == (10,)
    assert FAR(obs, mod, 0.5, axis=0).shape == (5,)
    assert FBI(obs, mod, 0.5, axis=1).shape == (10,)
    assert TSS(obs, mod, 0.5, axis=0).shape == (5,)
    assert BSS_binary(obs, mod, 0.5, axis=1).shape == (10,)


def test_hss_max_threshold_xarray(sample_da):
    """Boost coverage: HSS_max_threshold with xarray."""
    obs = sample_da
    mod = sample_da + 0.1
    thresh, val = HSS_max_threshold(obs, mod, 0.1, 0.5, 0.1)
    assert isinstance(thresh, float)
    assert isinstance(val, float)


def test_all_metrics_xarray_axis(sample_pair_da):
    """Boost coverage: Test all metrics with xarray and axis (dim name)."""
    obs, mod = sample_pair_da

    # Correlation metrics
    corr_metrics = [
        R2,
        d1,
        E1,
        IOA_m,
        AC,
        WDAC,
        taylor_skill,
        KGE,
        pearsonr,
        spearmanr,
        kendalltau,
        CCC,
        E1_prime,
        IOA_prime,
    ]
    for func in corr_metrics:
        res = func(obs, mod, axis="lat")
        assert "lon" in res.coords
        assert res.dims == ("lon",)

    # Error metrics
    err_metrics = [
        MNB,
        MNE,
        MdnNB,
        MdnNE,
        NMdnGE,
        NO,
        NOP,
        NP,
        MO,
        MP,
        MdnO,
        MdnP,
        RM,
        RMdn,
        MB,
        MdnB,
        WDMB_m,
        WDMB,
        WDMdnB,
        MAE,
        MedAE,
        CRMSE,
        MAPE,
        sMAPE,
        NRMSE,
        RMSPE,
        NSC,
        NSE_alpha,
        NSE_beta,
        MAE_m,
        MedAE_m,
        RMSE_m,
        IOA,
        IOA_m_err,
        MAPE_mod,
        MASE_mod,
        RMSE_norm,
        MAE_norm,
        bias_fraction,
        NMSE,
        LOG_ERROR,
        COE,
        VOLUMETRIC_ERROR,
        CORR_INDEX,
    ]
    for func in err_metrics:
        res = func(obs, mod, axis="lat")
        if isinstance(res, xr.DataArray):
            assert "lon" in res.coords
            assert res.dims == ("lon",)

    # Efficiency metrics
    eff_metrics = [NSE, NSElog, rNSE, mNSE, PC, MSE]
    for func in eff_metrics:
        res = func(obs, mod, axis="lat")
        assert "lon" in res.coords
        assert res.dims == ("lon",)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_all_metrics_dask(sample_pair_da):
    """Boost coverage: Test all metrics with Dask-backed DataArrays."""
    obs, mod = sample_pair_da
    obs_d = obs.chunk({"lat": 2, "lon": 2})
    mod_d = mod.chunk({"lat": 2, "lon": 2})

    # Selected heavy metrics
    metrics = [R2, KGE, pearsonr, spearmanr, kendalltau, CRMSE, MASE, NSE]

    for func in metrics:
        res = func(obs_d, mod_d, axis="lat")
        assert hasattr(res.data, "dask")
        # Ensure it computes correctly
        res_c = res.compute()
        assert not hasattr(res_c.data, "dask")


def test_max_threshold_others(sample_pair_da):
    """Boost coverage: Other max/min threshold functions."""
    obs, mod = sample_pair_da
    ETS_max_threshold(obs, mod, 1.0, 2.0, 0.5)
    POD_max_threshold(obs, mod, 1.0, 2.0, 0.5)
    FAR_min_threshold(obs, mod, 1.0, 2.0, 0.5)


def test_error_metrics_edge_cases():
    """Boost coverage: Edge cases like zero variance/range."""
    obs = np.ones(5)
    mod = np.ones(5)
    assert np.isclose(NMSE(obs, mod), 0.0)
    assert np.isclose(NRMSE(obs, mod), 0.0)
    assert np.isclose(NSE(obs, mod), 1.0)
    assert np.isclose(rNSE(obs, mod), 1.0)

    mod2 = np.ones(5) * 2.0
    # Mismatch with zero variance in obs
    assert NSE(obs, mod2) == -np.inf


def test_correlation_edge_cases():
    """Boost coverage: Edge cases for correlation."""
    obs = np.ones(5)
    mod = np.ones(5) * 2.0
    assert np.isclose(R2(obs, mod), 0.0)
    assert np.isclose(pearsonr(obs, mod), 0.0)

    obs_m = np.ma.masked_array([1, 2, 3], mask=[1, 0, 0])
    mod_m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
    # Single pair after matching
    assert np.isnan(spearmanr(obs_m, mod_m))


def test_relative_metrics_numpy_axis():
    """Boost coverage: Target numpy fallback with axis in relative_metrics."""
    obs = np.random.rand(10, 5) + 1.0
    mod = np.random.rand(10, 5) + 1.0

    metrics = [
        NMB,
        WDNMB_m,
        NMB_ABS,
        NMdnB,
        FB,
        ME,
        MdnE,
        WDME_m,
        WDME,
        WDMdnE,
        NME_m,
        NME_m_ABS,
        NME,
        NMdnE,
        FE,
        USUTPB,
        USUTPE,
        MPE,
        MdnPE,
    ]
    for func in metrics:
        res = func(obs, mod, axis=0)
        # Handle scalar vs array return
        if hasattr(res, "shape") and res.shape:
            assert res.shape == (5,)

    # Metrics with paxis
    peak_metrics = [MNPB, MdnNPB, MNPE, MdnNPE, NMPB, NMdnPB, NMPE, NMdnPE]
    for func in peak_metrics:
        res = func(obs, mod, paxis=0, axis=None)
        assert np.shape(res) == ()
        res_ax = func(obs, mod, paxis=1, axis=0)
        assert np.shape(res_ax) == ()

    # PSUT wrappers
    wrappers = [PSUTMNPB, PSUTMdnNPB, PSUTMNPE, PSUTMdnNPE, PSUTNMPB, PSUTNMPE, PSUTNMdnPB, PSUTNMdnPE]
    for func in wrappers:
        res = func(obs, mod)
        assert np.shape(res) == ()


def test_relative_metrics_xarray_axis(sample_pair_da):
    """Boost coverage: Test relative metrics with xarray and axis."""
    obs, mod = sample_pair_da
    metrics = [
        NMB,
        WDNMB_m,
        NMB_ABS,
        NMdnB,
        FB,
        ME,
        MdnE,
        WDME_m,
        WDME,
        WDMdnE,
        NME_m,
        NME_m_ABS,
        NME,
        NMdnE,
        FE,
        USUTPB,
        USUTPE,
        MPE,
        MdnPE,
        PSUTMNPB,
        PSUTMdnNPB,
        PSUTMNPE,
        PSUTMdnNPE,
        PSUTNMPB,
        PSUTNMPE,
        PSUTNMdnPB,
        PSUTNMdnPE,
    ]
    for func in metrics:
        func(obs, mod, axis="lat")

    peak_metrics = [MNPB, MdnNPB, MNPE, MdnNPE, NMPB, NMdnPB, NMPE, NMdnPE]
    for func in peak_metrics:
        func(obs, mod, paxis="lat", axis="lon")


def test_spatial_ensemble_metrics_numpy():
    """Boost coverage: spatial_ensemble_metrics with numpy."""
    obs = np.random.rand(5, 5)
    mod = np.random.rand(5, 5)
    ens = np.random.rand(10, 5, 5)

    assert np.shape(EDS(obs, mod, threshold=0.5)) == ()
    assert np.shape(CRPS(ens, obs, axis=0)) == (5, 5)
    s, e = spread_error(ens, obs, axis=0)
    assert np.shape(s) == ()
    assert np.shape(e) == ()
    assert np.shape(BSS(obs, mod, threshold=0.5)) == ()

    s, a, l_val = SAL(obs, mod, threshold=0.5)
    assert isinstance(s, float)

    assert np.shape(ensemble_mean(ens, axis=0)) == (5, 5)
    assert np.shape(ensemble_std(ens, axis=0)) == (5, 5)
    assert np.shape(rank_histogram(ens, obs, axis=0)) == (11,)


def test_spatial_ensemble_metrics_xarray(sample_pair_da):
    """Boost coverage: spatial_ensemble_metrics with xarray."""
    obs, mod = sample_pair_da
    # Create ensemble
    ens = xr.concat([obs + np.random.rand(5, 5) for _ in range(3)], dim="ensemble")

    assert isinstance(EDS(obs, mod, threshold=1.5), xr.DataArray)
    assert isinstance(CRPS(ens, obs, axis="ensemble"), xr.DataArray)
    s, e = spread_error(ens, obs, axis="ensemble")
    assert isinstance(s, xr.DataArray)
    assert isinstance(BSS(obs, mod, threshold=1.5), xr.DataArray)

    # SAL returns tuple of floats
    SAL(obs, mod, threshold=1.5)

    assert isinstance(ensemble_mean(ens, axis="ensemble"), xr.DataArray)
    assert isinstance(rank_histogram(ens, obs, axis="ensemble"), xr.DataArray)


def test_spatial_skill_metrics(sample_pair_da):
    """Boost coverage: spatial_skill_metrics."""
    obs, mod = sample_pair_da
    assert isinstance(FSS(obs, mod, threshold=1.5), xr.DataArray)
    assert isinstance(VETS(obs, mod, axis="lat"), xr.DataArray)

    # Numpy
    assert np.shape(FSS(obs.values, mod.values, threshold=1.5)) == ()
    assert np.shape(VETS(obs.values, mod.values, axis=0)) == (5,)


def test_stats_backend_agnostic(sample_pair_da):
    """Boost coverage: stats function in __init__.py."""
    obs, mod = sample_pair_da
    ds_in = xr.Dataset({"Obs": obs, "Mod": mod})
    # Xarray Dataset case
    res = stats(ds_in)
    assert isinstance(res, dict)
    assert "RMSE" in res
    assert "MAE" in res

    # Pandas DataFrame case
    df = pd.DataFrame({"Obs": np.random.rand(10), "Mod": np.random.rand(10)})
    res_df = stats(df)
    assert isinstance(res_df, dict)
    assert res_df["N"] == 10


def test_analysis_module(sample_da):
    """Boost coverage: analysis.py."""
    # kz filter handle numpy
    analysis.kz_filter(sample_da.values, m=3, k=2)

    # Temporal analysis needs time dimension
    times = pd.date_range("2023-01-01", periods=24, freq="h")
    da_time = xr.DataArray(np.random.rand(24), coords={"time": times}, dims="time")

    # Rolling mean
    rm8 = analysis.rolling_mean_8h(da_time)
    assert isinstance(rm8, xr.DataArray)
    rm24 = analysis.rolling_mean_24h(da_time)
    assert isinstance(rm24, xr.DataArray)

    # Diurnal cycle
    dc = analysis.diurnal_cycle(da_time)
    assert "hour" in dc.coords

    # Weighted spatial mean
    wsm = analysis.weighted_spatial_mean(sample_da)
    assert isinstance(wsm, xr.DataArray)

    # KZ filter xarray
    kz = analysis.kz_filter(da_time, m=3, k=2)
    assert isinstance(kz, xr.DataArray)

    # MDA8
    mda8 = analysis.mda8(da_time)
    assert isinstance(mda8, xr.DataArray)

    # Resample
    res = analysis.resample_data(da_time, freq="D")
    assert isinstance(res, xr.DataArray)

    # Exceedance
    exc = analysis.exceedance_count(da_time, threshold=0.5)
    assert isinstance(exc, xr.DataArray)

    # Percentile
    p = analysis.percentile(da_time, q=95)
    assert isinstance(p, xr.DataArray)

    # Peak timing
    pt = analysis.peak_timing(da_time)
    assert isinstance(pt, xr.DataArray)

    # FFT
    psd = analysis.fft_analysis(da_time)
    assert isinstance(psd, xr.DataArray)

    # Power spectrum
    spec = analysis.power_spectrum(da_time)
    assert isinstance(spec, xr.DataArray)

    # Climatology
    climo = analysis.climatology(da_time, freq="month")
    assert "month" in climo.coords


def test_utils_stats(sample_pair_da):
    """Boost coverage: utils_stats.py."""
    obs, mod = sample_pair_da

    # angular_difference
    ad = utils.angular_difference(obs, mod)
    assert isinstance(ad, xr.DataArray)
    ad_np = utils.angular_difference(obs.values, mod.values)
    assert np.shape(ad_np) == (5, 5)

    # rmse, mae, correlation
    assert isinstance(utils.rmse(obs, mod), xr.DataArray)
    assert isinstance(utils.mae(obs, mod), xr.DataArray)
    assert isinstance(utils.correlation(obs, mod), xr.DataArray)

    assert isinstance(utils.rmse(obs.values, mod.values), float)

    # matchmasks xarray
    m1, m2 = utils.matchmasks(obs, mod)
    assert isinstance(m1, xr.DataArray)


def test_performance_module(sample_pair_da):
    """Boost coverage: performance.py."""
    obs, mod = sample_pair_da

    # chunk_array
    chunks = perf.chunk_array(obs.values, chunk_size=5)
    assert len(chunks) == 5

    # vectorize_function
    def my_func(x, y):
        return x + y

    vec = perf.vectorize_function(my_func, [1, 2], [3, 4])
    assert np.all(vec == [4, 6])

    # parallel_compute
    def my_sum(x, axis=None):
        return np.sum(x, axis=axis)

    res = perf.parallel_compute(my_sum, obs.values, chunk_size=5)
    assert isinstance(res, (float, np.float64))

    # parallel_compute xarray
    # Actually perf.parallel_compute expects func(data, axis=axis)
    res_xr = perf.parallel_compute(lambda x, axis: x.mean(dim=axis), obs, axis="lat")
    assert isinstance(res_xr, xr.DataArray)

    # fast_rmse, fast_mae, memory_efficient_correlation
    perf.fast_rmse(obs, mod)
    perf.fast_mae(obs, mod)
    perf.memory_efficient_correlation(obs, mod)
    perf.optimize_for_size(utils.rmse, obs, mod)


def test_interfaces_module(sample_pair_da):
    """Boost coverage: interfaces.py."""
    obs, mod = sample_pair_da

    class MyMetric(interfaces.BaseStatisticalMetric):
        def compute(self, obs, mod, **kwargs):
            self.validate_inputs(obs, mod)
            if isinstance(obs, xr.DataArray):
                return self._handle_xarray(obs, mod, lambda o, m, **kw: (o - m).mean(**kw), **kwargs)
            else:
                return self._handle_numpy(obs, mod, lambda o, m, **kw: np.mean(o - m, **kw), **kwargs)

    metric = MyMetric()
    metric.compute(obs, mod, axis="lat")
    metric.compute(obs.values, mod.values, axis=0)

    # Masked arrays
    metric._handle_masked_arrays(obs.values, mod.values, lambda o, m, **kw: np.ma.mean(o - m, **kw), axis=0)

    # Legacy wrappers
    interfaces.DataProcessor.to_numpy(obs)
    interfaces.DataProcessor.align_arrays(obs, mod)
    interfaces.DataProcessor.handle_missing_values(obs, mod)
    interfaces.PerformanceOptimizer.chunk_array(obs.values)
    interfaces.PerformanceOptimizer.vectorize_function(lambda x: x * 2, [1, 2])
