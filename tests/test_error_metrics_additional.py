#!/usr/bin/env python3
"""
Additional tests for error_metrics.py to improve coverage.
Tests focus on functions that are not well-covered by existing tests.
"""

import numpy as np

from monet_stats.error_metrics import (
    CRMSE,
    MAE,
    MAPE,
    MASE,
    MB,
    MNB,
    MNE,
    MO,
    MP,
    NO,
    NOP,
    NP,
    NRMSE,
    NSC,
    RM,
    RMSPE,
    STDO,
    STDP,
    WDMB,
    MAE_m,
    MAPEm,
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
    WDMB_m,
    WDMdnB,
    sMAPE,
    sMAPE_original,
    sMAPEm,
)


class TestBasicErrorMetrics:
    """Test basic error metrics functions."""

    def test_stdo_basic(self):
        """Test STDO with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = STDO(obs, mod)
        expected = np.std(obs - mod)
        assert np.isclose(result, expected)

    def test_stdp_basic(self):
        """Test STDP with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = STDP(obs, mod)
        expected = np.std(mod - obs)
        assert np.isclose(result, expected)

    def test_mnb_basic(self):
        """Test MNB with basic numpy arrays."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = np.array([11.0, 22.0, 28.0, 43.0, 55.0])
        result = MNB(obs, mod)
        expected = np.ma.masked_invalid((mod - obs) / obs).mean() * 100.0
        assert np.isclose(result, expected)

    def test_mne_basic(self):
        """Test MNE with basic numpy arrays."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = np.array([11.0, 22.0, 28.0, 43.0, 55.0])
        result = MNE(obs, mod)
        expected = np.ma.masked_invalid(np.ma.abs(mod - obs) / obs).mean() * 100.0
        assert np.isclose(result, expected)


class TestMedianErrorMetrics:
    """Test median-based error metrics functions."""

    def test_mdnb_basic(self):
        """Test MdnNB with basic numpy arrays."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = np.array([11.0, 22.0, 28.0, 43.0, 55.0])
        result = MdnNB(obs, mod)
        expected = (
            np.ma.median(np.ma.masked_invalid((mod - obs) / obs), axis=None) * 100.0
        )
        assert np.isclose(result, expected)

    def test_mdne_basic(self):
        """Test MdnNE with basic numpy arrays."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = np.array([11.0, 22.0, 28.0, 43.0, 55.0])
        result = MdnNE(obs, mod)
        expected = (
            np.ma.median(np.ma.masked_invalid(np.ma.abs(mod - obs) / obs), axis=None)
            * 100.0
        )
        assert np.isclose(result, expected)

    def test_nmdnge_basic(self):
        """Test NMdnGE with basic numpy arrays."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = np.array([11.0, 22.0, 28.0, 43.0, 55.0])
        result = NMdnGE(obs, mod)
        expected = (
            np.ma.masked_invalid(
                np.ma.abs(mod - obs).mean(axis=None) / obs.mean(axis=None)
            )
            * 100.0
        )
        assert np.isclose(result, expected)


class TestCountMetrics:
    """Test observation/prediction count metrics."""

    def test_no_basic(self):
        """Test NO with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = NO(obs, mod)
        assert result == 5

    def test_nop_basic(self):
        """Test NOP with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = NOP(obs, mod)
        assert result == 5

    def test_np_basic(self):
        """Test NP with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = NP(obs, mod)
        assert result == 5


class TestMeanMedianErrorMetrics:
    """Test mean and median error metrics."""

    def test_mo_basic(self):
        """Test MO with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = MO(obs, mod)
        expected = np.mean(obs - mod)
        assert np.isclose(result, expected)

    def test_mp_basic(self):
        """Test MP with basic numpy arrays."""
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = MP(obs, mod)
        expected = np.mean(mod)
        assert np.isclose(result, expected)

    def test_mdno_basic(self):
        """Test MdnO with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = MdnO(obs, mod)
        expected = np.median(obs - mod)
        assert np.isclose(result, expected)

    def test_mdnp_basic(self):
        """Test MdnP with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = MdnP(obs, mod)
        expected = np.median(mod - obs)
        assert np.isclose(result, expected)


class TestRootErrorMetrics:
    """Test root error metrics."""

    def test_rm_basic(self):
        """Test RM with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = RM(obs, mod)
        expected = np.sqrt(np.mean((obs - mod) ** 2))
        assert np.isclose(result, expected)

    def test_rmdn_basic(self):
        """Test RMdn with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = RMdn(obs, mod)
        expected = np.sqrt(np.median((obs - mod) ** 2))
        assert np.isclose(result, expected)


class TestBiasMetrics:
    """Test bias metrics."""

    def test_mb_basic(self):
        """Test MB with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = MB(obs, mod)
        expected = np.ma.mean(obs - mod)
        assert np.isclose(result, expected)

    def test_mdnb_basic(self):
        """Test MdnB with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = MdnB(obs, mod)
        expected = np.ma.median(obs - mod)
        assert np.isclose(result, expected)


class TestWindDirectionMetrics:
    """Test wind direction metrics."""

    def test_wdmb_m_basic(self):
        """Test WDMB_m with basic numpy arrays."""
        obs = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        mod = np.array([10.0, 100.0, 190.0, 280.0, 370.0])
        result = WDMB_m(obs, mod)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_wdmb_basic(self):
        """Test WDMB with basic numpy arrays."""
        obs = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        mod = np.array([10.0, 100.0, 190.0, 280.0, 370.0])
        result = WDMB(obs, mod)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_wdmnnb_basic(self):
        """Test WDMdnB with basic numpy arrays."""
        obs = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        mod = np.array([10.0, 100.0, 190.0, 280.0, 370.0])
        result = WDMdnB(obs, mod)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)


class TestAbsoluteErrorMetrics:
    """Test absolute error metrics."""

    def test_mae_basic(self):
        """Test MAE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAE(obs, mod)
        expected = 0.6666666666666666
        assert np.isclose(result, expected)

    def test_medae_basic(self):
        """Test MedAE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MedAE(obs, mod)
        expected = 1.0
        assert np.isclose(result, expected)


class TestPercentageErrorMetrics:
    """Test percentage-based error metrics."""

    def test_smape_original_basic(self):
        """Test sMAPE_original with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = sMAPE_original(obs, mod)
        # Calculate expected value: 200 * |mod-obs| / (|mod| + |obs|) then mean
        expected = (
            200 * np.ma.abs(mod - obs) / (np.ma.abs(mod) + np.ma.abs(obs))
        ).mean()
        assert np.isclose(result, expected, rtol=1e-9)

    def test_crmse_basic(self):
        """Test CRMSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = CRMSE(obs, mod)
        expected = 0.4714045207910317
        assert np.isclose(result, expected, rtol=1e-9)

    def test_mape_basic(self):
        """Test MAPE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAPE(obs, mod)
        # Calculate expected value: 100 * |mod-obs| / |obs| then mean
        expected = (100 * np.ma.abs(mod - obs) / np.ma.abs(obs)).mean()
        assert np.isclose(result, expected)

    def test_smape_basic(self):
        """Test sMAPE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = sMAPE(obs, mod)
        # Calculate expected value: 200 * |mod-obs| / (|mod| + |obs|) then mean
        expected = (
            200 * np.ma.abs(mod - obs) / (np.ma.abs(mod) + np.ma.abs(obs))
        ).mean()
        assert np.isclose(result, expected, rtol=1e-9)


class TestNormalizedErrorMetrics:
    """Test normalized error metrics."""

    def test_nrmse_basic(self):
        """Test NRMSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([2.0, 2.0, 2.0, 2.0])
        result = NRMSE(obs, mod)
        # Calculate expected value: RMSE / (max(obs) - min(obs))
        rmse = np.ma.sqrt(np.ma.mean((mod - obs) ** 2))
        obs_range = np.ma.max(obs) - np.ma.min(obs)
        expected = rmse / obs_range
        assert np.isclose(result, expected, rtol=1e-9)

    def test_mase_basic(self):
        """Test MASE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1])
        result = MASE(obs, mod)
        expected = 0.1
        assert np.isclose(result, expected)

    def test_masem_basic(self):
        """Test MASEm with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1])
        result = MASEm(obs, mod)
        expected = 0.1
        assert np.isclose(result, expected)

    def test_rmspe_basic(self):
        """Test RMSPE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = RMSPE(obs, mod)
        # Calculate expected value: 100 * sqrt(mean((mod-obs)/obs)^2)
        expected = 100 * np.ma.sqrt(np.ma.mean(((mod - obs) / obs) ** 2))
        assert np.isclose(result, expected)


class TestRobustErrorMetrics:
    """Test robust error metrics."""

    def test_mape_m_basic(self):
        """Test MAPEm with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAPEm(obs, mod)
        # Calculate expected value: 100 * mean(|mod-obs| / |obs|)
        expected = 100 * np.ma.mean(np.ma.abs((mod - obs) / obs))
        assert np.isclose(result, expected)

    def test_smape_m_basic(self):
        """Test sMAPEm with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = sMAPEm(obs, mod)
        # Calculate expected value: 200 * mean(|mod-obs| / (|mod| + |obs|))
        expected = 200 * np.ma.mean(
            np.ma.abs(mod - obs) / (np.ma.abs(mod) + np.ma.abs(obs))
        )
        assert np.isclose(result, expected, rtol=1e-9)

    def test_nsc_basic(self):
        """Test NSC with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([2.0, 2.0, 2.0, 2.0])
        result = NSC(obs, mod)
        # Calculate expected value: 1 - sum((obs-mod)^2) / sum((obs-mean(obs))^2)
        obs_mean = np.mean(obs)
        numerator = np.sum((obs - mod) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        expected = 1.0 - (numerator / denominator)
        assert np.isclose(result, expected, rtol=1e-4)


class TestNSEComponents:
    """Test NSE decomposition components."""

    def test_nse_alpha_basic(self):
        """Test NSE_alpha with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([2.0, 2.0, 2.0, 2.0])
        result = NSE_alpha(obs, mod)
        expected = 0.0
        assert np.isclose(result, expected)

    def test_nse_beta_basic(self):
        """Test NSE_beta with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([2.0, 2.0, 2.0, 2.0])
        result = NSE_beta(obs, mod)
        # Calculate expected value: mean(mod) / mean(obs)
        expected = np.mean(mod) / np.mean(obs)
        assert np.isclose(result, expected)


class TestRobustAbsoluteErrorMetrics:
    """Test robust absolute error metrics."""

    def test_mae_m_basic(self):
        """Test MAE_m with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAE_m(obs, mod)
        expected = 0.66666666
        assert np.isclose(result, expected, rtol=1e-6)

    def test_medae_m_basic(self):
        """Test MedAE_m with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MedAE_m(obs, mod)
        expected = 1.0
        assert np.isclose(result, expected)
