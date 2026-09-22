"""Checks on the mean return difference test.

Three that carry weight: `test_matches_scalar_hac_route` -- the pair-and-contract
construction has to agree with running HAC straight on `D_t`, or the choice of
plumbing is silently moving the answer -- `test_exact_under_iid_normal`, which
uses the fact that a *linear* statistic has a closed-form standard error under
i.i.d. data to check the whole delta-method path against something known, and
the simulation-backed level checks at the bottom, which decide whether
`pval_boot` is worth preferring to `pval_hac`.
"""
import math
import numpy as np
import pandas as pd
import pytest

from src.Empirical_Analysis.inference.mean_inference import (
    mean_difference_test, _centred, _standard_error, _GRADIENT
)
from src.Empirical_Analysis.inference.sharpe_inference import _psi_hac


def _series (values: np.ndarray)-> pd.Series:
    return pd.Series(values,
                     index=pd.date_range('2015-01-31', periods=len(values),
                                         freq='ME'))


def _paired (n_obs: int = 132, seed: int = 0, tdf: float = 5.0,
             vol: float = 0.045, phi: float = 0.0, shift: float = 0.0,
             rho: float = 0.7)-> tuple[pd.Series, pd.Series]:
    """A correlated pair of heavy-tailed monthly return series.

    The same generator as `test_ce_inference`, so the two tests are exercised on
    identical data and any difference between them is the statistic rather than
    the sample. `shift` moves `x`'s mean only, so `shift=0` is the null and
    anything else a known alternative; `phi` adds the AR(1) persistence that the
    HAC estimator and the block bootstrap are both there to handle; `rho` loads
    a common factor against an idiosyncratic one with weights summing in
    quadrature to 1, moving the correlation while leaving each variance alone.
    """
    rng = np.random.default_rng(seed)
    burn: int = 50
    z = rng.standard_t(tdf, size=(n_obs+burn, 3))/np.sqrt(tdf/(tdf-2))
    idio: float = np.sqrt(1.0-rho**2)
    e_x = rho*z[:, 0]+idio*z[:, 1]
    e_b = rho*z[:, 0]+idio*z[:, 2]
    a_x = np.zeros(n_obs+burn)
    a_b = np.zeros(n_obs+burn)
    for t in range(1, n_obs+burn):
        a_x[t] = phi*a_x[t-1]+e_x[t]
        a_b[t] = phi*a_b[t-1]+e_b[t]
    r_x = np.maximum(0.008+shift+vol*a_x[burn:], -0.95)
    r_b = np.maximum(0.008+vol*a_b[burn:], -0.95)
    return _series(r_x), _series(r_b)


def _scalar_route (x: pd.Series, b: pd.Series)-> float:
    """The HAC standard error from running the estimator straight on `D_t`.

    This is what section 13 of the note describes and what the module docstring
    says is asymptotically the same estimator. Annualised to match.
    """
    d: np.ndarray = (x-b).dropna().to_numpy(float)
    psi: float = float(_psi_hac((d-d.mean())[:, None], prewhite=True)[0, 0])
    return math.sqrt(max(psi, 0.0)/len(d))*12


class TestPointEstimate:
    def test_mean_diff_is_the_difference_of_means (self):
        x, b = _paired()
        out = mean_difference_test(x, b, n_boot=0)
        assert out['mean_diff'] == pytest.approx(out['mean_x']-out['mean_b'])
        assert out['mean_x'] == pytest.approx(x.mean()*12)
        assert out['mean_b'] == pytest.approx(b.mean()*12)

    def test_annualisation_is_linear (self):
        """Unlike the CE there is no compounding and unlike the Sharpe no sqrt."""
        x, b = _paired()
        monthly = mean_difference_test(x, b, n_boot=0, periods_per_year=1)
        annual = mean_difference_test(x, b, n_boot=0)
        assert monthly['mean_diff']*12 == pytest.approx(annual['mean_diff'])
        assert monthly['mean_diff_se']*12 == pytest.approx(annual['mean_diff_se'])

    def test_t_stat_is_invariant_to_scale (self):
        """The p-values cannot depend on the reporting horizon."""
        x, b = _paired()
        monthly = mean_difference_test(x, b, n_boot=299, periods_per_year=1)
        annual = mean_difference_test(x, b, n_boot=299)
        assert monthly['t_hac'] == pytest.approx(annual['t_hac'])
        assert monthly['pval_hac'] == pytest.approx(annual['pval_hac'])
        assert monthly['pval_boot'] == annual['pval_boot']

    def test_risk_free_rate_cancels (self):
        """Excess or total returns give the identical test.

        Documented in the module docstring, and worth pinning: it is the one
        convention difference between this test and its two companions.
        """
        x, b = _paired()
        rf = _series(np.full(len(x), 0.002))
        total = mean_difference_test(x, b, n_boot=299)
        excess = mean_difference_test(x-rf, b-rf, n_boot=299)
        assert excess['mean_diff'] == pytest.approx(total['mean_diff'])
        assert excess['mean_diff_se'] == pytest.approx(total['mean_diff_se'])
        assert excess['pval_boot'] == total['pval_boot']


class TestConstruction:
    def test_matches_scalar_hac_route (self):
        """Contracting the 2x2 `Psi` with `(1,-1)` against HAC on `D_t` directly.

        Asymptotically the same estimator; in finite samples they differ through
        the prewhitening filter, the bandwidth and the `T/(T-k)` factor. If that
        gap were large the choice of route would be moving the published
        standard error, so it is pinned rather than assumed.
        """
        for seed in range(6):
            x, b = _paired(seed=seed, phi=0.2)
            paired = mean_difference_test(x, b, n_boot=0)['mean_diff_se']
            assert paired == pytest.approx(_scalar_route(x, b), rel=0.10)

    def test_exact_under_iid_normal (self):
        """Against the closed-form i.i.d. standard error of a paired mean.

        With no serial dependence the long-run variance is the contemporaneous
        one, so `SE = sd(D)/sqrt(T)` up to the HAC estimator's own small-sample
        factor. A linear statistic is the only one of the three where such a
        check is available at all.
        """
        rng = np.random.default_rng(4)
        n_obs: int = 4000
        d_cov = np.array([[1.0, 0.6], [0.6, 1.0]])*0.04**2
        draws = rng.multivariate_normal([0.008, 0.006], d_cov, size=n_obs)
        x, b = _series(draws[:, 0]), _series(draws[:, 1])
        out = mean_difference_test(x, b, n_boot=0, periods_per_year=1)
        closed = (draws[:, 0]-draws[:, 1]).std(ddof=1)/math.sqrt(n_obs)
        assert out['mean_diff_se'] == pytest.approx(closed, rel=0.05)

    def test_gradient_contraction_is_the_difference_variance (self):
        """`grad' Psi grad` is `Psi_00 - 2 Psi_01 + Psi_11`, by construction."""
        x, b = _paired()
        xa, ba = x.to_numpy(float), b.to_numpy(float)
        v = np.array([xa.mean(), ba.mean()])
        psi = _psi_hac(_centred(xa, ba, v))
        assert _standard_error(psi, len(xa)) == pytest.approx(
            math.sqrt((psi[0, 0]-2*psi[0, 1]+psi[1, 1])/len(xa)))
        assert list(_GRADIENT) == [1.0, -1.0]


class TestSymmetry:
    def test_swapping_flips_the_sign (self):
        x, b = _paired(shift=0.004)
        fwd = mean_difference_test(x, b, n_boot=0)
        rev = mean_difference_test(b, x, n_boot=0)
        assert fwd['mean_diff'] == pytest.approx(-rev['mean_diff'])
        assert fwd['mean_diff_se'] == pytest.approx(rev['mean_diff_se'])

    def test_greater_and_less_are_complementary (self):
        x, b = _paired(shift=0.004)
        hi = mean_difference_test(x, b, n_boot=0, alternative='greater')
        lo = mean_difference_test(x, b, n_boot=0, alternative='less')
        assert hi['pval_hac']+lo['pval_hac'] == pytest.approx(1.0)

    def test_two_sided_is_twice_the_smaller_tail (self):
        x, b = _paired(shift=0.004)
        two = mean_difference_test(x, b, n_boot=0)['pval_hac']
        hi = mean_difference_test(x, b, n_boot=0,
                                  alternative='greater')['pval_hac']
        assert two == pytest.approx(2*min(hi, 1-hi))

    def test_identical_series_give_zero_difference (self):
        x, _ = _paired()
        out = mean_difference_test(x, x, n_boot=199)
        assert out['mean_diff'] == pytest.approx(0.0, abs=1e-15)


class TestPairing:
    def test_cross_covariance_shrinks_the_standard_error (self):
        """Two positively correlated books differ more precisely than the
        independence formula would allow.

        This is the reason the test keeps `Psi`'s off-diagonal and resamples the
        pair together. If the standard error came out at
        `sqrt(SE_x^2 + SE_b^2)` instead, a real difference between two
        overlapping books would be buried.
        """
        tight = mean_difference_test(*_paired(rho=0.95, seed=3), n_boot=0)
        loose = mean_difference_test(*_paired(rho=0.05, seed=3), n_boot=0)
        assert tight['mean_diff_se'] < loose['mean_diff_se']

    def test_bootstrap_preserves_pairing (self):
        """Resampling the two series with independent blocks would widen the
        pivot's spread and push the p-value up on a correlated pair."""
        x, b = _paired(rho=0.95, shift=0.006, seed=5)
        out = mean_difference_test(x, b, n_boot=999, alternative='greater')
        shuffled = mean_difference_test(
            x, _series(b.to_numpy()[::-1]), n_boot=999, alternative='greater')
        assert out['pval_boot'] < shuffled['pval_boot']


class TestValidation:
    def test_unknown_alternative_raises (self):
        x, b = _paired()
        with pytest.raises(ValueError, match='alternative must be'):
            mean_difference_test(x, b, alternative='bigger')

    def test_too_few_months_raises (self):
        x, b = _paired(n_obs=3)
        with pytest.raises(ValueError, match='overlapping months'):
            mean_difference_test(x, b)

    def test_block_size_out_of_range_raises (self):
        x, b = _paired()
        with pytest.raises(ValueError, match='block_size must be in'):
            mean_difference_test(x, b, block_size=999)

    def test_misaligned_months_are_dropped_from_both (self):
        x, b = _paired()
        out = mean_difference_test(x, b.iloc[10:], n_boot=0)
        assert out['n_month'] == len(b)-10

    def test_seed_reproduces_the_bootstrap (self):
        x, b = _paired()
        first = mean_difference_test(x, b, n_boot=199, seed=7)
        again = mean_difference_test(x, b, n_boot=199, seed=7)
        assert first['pval_boot'] == again['pval_boot']


class TestPower:
    def test_a_large_true_difference_is_detected (self):
        x, b = _paired(shift=0.012, vol=0.02, seed=11)
        out = mean_difference_test(x, b, n_boot=999, alternative='greater')
        assert out['mean_diff'] > 0
        assert out['pval_boot'] < 0.05

    def test_no_difference_is_not_detected (self):
        x, b = _paired(seed=12)
        out = mean_difference_test(x, b, n_boot=999)
        assert out['pval_boot'] > 0.10


@pytest.mark.slow
class TestLevel:
    """Does the test reject at its nominal rate when the null is true?

    The same check `test_ce_inference` runs, on the same data-generating
    process, so the two are directly comparable. The expectation going in is
    that a linear statistic is better behaved than the CE's non-linear one --
    there is no transformation to skew the sampling distribution, only the HAC
    estimate of `Psi` to get wrong. These assertions record what was measured.
    """
    NOMINAL: float = 0.10
    NSIM: int = 300

    def _rejection_rate (self, **kwargs)-> tuple[float, float]:
        hac = boot = 0
        for sim in range(self.NSIM):
            x, b = _paired(seed=1000+sim, **kwargs)
            out = mean_difference_test(x, b, n_boot=299, seed=sim)
            hac += out['pval_hac'] < self.NOMINAL
            boot += out['pval_boot'] < self.NOMINAL
        return hac/self.NSIM, boot/self.NSIM

    def test_moderate_tails_hold_nominal_level (self):
        """At t5 and 4.5% monthly vol both estimators are correctly sized."""
        hac, boot = self._rejection_rate(tdf=5.0, vol=0.045)
        assert hac == pytest.approx(self.NOMINAL, abs=0.05)
        assert boot == pytest.approx(self.NOMINAL, abs=0.05)

    def test_heavy_tails_barely_move_the_level (self):
        """At t3, 9% vol and AR(1) persistence -- the profile of a concentrated
        book -- the linear statistic holds its level where the CE difference
        does not.

        Measured at 0.110 for HAC and 0.100 for the bootstrap against a nominal
        0.10, on the identical process that drove the CE test's HAC to
        0.18-0.20. The mean test's size distortion is the mildest of the three,
        which is what makes its rejection the least likely to be an artefact --
        and therefore throws the whole weight of interpretation onto what the
        statistic measures rather than onto how well it is estimated. See the
        warning at the end of `mean_inference`'s module docstring.
        """
        hac, boot = self._rejection_rate(tdf=3.0, vol=0.09, phi=0.2)
        assert hac == pytest.approx(self.NOMINAL, abs=0.05), f'hac {hac}'
        assert boot == pytest.approx(self.NOMINAL, abs=0.05), f'boot {boot}'
