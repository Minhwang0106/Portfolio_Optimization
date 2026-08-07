"""Checks on the CRRA certainty equivalent difference test.

The two that matter are `test_ce_matches_metrics` -- the test's own point
estimate has to be the number the summary table prints, or it is testing
something nobody reported -- and the simulation-backed level checks at the
bottom, which are what decide whether `pval_boot` is worth preferring to
`pval_hac`.
"""
import numpy as np
import pandas as pd
import pytest

from src.Empirical_Analysis.ce_inference import (
    ce_difference_test, _exponent, _gradient, _utility
)
from src.Empirical_Analysis.metrics import crra_certainty_equivalent

GAMMA: float = 5.0


def _series (values: np.ndarray)-> pd.Series:
    return pd.Series(values,
                     index=pd.date_range('2015-01-31', periods=len(values),
                                         freq='ME'))


def _paired (n_obs: int = 132, seed: int = 0, tdf: float = 5.0,
             vol: float = 0.045, phi: float = 0.0, shift: float = 0.0,
             rho: float = 0.7)-> tuple[pd.Series, pd.Series]:
    """A correlated pair of heavy-tailed monthly return series.

    `shift` moves `x`'s mean only, so `shift=0` is the null and anything else is
    a known alternative. `phi` adds AR(1) persistence, which is what the HAC
    estimator and the block bootstrap are both there to handle.

    `rho` loads a common factor against an idiosyncratic one with weights
    summing in quadrature to 1, so it moves the correlation between the two
    series (to `rho**2`) while leaving each one's variance alone. Scaling both
    components by `rho` instead would hold the correlation fixed at 0.5 and
    vary the volatility, which is the opposite of what the pairing test needs.
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


class TestPointEstimate:
    def test_ce_matches_metrics (self):
        """`ce_x`/`ce_b` are exactly what `crra_certainty_equivalent` reports.

        The annualisation is folded into the exponent
        (`mu^(12/(1-g))-1`) rather than compounded afterwards, and this pins
        that the algebra agrees with the metric actually printed.
        """
        x, b = _paired()
        out = ce_difference_test(x, b, gamma=GAMMA, n_boot=0)
        assert out['ce_x'] == pytest.approx(
            crra_certainty_equivalent(x, GAMMA), rel=1e-12)
        assert out['ce_b'] == pytest.approx(
            crra_certainty_equivalent(b, GAMMA), rel=1e-12)

    def test_ce_diff_is_the_difference (self):
        x, b = _paired()
        out = ce_difference_test(x, b, gamma=GAMMA, n_boot=0)
        assert out['ce_diff'] == pytest.approx(out['ce_x']-out['ce_b'])

    def test_monthly_scale_compounds_to_annual (self):
        """`periods_per_year=1` is the monthly CE; compounding it gives the annual."""
        x, b = _paired()
        monthly = ce_difference_test(x, b, gamma=GAMMA, n_boot=0,
                                     periods_per_year=1)
        annual = ce_difference_test(x, b, gamma=GAMMA, n_boot=0)
        assert (1+monthly['ce_x'])**12-1 == pytest.approx(annual['ce_x'])

    def test_gradient_matches_numerical (self):
        """The delta-method gradient against a finite difference of `h`."""
        c: float = _exponent(GAMMA, 12)
        v = np.array([0.93, 0.97])
        grad = _gradient(v, c)
        step: float = 1e-7
        for i in (0, 1):
            bump = v.copy()
            bump[i] += step
            numeric = ((bump[0]**c-bump[1]**c)-(v[0]**c-v[1]**c))/step
            assert grad[i] == pytest.approx(numeric, rel=1e-5)


class TestSymmetry:
    def test_swapping_flips_the_sign (self):
        x, b = _paired(shift=0.004)
        fwd = ce_difference_test(x, b, gamma=GAMMA, n_boot=0)
        rev = ce_difference_test(b, x, gamma=GAMMA, n_boot=0)
        assert fwd['ce_diff'] == pytest.approx(-rev['ce_diff'])
        assert fwd['ce_diff_se'] == pytest.approx(rev['ce_diff_se'])

    def test_greater_and_less_are_complementary (self):
        """One-sided p-values in opposite directions sum to 1 under HAC."""
        x, b = _paired(shift=0.004)
        hi = ce_difference_test(x, b, gamma=GAMMA, n_boot=0,
                                alternative='greater')
        lo = ce_difference_test(x, b, gamma=GAMMA, n_boot=0,
                                alternative='less')
        assert hi['pval_hac']+lo['pval_hac'] == pytest.approx(1.0)

    def test_identical_series_give_zero_difference (self):
        x, _ = _paired()
        out = ce_difference_test(x, x, gamma=GAMMA, n_boot=199)
        assert out['ce_diff'] == pytest.approx(0.0, abs=1e-15)


class TestPairing:
    def test_cross_covariance_shrinks_the_standard_error (self):
        """Two positively correlated books differ more precisely than the
        independence formula would allow.

        This is the reason the test resamples the pair together and keeps
        `Psi`'s off-diagonal. If the standard error came out at
        `sqrt(SE_A^2 + SE_B^2)` instead, a real difference between two
        overlapping books would be buried.
        """
        tight_x, tight_b = _paired(rho=0.95, seed=3)
        loose_x, loose_b = _paired(rho=0.05, seed=3)
        tight = ce_difference_test(tight_x, tight_b, gamma=GAMMA, n_boot=0)
        loose = ce_difference_test(loose_x, loose_b, gamma=GAMMA, n_boot=0)
        assert tight['ce_diff_se'] < loose['ce_diff_se']


class TestValidation:
    def test_gamma_one_raises (self):
        x, b = _paired()
        with pytest.raises(ValueError, match='gamma must not be 1'):
            ce_difference_test(x, b, gamma=1.0)

    def test_unknown_alternative_raises (self):
        x, b = _paired()
        with pytest.raises(ValueError, match='alternative must be'):
            ce_difference_test(x, b, alternative='bigger')

    def test_total_loss_raises (self):
        """A month at -100% has no CRRA utility, so there is no CE to test."""
        x, b = _paired()
        x.iloc[5] = -1.0
        with pytest.raises(ValueError, match='-100%'):
            ce_difference_test(x, b, gamma=GAMMA)

    def test_too_few_months_raises (self):
        x, b = _paired(n_obs=3)
        with pytest.raises(ValueError, match='overlapping months'):
            ce_difference_test(x, b, gamma=GAMMA)

    def test_block_size_out_of_range_raises (self):
        x, b = _paired()
        with pytest.raises(ValueError, match='block_size must be in'):
            ce_difference_test(x, b, gamma=GAMMA, block_size=999)

    def test_misaligned_months_are_dropped_from_both (self):
        x, b = _paired()
        out = ce_difference_test(x, b.iloc[10:], gamma=GAMMA, n_boot=0)
        assert out['n_month'] == len(b)-10

    def test_seed_reproduces_the_bootstrap (self):
        x, b = _paired()
        first = ce_difference_test(x, b, gamma=GAMMA, n_boot=199, seed=7)
        again = ce_difference_test(x, b, gamma=GAMMA, n_boot=199, seed=7)
        assert first['pval_boot'] == again['pval_boot']


class TestPower:
    def test_a_large_true_difference_is_detected (self):
        """A CE gap far outside sampling noise rejects the one-sided null."""
        x, b = _paired(shift=0.02, vol=0.02, seed=11)
        out = ce_difference_test(x, b, gamma=GAMMA, n_boot=999,
                                 alternative='greater')
        assert out['ce_diff'] > 0
        assert out['pval_boot'] < 0.05

    def test_no_difference_is_not_detected (self):
        x, b = _paired(seed=12)
        out = ce_difference_test(x, b, gamma=GAMMA, n_boot=999,
                                 alternative='two-sided')
        assert out['pval_boot'] > 0.10


@pytest.mark.slow
class TestLevel:
    """Does the test reject at its nominal rate when the null is true?

    This is the check that decides which p-value the summary table should
    report, and the answer depends on the series. Both estimators are fine on
    moderate tails; both degrade as the tails thicken, the HAC one considerably
    faster. See the module docstring of `ce_inference`.
    """
    NOMINAL: float = 0.10
    NSIM: int = 300

    def _rejection_rate (self, **kwargs)-> tuple[float, float]:
        hac = boot = 0
        for sim in range(self.NSIM):
            x, b = _paired(seed=1000+sim, **kwargs)
            out = ce_difference_test(x, b, gamma=GAMMA, n_boot=299, seed=sim,
                                     alternative='two-sided')
            hac += out['pval_hac'] < self.NOMINAL
            boot += out['pval_boot'] < self.NOMINAL
        return hac/self.NSIM, boot/self.NSIM

    def test_moderate_tails_hold_nominal_level (self):
        """At t5 and 4.5% monthly vol both estimators are correctly sized."""
        hac, boot = self._rejection_rate(tdf=5.0, vol=0.045)
        assert hac == pytest.approx(self.NOMINAL, abs=0.05)
        assert boot == pytest.approx(self.NOMINAL, abs=0.05)

    def test_heavy_tails_make_hac_liberal_and_boot_less_so (self):
        """At t3 and 9% vol -- the profile of a concentrated book -- the HAC
        p-value over-rejects badly and the bootstrap over-rejects less.

        Neither is exact, which is the point: this is a caveat to report
        alongside the number, not a solved problem. Measured at roughly
        HAC 0.18-0.20 against bootstrap 0.14-0.15 for a nominal 0.10.
        """
        hac, boot = self._rejection_rate(tdf=3.0, vol=0.09, phi=0.2)
        assert hac > self.NOMINAL+0.04, f'expected HAC liberal, got {hac}'
        assert boot < hac, f'expected boot tighter than HAC, got {boot} vs {hac}'
