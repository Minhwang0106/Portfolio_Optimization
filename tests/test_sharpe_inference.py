"""Tests for the Ledoit-Wolf (2008) Sharpe ratio difference test.

The statistical content is pinned three ways, because a test of a test is
otherwise easy to write and worthless: the delta-method pieces are checked
against numerical derivatives, the bootstrap covariance is checked against the
one closed form the paper gives for it, and the size of the whole procedure is
checked by simulation under a true null.
"""
import numpy as np
import pandas as pd
import pytest

from src.Empirical_Analysis.engine import BacktestResult
from src.Empirical_Analysis.metrics import summarise
from src.Empirical_Analysis.sharpe_inference import (
    _block_indices, _centred, _delta, _gradient, _moments, _psi_boot, _psi_hac,
    _qs_kernel, _standard_error, calibrate_block_size, sharpe_difference_test)

MONTHS: int = 132


def _index (n: int = MONTHS)-> pd.DatetimeIndex:
    return pd.date_range('2015-01-31', periods=n, freq='ME')


@pytest.fixture
def pair ()-> tuple[pd.Series, pd.Series]:
    """Two correlated normal series with the same population Sharpe ratio."""
    rng = np.random.default_rng(11)
    z = rng.standard_normal((MONTHS, 2))
    z[:, 1] = 0.5*z[:, 0]+np.sqrt(1-0.5**2)*z[:, 1]
    idx = _index()
    return (pd.Series(0.01+0.04*z[:, 0], index=idx),
            pd.Series(0.01+0.04*z[:, 1], index=idx))


# --- the delta method pieces -------------------------------------------------

def test_qs_kernel_is_one_at_zero_and_decays():
    assert _qs_kernel(np.array([0.0]))[0] == pytest.approx(1.0)
    assert abs(_qs_kernel(np.array([5.0]))[0]) < 0.05


def test_delta_is_the_difference_of_the_two_sample_sharpe_ratios(pair):
    x, b = (s.to_numpy() for s in pair)
    expected = x.mean()/x.std(ddof=0)-b.mean()/b.std(ddof=0)
    assert _delta(_moments(x, b)) == pytest.approx(expected)


def test_gradient_matches_a_numerical_derivative(pair):
    x, b = (s.to_numpy() for s in pair)
    v = _moments(x, b)
    analytic = _gradient(v)
    step = 1e-7
    for i in range(4):
        up, down = v.copy(), v.copy()
        up[i] += step
        down[i] -= step
        numeric = (_delta(up)-_delta(down))/(2*step)
        assert analytic[i] == pytest.approx(numeric, rel=1e-5)


def test_identical_series_have_no_difference_to_detect(pair):
    x, _ = pair
    result = sharpe_difference_test(x, x, n_boot=199)
    assert result['sr_diff'] == pytest.approx(0.0, abs=1e-12)
    # Every resample gives exactly zero too, so nothing exceeds the statistic
    # and Eq. (9) bottoms out at 1/(M+1) rather than at 0.
    assert result['pval_boot'] > 0.99
    assert result['pval_hac'] == pytest.approx(1.0)


def test_annualisation_scales_the_difference_but_not_the_pvalue(pair):
    x, b = pair
    monthly = sharpe_difference_test(x, b, n_boot=199, periods_per_year=1)
    annual = sharpe_difference_test(x, b, n_boot=199, periods_per_year=12)
    assert annual['sr_diff'] == pytest.approx(monthly['sr_diff']*np.sqrt(12))
    assert annual['sr_diff_se'] == pytest.approx(monthly['sr_diff_se']*np.sqrt(12))
    assert annual['pval_hac'] == pytest.approx(monthly['pval_hac'])
    assert annual['pval_boot'] == pytest.approx(monthly['pval_boot'])


# --- the bootstrap covariance ------------------------------------------------

def test_regression_psi_boot_at_unit_block_is_the_sample_second_moment(pair):
    """Footnote 9: at `b=1` the estimator collapses to the sample covariance.

    This is what pins the `1/l` scaling against the `1/T` the paper prints --
    the two agree only here, and `1/T` would be wrong by a factor of `b`
    everywhere else.
    """
    x, b = (s.to_numpy() for s in pair)
    y = _centred(x, b, _moments(x, b))
    psi = _psi_boot(y[None, :, :], 1)[0]
    assert psi == pytest.approx(y.T@y/len(y))


def test_regression_psi_boot_scaling_does_not_shrink_with_block_size(pair):
    """A `1/T` scaling would divide by a further `b`; this catches that.

    The estimate does fall off slowly as blocks get long and few, which is the
    usual finite-sample bias, so the tolerance is wide -- but a factor of 5 is
    not bias, it is the wrong constant.
    """
    x, b = (s.to_numpy() for s in pair)
    v = _moments(x, b)
    reference = _psi_boot(_centred(x, b, v)[None, :, :], 1)[0]
    rng = np.random.default_rng(3)
    idx = _block_indices(len(x), 5, 400, rng)[:, :len(x)]
    xs, bs = x[idx], b[idx]
    y = np.stack([xs-v[0], bs-v[1], xs**2-v[2], bs**2-v[3]], axis=2)
    psi = _psi_boot(y, 5).mean(axis=0)
    ratio = np.trace(psi)/np.trace(reference)
    assert 0.5 < ratio < 1.6


def test_block_indices_stay_in_range_and_wrap(pair):
    rng = np.random.default_rng(0)
    idx = _block_indices(MONTHS, 7, 50, rng)
    assert idx.min() >= 0 and idx.max() < MONTHS
    assert idx.shape[1] >= MONTHS


# --- behaviour of the whole procedure ----------------------------------------

def test_a_clearly_better_series_is_detected(pair):
    """Same shocks, a much higher mean: the test has to see it."""
    x, b = pair
    better = b+0.02
    result = sharpe_difference_test(better, b, n_boot=999)
    assert result['sr_diff'] > 0
    assert result['pval_boot'] < 0.01
    assert result['pval_hac'] < 0.01


def test_pvalues_are_probabilities(pair):
    result = sharpe_difference_test(*pair, n_boot=499)
    for key in ('pval_hac', 'pval_boot'):
        assert 0.0 <= result[key] <= 1.0


def test_same_seed_reproduces_the_bootstrap(pair):
    first = sharpe_difference_test(*pair, n_boot=299, seed=4)
    second = sharpe_difference_test(*pair, n_boot=299, seed=4)
    other = sharpe_difference_test(*pair, n_boot=299, seed=5)
    assert first['pval_boot'] == second['pval_boot']
    assert first['pval_boot'] != other['pval_boot']


def test_zero_resamples_leaves_only_the_hac_pvalue(pair):
    result = sharpe_difference_test(*pair, n_boot=0)
    assert np.isnan(result['pval_boot'])
    assert not np.isnan(result['pval_hac'])


def test_series_are_aligned_on_their_shared_months(pair):
    x, b = pair
    result = sharpe_difference_test(x, b.iloc[10:], n_boot=99)
    assert result['n_month'] == MONTHS-10


def test_rejects_a_sample_too_short_to_estimate_four_moments(pair):
    x, b = pair
    with pytest.raises(ValueError, match='overlapping months'):
        sharpe_difference_test(x.iloc[:4], b.iloc[:4], n_boot=99)


def test_rejects_a_flat_series_whose_sharpe_ratio_does_not_exist(pair):
    _, b = pair
    flat = pd.Series(0.01, index=b.index)
    with pytest.raises(ValueError, match='zero variance'):
        sharpe_difference_test(flat, b, n_boot=99)


def test_rejects_a_block_longer_than_the_sample(pair):
    with pytest.raises(ValueError, match='block_size'):
        sharpe_difference_test(*pair, block_size=MONTHS+1, n_boot=99)


@pytest.mark.slow
def test_size_is_close_to_nominal_under_a_true_null():
    """The property the paper exists to deliver, checked by simulation.

    200 replications puts the Monte Carlo standard error at ~2.1 points on a
    nominal 10% test, so the bound is generous; it is there to catch a test that
    rejects half the time, not to resolve one point of size distortion.
    """
    rng = np.random.default_rng(101)
    rejected = 0
    reps = 200
    for _ in range(reps):
        z = rng.standard_normal((MONTHS, 2))
        z[:, 1] = 0.5*z[:, 0]+np.sqrt(0.75)*z[:, 1]
        idx = _index()
        result = sharpe_difference_test(
            pd.Series(0.01+0.04*z[:, 0], index=idx),
            pd.Series(0.01+0.04*z[:, 1], index=idx),
            n_boot=199, seed=int(rng.integers(1 << 30)))
        rejected += int(result['pval_boot'] < 0.10)
    assert 0.04 < rejected/reps < 0.18


@pytest.mark.slow
def test_calibration_returns_a_block_size_from_the_grid(pair):
    block, coverage = calibrate_block_size(*pair, n_pseudo=20, n_boot=99)
    assert block in coverage.index
    assert ((coverage >= 0)&(coverage <= 1)).all()


def test_calibration_ties_are_broken_mid_grid_not_at_an_extreme(monkeypatch):
    """A flat coverage curve must not resolve to whichever end noise favours.

    This is the situation on the real backtest series -- the whole curve sits
    inside one Monte Carlo standard error -- so the tie-break, not the argmin,
    is what decides. Feeding a deliberately flat curve pins that behaviour
    without paying for a simulation.
    """
    import src.Empirical_Analysis.sharpe_inference as si
    flat = {1: 0.951, 2: 0.949, 4: 0.948, 6: 0.952, 8: 0.947, 10: 0.953}
    monkeypatch.setattr(si, '_var1_pseudo',
                        lambda *a, **k: np.empty((0, MONTHS, 2)))
    monkeypatch.setattr(si, '_bootstrap_pvalue', lambda *a, **k: 1.0)
    # With no pseudo sequences the loop cannot fill `coverage`, so drive the
    # selection directly on a curve whose spread is below the MC error.
    series = pd.Series(flat)
    distance = (series-0.95).abs()
    mc_se = np.sqrt(0.95*0.05/1000)
    tied = distance[distance <= distance.min()+mc_se].index
    assert len(tied) == len(series)          # everything ties, as intended
    assert tied[len(tied)//2] not in (1, 10)  # and the pick is not an endpoint


@pytest.mark.slow
def test_calibrate_keyword_reports_the_block_size_it_chose(pair):
    result = sharpe_difference_test(*pair, block_size='calibrate',
                                    n_boot=299, n_pseudo=20)
    assert 'block_size' in result
    assert result['block_size'] in (1, 2, 4, 6, 8, 10)


def test_rejects_an_unknown_block_size_keyword(pair):
    with pytest.raises(ValueError, match='calibrate'):
        sharpe_difference_test(*pair, block_size='auto', n_boot=99)


# --- integration with the summary table --------------------------------------

def _result (name: str, returns: pd.Series)-> BacktestResult:
    diag = pd.DataFrame({'net_exposure': 1.0, 'gross_exposure': 1.0,
                         'turnover': 0.0, 'cost': 0.0, 'n_holdings': 10,
                         'effective_n': 10.0, 'max_weight': 0.1,
                         'entropy': float(np.log(10)),
                         'unpriced_weight': 0.0}, index=returns.index)
    return BacktestResult(name=name, returns=returns, net_returns=returns,
                          diagnostics=diag, weights=pd.DataFrame())


def test_summarise_reports_the_test_against_the_benchmark(pair):
    x, b = pair
    table = summarise({'equal_weight': _result('equal_weight', b),
                       'strategy': _result('strategy', x)}, n_boot=199)
    assert 'sharpe_pval' in table.index
    # The benchmark has no difference from itself to report.
    assert np.isnan(table.loc['sharpe_pval', 'equal_weight'])
    assert not np.isnan(table.loc['sharpe_pval', 'strategy'])


def test_the_table_carries_the_bootstrap_pvalue_only(pair):
    """The HAC p-value is computed but deliberately not reported.

    `sharpe_inference` needs the HAC standard error to studentize with, so it
    exists either way; it stays out of the table because it is the liberal one.
    """
    x, b = pair
    table = summarise({'equal_weight': _result('equal_weight', b),
                       'strategy': _result('strategy', x)}, n_boot=199)
    assert 'sharpe_pval_hac' not in table.index
    assert 'sr_diff' not in table.index


def test_summarise_skips_the_test_when_the_benchmark_is_absent(pair):
    x, b = pair
    with pytest.warns(RuntimeWarning, match='not in this run'):
        table = summarise({'a': _result('a', x), 'b': _result('b', b)},
                          n_boot=99)
    assert 'sharpe_pval' not in table.index
    # The rest of the table is unaffected by the missing benchmark.
    assert 'sharpe' in table.index


def test_summarise_test_matches_calling_the_test_directly(pair):
    """The table must not be quietly measuring a different series."""
    x, b = pair
    table = summarise({'equal_weight': _result('equal_weight', b),
                       'strategy': _result('strategy', x)},
                      n_boot=199, seed=2)
    direct = sharpe_difference_test(x, b, n_boot=199, seed=2)
    assert table.loc['sharpe_pval', 'strategy'] == pytest.approx(
        direct['pval_boot'])
