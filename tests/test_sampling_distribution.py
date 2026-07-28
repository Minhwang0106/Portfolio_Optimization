"""Tests for `src.Proposed_Model.utils.sampling_distribution`.

The `test_regression_*` cases cover defects found while wiring this module into
`RIM_PortOp.sampling`: a conditional draw scaled by the variance instead of its
square root, a signal read from the wrong end of the window, a `LagMoments`
demanded where two arrays would do, and a forward recursion that skipped the
history its first signal is built from. They are named so that a future
refactor reintroducing any of them fails loudly.

`model.py` itself is not imported here: it runs `generator()` at class-definition
time, so merely collecting it would pull the whole data pipeline into the suite.
`test_forward_recursion_*` instead mirrors the loop that method runs, on a
synthetic panel, which is where the logic worth pinning down actually lives.
"""

from collections import deque

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.Proposed_Model.utils.sampling_distribution import (
    LagMoments,
    conditional_mean,
    conditional_moments,
    lag_moments,
    rolling_signal,
    sample_conditional,
    signal_cal,
    signal_moments,
    signal_weights,
)
from src.Proposed_Model.utils.theta import find_theta, objective_func

PHI = 0.7


def ar1(n: int, seed: int, phi: float = PHI) -> np.ndarray:
    """An AR(1) with autocorrelation `phi**k`, so the truth is known in closed
    form and the fitted moments have something to be checked against."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x[:, None]


@pytest.fixture
def panel() -> np.ndarray:
    """A `(120, 3)` panel of independent AR(1) columns."""
    return np.hstack([ar1(120, s) for s in range(3)])


# --------------------------------------------------------------------------
# signal_weights
# --------------------------------------------------------------------------

@pytest.mark.parametrize('n_lags', [1, 2, 4, 9])
@pytest.mark.parametrize('theta', [0.0, 0.15, 0.5, 0.9, 1.0])
def test_weights_sum_to_one(n_lags, theta):
    """What makes `E[y_t] == E[x_t]`, and so lets eq (17) use one mean for
    both sides. Every other property here rests on it."""
    assert signal_weights(n_lags, theta).sum() == pytest.approx(1.0)


def test_weights_sum_to_one_per_row_for_array_theta():
    w = signal_weights(5, np.array([0.0, 0.3, 1.0]))
    assert w.shape == (3, 5)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_scalar_theta_gives_one_vector_array_theta_one_row_each():
    assert signal_weights(4, 0.5).shape == (4,)
    assert signal_weights(4, np.array([0.5])).shape == (1, 4)


def test_theta_one_is_all_weight_on_the_most_recent_lag():
    np.testing.assert_allclose(signal_weights(4, 1.0), [1.0, 0.0, 0.0, 0.0])


def test_theta_zero_is_all_weight_on_the_tail_term():
    np.testing.assert_allclose(signal_weights(4, 0.0), [0.0, 0.0, 0.0, 1.0])


def test_weights_decay_geometrically():
    w = signal_weights(5, 0.4)
    np.testing.assert_allclose(w[:4], 0.4 * 0.6 ** np.arange(4))


def test_n_lags_below_one_raises():
    with pytest.raises(ValueError, match='at least 1'):
        signal_weights(0, 0.5)


# --------------------------------------------------------------------------
# signal_cal -- array form
# --------------------------------------------------------------------------

def test_window_is_read_newest_last():
    """Time order in, most recent weighted first. With theta=1 all the weight
    sits on the last row, which is the only unambiguous way to pin the
    orientation down."""
    w = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    np.testing.assert_allclose(signal_cal(w, 3, 1.0), [3.0, 30.0])
    np.testing.assert_allclose(signal_cal(w, 3, 0.0), [1.0, 10.0])


def test_output_drops_the_time_axis():
    rng = np.random.default_rng(0)
    assert np.ndim(signal_cal(rng.normal(size=4), 4, 0.5)) == 0
    assert signal_cal(rng.normal(size=(4, 3)), 4, 0.5).shape == (3,)
    assert signal_cal(rng.normal(size=(4, 7, 3)), 4, 0.5).shape == (7, 3)


def test_only_the_last_n_lags_are_read():
    rng = np.random.default_rng(1)
    long = rng.normal(size=(20, 3))
    np.testing.assert_allclose(signal_cal(long, 4, 0.4),
                               signal_cal(long[-4:], 4, 0.4))


def test_short_window_renormalises_onto_the_horizon_available():
    """Not zero padded: a two-row window is eq (10) at n_lags=2, so the
    weights still sum to 1 and a constant window returns that constant."""
    const = np.ones((2, 3))
    np.testing.assert_allclose(signal_cal(const, 5, 0.4), np.ones(3))


def test_empty_array_window_raises():
    with pytest.raises(ValueError, match='window is empty'):
        signal_cal(np.empty((0, 3)), 4, 0.5)


def test_regression_multi_dim_window_does_not_contract_the_wrong_axis():
    """`np.dot` reduces against the second-to-last axis, which on a
    `(T, n_simulation, d)` block silently returns the wrong sum whenever
    `n_simulation == n_lags`. The contraction is written out to avoid it, so
    the square case has to agree with a non-square one.
    """
    rng = np.random.default_rng(2)
    block = rng.normal(size=(4, 4, 3))          # n_simulation == n_lags
    got = signal_cal(block, 4, 0.35)
    w = signal_weights(4, 0.35)
    expected = np.einsum('i,isd->sd', w, block[::-1])
    np.testing.assert_allclose(got, expected)
    assert got.shape == (4, 3)


# --------------------------------------------------------------------------
# signal_cal -- sequence form, the input the forward recursion actually holds
# --------------------------------------------------------------------------

@pytest.mark.parametrize('theta', [0.0, 0.4, np.array([0.1, 0.5, 0.9])])
def test_sequence_of_blocks_matches_the_stacked_array(theta):
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(6, 3)) for _ in range(4)]
    np.testing.assert_allclose(signal_cal(blocks, 4, theta),
                               signal_cal(np.stack(blocks), 4, theta))


@pytest.mark.parametrize('container', [list, tuple, deque])
def test_any_sequence_container_works(container):
    rng = np.random.default_rng(4)
    blocks = [rng.normal(size=(6, 3)) for _ in range(4)]
    np.testing.assert_allclose(signal_cal(container(blocks), 4, 0.4),
                               signal_cal(np.stack(blocks), 4, 0.4))


def test_bounded_deque_is_a_usable_buffer():
    """A `deque(maxlen=n_lags)` is the natural buffer for the recursion; it
    must give what an unbounded history would."""
    rng = np.random.default_rng(5)
    hist = [rng.normal(size=(6, 3)) for _ in range(11)]
    buf = deque(hist, maxlen=4)
    np.testing.assert_allclose(signal_cal(buf, 4, 0.4),
                               signal_cal(hist, 4, 0.4))


def test_sequence_of_vectors_and_of_scalars():
    rng = np.random.default_rng(6)
    vec = [rng.normal(size=3) for _ in range(4)]
    assert signal_cal(vec, 4, 0.4).shape == (3,)
    np.testing.assert_allclose(signal_cal(vec, 4, 0.4),
                               signal_cal(np.stack(vec), 4, 0.4))
    assert np.ndim(signal_cal([1.0, 2.0, 3.0], 3, 0.4)) == 0


def test_empty_sequence_raises_like_an_empty_array():
    with pytest.raises(ValueError, match='window is empty'):
        signal_cal([], 4, 0.5)


def test_short_sequence_renormalises_like_a_short_array():
    blocks = [np.ones((6, 3)), np.ones((6, 3))]
    np.testing.assert_allclose(signal_cal(blocks, 5, 0.4),
                               signal_cal(np.stack(blocks), 5, 0.4))


def test_regression_growing_history_does_not_change_the_signal():
    """The buffer may be a plain list that is only ever appended to, so the
    signal must depend on the last `n_lags` entries and nothing else --
    otherwise it drifts as the simulation walks forward.

    Only the invariance is asserted, not how it is achieved: `signal_cal`
    stacks just the tail, but stacking the whole sequence would give the same
    numbers, since the contraction slices the last `n_lags` either way. That
    is a cost difference on a long history, not a behavioural one.
    """
    rng = np.random.default_rng(7)
    tail = [rng.normal(size=(6, 3)) for _ in range(4)]
    for extra in (0, 1, 30):
        history = [rng.normal(size=(6, 3)) for _ in range(extra)] + tail
        np.testing.assert_allclose(signal_cal(history, 4, 0.4),
                                   signal_cal(tail, 4, 0.4))


def test_regression_mixed_block_shapes_raise_rather_than_broadcast():
    """History rows are `(d,)` while simulated blocks are `(n_simulation, d)`.
    Handing both over unbroadcast must fail loudly -- a silent broadcast would
    put one simulation's value behind every path."""
    with pytest.raises(ValueError):
        signal_cal([np.zeros(3), np.zeros((6, 3)), np.zeros((6, 3))], 3, 0.4)


# --------------------------------------------------------------------------
# rolling_signal, and its alignment against signal_cal
# --------------------------------------------------------------------------

def test_rolling_signal_shape_and_degenerate_input(panel):
    assert rolling_signal(panel, 4, 0.4).shape == (len(panel) - 1, 3)
    assert rolling_signal(panel[:1], 4, 0.4).shape == (0, 3)


def test_regression_row_k_stands_against_row_k_plus_one(panel):
    """Row `k` is built from `data[:k+1]`, so it predicts `data[k+1]` and never
    reads its own target. Off by one here and every simulated path is shifted a
    quarter against the value it is supposed to explain.
    """
    n_lags, theta = 4, 0.4
    roll = rolling_signal(panel, n_lags, theta)
    for k in (n_lags, 40, len(panel) - 2):
        window = panel[max(k - n_lags + 1, 0):k + 1]
        np.testing.assert_allclose(roll[k], signal_cal(window, n_lags, theta))
        np.testing.assert_allclose(roll[k],
                                   signal_cal(list(window), n_lags, theta))


def test_rolling_signal_head_rows_use_short_windows(panel):
    """The documented caveat, pinned so it is a decision and not a surprise."""
    roll = rolling_signal(panel, 4, 0.4)
    np.testing.assert_allclose(roll[0], panel[0])


# --------------------------------------------------------------------------
# lag_moments
# --------------------------------------------------------------------------

def test_lag_moments_recovers_ar1_autocorrelation():
    mom = lag_moments(ar1(200_000, 0), 4)
    np.testing.assert_allclose(mom.corr[0, 0, :], PHI ** np.arange(5),
                               atol=0.01)


def test_lag_moments_matches_pandas_pairwise_deletion():
    """Reproduces `DataFrame.corr()` on the shifted frame exactly, pairwise
    deletion and all -- the property that keeps a 20-quarter window usable."""
    X = ar1(60, 3)
    X[[5, 17, 31]] = np.nan
    n_lags = 3
    frame = pd.DataFrame({k: pd.Series(X[:, 0]).shift(k)
                          for k in range(n_lags + 1)})
    np.testing.assert_allclose(lag_moments(X, n_lags).corr[0],
                               frame.corr().to_numpy())


def test_lag_moments_counts_differ_entry_to_entry():
    X = ar1(60, 4)
    X[[5, 17, 31]] = np.nan
    n_obs = lag_moments(X, 4).n_obs[0]
    assert n_obs[0, 1] > n_obs[0, 4]


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match='need more than n_lags'):
        lag_moments(ar1(4, 0), 4)


# --------------------------------------------------------------------------
# signal_moments / conditional_moments
# --------------------------------------------------------------------------

def test_signal_moments_rejects_a_lag_moments_tuple():
    mom = lag_moments(ar1(300, 0), 4)
    with pytest.raises(TypeError, match='not a LagMoments'):
        signal_moments(mom, mom.var, 0.4)          # type: ignore[arg-type]


def test_regression_conditional_moments_takes_two_arrays_not_a_lag_moments():
    """It reads `corr` and `var` only. Demanding the whole tuple forced a
    caller whose variance is elicited rather than measured to fabricate the
    means and counts, which is exactly what `signal_moments` refuses to make
    anyone do.
    """
    mom = lag_moments(ar1(300, 0), 4)
    beta, con_var = conditional_moments(mom.corr, mom.var, 0.4)
    assert beta.shape == con_var.shape == (1,)
    # The old `(moments, theta)` call no longer type checks, and a LagMoments
    # smuggled into the `corr` slot is turned away by name.
    with pytest.raises(TypeError, match='required positional argument'):
        conditional_moments(mom, 0.4)              # type: ignore[call-arg]
    with pytest.raises(TypeError, match='not a LagMoments'):
        conditional_moments(mom, mom.var, 0.4)     # type: ignore[arg-type]


def test_conditional_moments_matches_ols_on_ar1():
    """The claim the whole module rests on: `beta` is the regression slope of
    x_t on the eq (10) signal, and eq (18) is that regression's residual
    variance. Checked against OLS, which knows nothing of either formula.
    """
    n_lags, theta = 4, 0.4
    X = ar1(200_000, 11)
    mom = lag_moments(X, n_lags)
    beta, con_var = conditional_moments(mom.corr, mom.var, theta)

    y = rolling_signal(X, n_lags, theta)[n_lags - 1:, 0]
    target = X[1:, 0][n_lags - 1:]
    design = np.column_stack([np.ones_like(y), y])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)

    assert beta[0] == pytest.approx(coef[1], rel=2e-2)
    assert con_var[0] == pytest.approx((target - design @ coef).var(), rel=2e-2)


def test_signal_moments_matches_the_realised_moments_of_the_signal():
    n_lags, theta = 4, 0.4
    X = ar1(200_000, 12)
    mom = lag_moments(X, n_lags)
    sig_var, sig_cov = signal_moments(mom.corr, mom.var, theta)
    y = rolling_signal(X, n_lags, theta)[n_lags - 1:, 0]
    target = X[1:, 0][n_lags - 1:]
    assert sig_var[0] == pytest.approx(y.var(), rel=2e-2)
    assert sig_cov[0] == pytest.approx(np.cov(target, y)[0, 1], rel=2e-2)


def test_slope_is_invariant_to_the_variance_scale():
    mom = lag_moments(ar1(2_000, 13), 4)
    base, _ = conditional_moments(mom.corr, 1.0, 0.4)
    for scale in (1e-3, 1e3):
        other, _ = conditional_moments(mom.corr, scale, 0.4)
        np.testing.assert_allclose(other, base)


def test_conditional_variance_scales_linearly_and_keeps_its_sign():
    """With `var_eli == var` the scale factors straight out of eq (18), so its
    sign is a property of `corr` and `theta` alone. This is why a positive
    semi-definite `corr` cannot produce a negative conditional variance."""
    mom = lag_moments(ar1(2_000, 14), 4)
    _, unit = conditional_moments(mom.corr, 1.0, 0.4)
    for scale in (1e-3, 7.0, 1e3):
        _, scaled = conditional_moments(mom.corr, scale, 0.4)
        np.testing.assert_allclose(scaled / scale, unit, rtol=1e-10)


def test_regression_var_eli_decoupled_from_var_can_break_cauchy_schwarz():
    """Eq (18) is guaranteed positive only when the two variances are the same
    number. Pulling them apart subtracts a term scaled by one variance from a
    different variance, with no inequality tying them -- and the result is
    reported as NaN rather than reaching a square root.
    """
    mom = lag_moments(ar1(2_000, 15), 4)
    assert np.linalg.eigvalsh(mom.corr[0]).min() > 0      # corr itself is fine
    _, ok = conditional_moments(mom.corr, mom.var, 0.4)
    assert np.isfinite(ok).all()
    _, broken = conditional_moments(mom.corr, mom.var, 0.4,
                                    var_eli=mom.var * 50.0)
    assert np.isnan(broken).all()


def test_infeasible_column_is_nan_and_does_not_take_the_panel_down():
    """One bad column must not cost the others their fit."""
    mom = lag_moments(np.hstack([ar1(2_000, 16), ar1(2_000, 17)]), 4)
    var_eli = np.array([mom.var[0], mom.var[1] * 50.0])
    beta, con_var = conditional_moments(mom.corr, mom.var, 0.4,
                                        var_eli=var_eli)
    assert np.isfinite(beta[0]) and np.isfinite(con_var[0])
    assert np.isnan(beta[1]) and np.isnan(con_var[1])


def test_non_square_corr_raises():
    with pytest.raises(ValueError, match='corr must be'):
        conditional_moments(np.zeros((3, 4)), 1.0, 0.4)


# --------------------------------------------------------------------------
# conditional_mean / sample_conditional
# --------------------------------------------------------------------------

def test_conditional_mean_is_the_unconditional_mean_when_beta_is_zero():
    """The fallback a column with no usable signal relies on."""
    np.testing.assert_allclose(
        conditional_mean(np.array([9.0, -3.0]), 0.5, 0.0), [0.5, 0.5])


def test_conditional_mean_returns_the_signal_when_beta_is_one():
    signal = np.array([9.0, -3.0])
    np.testing.assert_allclose(conditional_mean(signal, 0.5, 1.0), signal)


def test_regression_draw_is_scaled_by_the_standard_deviation():
    """`norm.ppf` returns a standard-normal quantile, so it must be multiplied
    by sigma. Multiplying by sigma**2 is invisible at variance 1 and wrong
    everywhere else -- inflating the spread above 1 and shrinking it below.
    """
    con_var = 4.0
    u = float(norm.cdf(1.0))                       # exactly one sd out
    got = sample_conditional(u, signal=0.0, mu_x=0.0, beta=0.0,
                             con_var=con_var)
    assert got == pytest.approx(np.sqrt(con_var))
    assert got != pytest.approx(con_var)


def test_draws_recover_the_conditional_mean_and_variance():
    rng = np.random.default_rng(18)
    u = rng.uniform(1e-9, 1 - 1e-9, size=400_000)
    mu_x, beta, con_var, signal = 0.3, 0.8, 0.25, 1.3
    draws = sample_conditional(u, signal, mu_x, beta, con_var)
    assert draws.mean() == pytest.approx(
        conditional_mean(signal, mu_x, beta), abs=0.01)
    assert draws.var() == pytest.approx(con_var, rel=0.02)


@pytest.mark.parametrize('bad', [0.0, 1.0, -0.1, 1.5, np.nan])
def test_uniforms_outside_the_open_unit_interval_raise(bad):
    with pytest.raises(ValueError, match='strictly inside'):
        sample_conditional(np.array([0.5, bad]), 0.0, 0.0, 0.5, 1.0)


@pytest.mark.parametrize('bad', [0.0, -1.0, np.nan, np.inf])
def test_non_positive_conditional_variance_raises(bad):
    """The guard that turns an infeasible eq (18) into a stop rather than a
    NaN quietly spreading down every simulated path."""
    with pytest.raises(ValueError, match='con_var must be positive'):
        sample_conditional(0.5, 0.0, 0.0, 0.5, bad)


def test_uniforms_broadcast_against_the_signal():
    rng = np.random.default_rng(19)
    u = rng.uniform(0.01, 0.99, size=(50, 3))
    out = sample_conditional(u, np.zeros(3), np.zeros(3), np.full(3, 0.5),
                             np.full(3, 0.2))
    assert out.shape == (50, 3)


# --------------------------------------------------------------------------
# objective_func / find_theta
# --------------------------------------------------------------------------

def test_find_theta_returns_a_float_for_one_series_and_a_series_for_a_panel(
        panel):
    assert isinstance(find_theta(panel[:, 0], 4), float)
    fitted = find_theta(pd.DataFrame(panel, columns=list('abc')), 4)
    assert isinstance(fitted, pd.Series)
    assert list(fitted.index) == list('abc')


@pytest.mark.parametrize('seed', range(5))
def test_find_theta_lands_on_a_minimiser_of_the_objective(seed):
    """A grid sweep plus golden section has to beat the grid it swept."""
    X = np.hstack([ar1(200, seed), ar1(200, seed + 50)])
    n_lags = 4
    mom = lag_moments(X, n_lags)
    fitted = find_theta(pd.DataFrame(X), n_lags).to_numpy()
    best = objective_func(fitted, mom, X, n_lags)
    for grid in np.linspace(0.0, 1.0, 21):
        other = objective_func(np.full(X.shape[1], grid), mom, X, n_lags)
        assert (best <= other + 1e-8).all()


def test_find_theta_stays_inside_the_unit_interval(panel):
    fitted = find_theta(pd.DataFrame(panel), 4).to_numpy()
    assert ((fitted >= 0.0) & (fitted <= 1.0)).all()


def test_objective_is_finite_on_clean_data(panel):
    mom = lag_moments(panel, 4)
    assert np.isfinite(objective_func(0.4, mom, panel, 4)).all()


def test_objective_is_inf_where_eq_18_is_infeasible(panel):
    """`inf` walks the search away from an infeasible theta instead of
    stopping it."""
    mom = lag_moments(panel, 4)
    score = objective_func(0.4, mom, panel, 4, var_eli=mom.var * 500.0)
    assert np.isinf(score).all()


def test_find_theta_does_not_mutate_the_caller_frame(panel):
    frame = pd.DataFrame(panel, columns=list('abc'))
    before = frame.copy()
    find_theta(frame, 4)
    pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------------
# find_theta pooled across panels
# --------------------------------------------------------------------------

def _universe(n: int = 60, k: int = 5, d: int = 2,
              start: int = 0) -> dict[str, pd.DataFrame]:
    """`k` tickers, each a `(n, d)` frame sharing one column layout."""
    return {f't{j}': pd.DataFrame(
        np.hstack([ar1(n, start + j * d + c) for c in range(d)]),
        columns=list('ab')[:d]) for j in range(k)}


def test_a_single_panel_is_detected_not_iterated(panel):
    """Iterating a DataFrame yields its column labels, so a bare frame has to
    be recognised rather than walked -- otherwise the fit would be handed
    strings."""
    frame = pd.DataFrame(panel, columns=list('abc'))
    solo = find_theta(frame, 4)
    assert isinstance(solo, pd.Series) and list(solo.index) == list('abc')
    np.testing.assert_allclose(solo.to_numpy(),
                               find_theta([frame], 4).to_numpy())


def test_one_dimensional_panel_still_returns_a_float():
    assert isinstance(find_theta(ar1(120, 0)[:, 0], 4), float)


def test_a_collection_always_returns_a_series():
    """Several panels have no single flat input to mirror."""
    fitted = find_theta([ar1(120, 0), ar1(120, 1)], 4)
    assert isinstance(fitted, pd.Series)


def test_pooled_fit_is_one_theta_per_column_not_per_panel():
    universe = _universe()
    fitted = find_theta(universe, 4)
    assert isinstance(fitted, pd.Series)
    assert list(fitted.index) == ['a', 'b']


def test_mapping_and_sequence_agree():
    """A `{ticker: frame}` dict reads its values in order, so it must give what
    the same frames in a list would."""
    universe = _universe()
    np.testing.assert_allclose(find_theta(universe, 4).to_numpy(),
                               find_theta(list(universe.values()), 4).to_numpy())


def test_pooling_one_panel_reduces_to_fitting_it_alone():
    """The pooled objective over a single panel is that panel's objective, so
    the two paths must not disagree."""
    frame = pd.DataFrame(np.hstack([ar1(120, 7), ar1(120, 8)]),
                         columns=list('ab'))
    np.testing.assert_allclose(find_theta([frame], 4).to_numpy(),
                               find_theta(frame, 4).to_numpy())


def test_pooled_fit_minimises_the_summed_objective():
    """The property that makes it a pooled fit at all: it beats every point of
    the grid it swept, scored the same weighted way."""
    universe = _universe(n=80, k=4)
    n_lags = 4
    frames = list(universe.values())
    prepared = [(f.to_numpy(), lag_moments(f, n_lags)) for f in frames]

    def pooled_score(th):
        num = np.zeros(2)
        den = np.zeros(2)
        for X, mom in prepared:
            seen = np.sum(~np.isnan(X[1:])
                          & ~np.isnan(rolling_signal(X, n_lags, 0.5)), axis=0)
            num += seen * objective_func(th, mom, X, n_lags)
            den += seen
        return num / den

    fitted = find_theta(universe, n_lags).to_numpy()
    best = pooled_score(fitted)
    for grid in np.linspace(0.0, 1.0, 21):
        assert (best <= pooled_score(np.full(2, grid)) + 1e-8).all()


def test_longer_panels_carry_more_weight():
    """Weighting by residuals, not by panel count: twenty tickers with a
    handful of quarters each must not outvote one with a long history.
    """
    n_lags = 4
    rng = np.random.default_rng(30)
    long_frame = pd.DataFrame(ar1(4_000, 40), columns=['a'])
    short = [pd.DataFrame(rng.normal(size=(n_lags + 3, 1)), columns=['a'])
             for _ in range(20)]
    alone = find_theta([long_frame], n_lags).to_numpy()
    with_noise = find_theta([long_frame] + short, n_lags).to_numpy()
    np.testing.assert_allclose(with_noise, alone, atol=0.05)


def test_panels_too_short_to_score_are_dropped_with_a_warning():
    universe = _universe(n=60, k=3, d=1)
    universe['stub'] = pd.DataFrame(np.zeros((3, 1)), columns=['a'])
    with pytest.warns(RuntimeWarning, match='cannot be scored'):
        fitted = find_theta(universe, 4)
    assert np.isfinite(fitted.to_numpy()).all()


def test_all_panels_too_short_raises():
    stub = pd.DataFrame(np.zeros((3, 1)), columns=['a'])
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match='no panel has more than'):
            find_theta([stub, stub], 4)


def test_empty_collection_raises():
    with pytest.raises(ValueError, match='nothing to fit'):
        find_theta([], 4)


def test_panels_with_different_columns_raise():
    a = pd.DataFrame(ar1(60, 0), columns=['a'])
    b = pd.DataFrame(ar1(60, 1), columns=['zzz'])
    with pytest.raises(ValueError, match='pooling needs one layout'):
        find_theta([a, b], 4)


def test_var_eli_length_must_match_the_panels():
    universe = list(_universe(k=3, d=1).values())
    with pytest.raises(ValueError, match='var_eli holds'):
        find_theta(universe, 4, var_eli=[1.0, 2.0])


def test_var_eli_is_applied_per_panel():
    """One scale per panel, in the order given -- and an infeasible one shows
    up as NaN rather than being quietly ignored."""
    universe = list(_universe(k=2, d=1).values())
    fitted = find_theta(universe, 4, var_eli=[None, None])
    assert np.isfinite(fitted.to_numpy()).all()
    huge = [np.array([5e3]), np.array([5e3])]
    assert np.isnan(find_theta(universe, 4, var_eli=huge).to_numpy()).all()


def blended(n: int, seed: int, theta_true: float = 0.5, rho: float = 0.8,
            n_lags: int = 4) -> np.ndarray:
    """`x_t = rho * (the eq (10) signal at theta_true) + noise`.

    An AR(`n_lags`) whose coefficients are the model's own weights, so theta is
    interior and identified in principle. AR(1) will not do for this: its best
    lag predictor is `x_{t-1}` alone, which puts the true theta at 1 and makes
    a boundary answer correct rather than a symptom.
    """
    w = signal_weights(n_lags, theta_true)
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 300)
    x = np.zeros(n + 300)
    for t in range(n_lags, n + 300):
        x[t] = rho * (w * x[t - n_lags:t][::-1]).sum() + e[t]
    return x[300:, None]                      # burn the transient


def _on_boundary(theta: np.ndarray) -> np.ndarray:
    return (theta <= 1e-6) | (theta >= 1 - 1e-6)


def test_the_estimator_is_consistent_on_an_interior_theta():
    """Before any claim about pooling: given enough data the fit finds the
    theta the process was built with."""
    for theta_true in (0.3, 0.5, 0.7):
        long = pd.DataFrame(blended(30_000, 1, theta_true), columns=['a'])
        assert find_theta(long, 4).iloc[0] == pytest.approx(theta_true,
                                                            abs=0.05)


def test_per_ticker_fits_pile_up_on_the_boundary_when_panels_are_short():
    """The identification problem pooling exists to answer, measured. The true
    theta is interior, so an endpoint here is the search reporting a flat
    likelihood, not a persistent characteristic.
    """
    short = [pd.DataFrame(blended(24, 300 + j), columns=['a'])
             for j in range(25)]
    per_ticker = np.array([find_theta(f, 4).iloc[0] for f in short])
    assert _on_boundary(per_ticker).mean() > 0.2


def test_pooling_cuts_the_sampling_spread_of_the_estimate():
    """What pooling actually buys: the same quantity, far less sensitive to
    which draw of the universe you happened to get. This is a variance claim,
    not an accuracy one -- see the bias test below.
    """
    universe_size = 25          # the benefit scales with it, roughly 1/sqrt(k)
    pooled, solo = [], []
    for r in range(10):
        frames = [pd.DataFrame(blended(24, 9000 + r * universe_size + j),
                               columns=['a']) for j in range(universe_size)]
        pooled.append(find_theta(frames, 4).iloc[0])
        solo.append(find_theta(frames[0], 4).iloc[0])
    assert np.std(pooled) < 0.5 * np.std(solo)
    assert not _on_boundary(np.array(pooled)).any()


def test_pooling_does_not_remove_the_short_panel_bias():
    """Honest limit of the method, pinned so nobody reads pooling as a fix for
    it. At ~20 quarters the fit sits well above the long-sample answer, and
    pooling inherits that -- it averages the bias rather than cancelling it,
    because every panel carries the same one. Only longer panels help.
    """
    truth = find_theta(pd.DataFrame(blended(30_000, 1), columns=['a']), 4)
    short = [pd.DataFrame(blended(24, 500 + j), columns=['a'])
             for j in range(20)]
    long = [pd.DataFrame(blended(400, 600 + j), columns=['a'])
            for j in range(20)]
    err_short = abs(find_theta(short, 4).iloc[0] - truth.iloc[0])
    err_long = abs(find_theta(long, 4).iloc[0] - truth.iloc[0])
    assert err_short > 0.1              # the bias is real at 24 observations
    assert err_long < err_short / 2     # and it is a length problem, not pooling


# --------------------------------------------------------------------------
# The forward recursion, as RIM_PortOp.sampling runs it
# --------------------------------------------------------------------------

def _recursion(history: np.ndarray, uniforms: np.ndarray, n_lags: int,
               theta, mu_x, beta, con_var) -> tuple[np.ndarray, list]:
    """The loop `RIM_PortOp.sampling` runs: seed with history, then for each
    period build the signal, draw, and only then append."""
    n_period, n_sim, d = uniforms.shape
    buf: list[np.ndarray] = [np.broadcast_to(row, (n_sim, d)).copy()
                             for row in history[-n_lags:]]
    signals = []
    for i in range(n_period):
        signal = signal_cal(buf, n_lags, theta)
        buf.append(np.asarray(sample_conditional(
            uniforms[i], signal, mu_x, beta, con_var)))
        signals.append(signal)
    return np.stack(buf[n_lags:]), signals


@pytest.fixture
def recursion_setup():
    rng = np.random.default_rng(20)
    n_lags, d, n_sim, n_period = 3, 2, 5, 8
    history = rng.normal(size=(10, d))
    uniforms = rng.uniform(0.001, 0.999, size=(n_period, n_sim, d))
    return dict(history=history, uniforms=uniforms, n_lags=n_lags,
                theta=np.array([0.3, 0.7]), mu_x=np.array([0.1, -0.2]),
                beta=np.array([0.6, 0.9]), con_var=np.array([0.05, 0.2]))


def test_forward_recursion_shapes(recursion_setup):
    path, signals = _recursion(**recursion_setup)
    n_period, n_sim, d = recursion_setup['uniforms'].shape
    assert path.shape == (n_period, n_sim, d)
    assert all(s.shape == (n_sim, d) for s in signals)


def test_regression_forward_recursion_agrees_with_rolling_signal(
        recursion_setup):
    """The independent check: rebuild each simulation's full series after the
    fact and ask `rolling_signal` -- which shares no code path with the
    recursion -- what signal stood against every drawn value.
    """
    path, signals = _recursion(**recursion_setup)
    history = recursion_setup['history']
    n_lags, theta = recursion_setup['n_lags'], recursion_setup['theta']
    n_period, n_sim, _ = recursion_setup['uniforms'].shape
    for s in range(n_sim):
        full = np.vstack([history, path[:, s, :]])
        roll = rolling_signal(full, n_lags, theta)
        for i in range(n_period):
            np.testing.assert_allclose(signals[i][s],
                                       roll[len(history) + i - 1])


def test_regression_first_period_uses_history_not_an_unconditional_draw(
        recursion_setup):
    """Eq (10) reads realised x, so quarter 0 has a signal of its own. Drawing
    it unconditionally instead discards the serial dependence at the boundary
    and leaves the first `n_lags` quarters wrong.
    """
    setup = dict(recursion_setup)
    _, signals = _recursion(**setup)
    n_lags, theta = setup['n_lags'], setup['theta']
    expected = signal_cal(setup['history'][-n_lags:], n_lags, theta)
    np.testing.assert_allclose(signals[0][0], expected)
    # ...and it is genuinely informative, not a constant standing in for one.
    assert not np.allclose(signals[0][0], setup['mu_x'])


def test_regression_a_period_never_reads_its_own_draw(recursion_setup):
    """Signal first, append after. Perturbing the uniforms of period `i` may
    move every later signal but must leave signals 0..i untouched.
    """
    setup = dict(recursion_setup)
    _, base = _recursion(**setup)
    perturbed = setup['uniforms'].copy()
    perturbed[3] = 0.5
    setup['uniforms'] = perturbed
    _, other = _recursion(**setup)
    for i in range(4):
        np.testing.assert_allclose(base[i], other[i])
    assert not np.allclose(base[4], other[4])


def test_seed_rows_are_dropped_from_the_result(recursion_setup):
    """The buffer starts with history; the result must hold simulation only."""
    path, _ = _recursion(**recursion_setup)
    history = recursion_setup['history']
    n_sim = recursion_setup['uniforms'].shape[1]
    for row in history[-recursion_setup['n_lags']:]:
        assert not np.allclose(path[0], np.broadcast_to(row, (n_sim,) + row.shape))


def test_recursion_reproduces_the_conditional_law(recursion_setup):
    """Across many simulations the first period must match eqs (17) and (18)
    exactly, since its signal is a constant fixed by history."""
    setup = dict(recursion_setup)
    rng = np.random.default_rng(21)
    setup['uniforms'] = rng.uniform(1e-9, 1 - 1e-9, size=(2, 200_000, 2))
    path, signals = _recursion(**setup)
    expected = conditional_mean(signals[0][0], setup['mu_x'], setup['beta'])
    np.testing.assert_allclose(path[0].mean(axis=0), expected, atol=0.01)
    np.testing.assert_allclose(path[0].var(axis=0), setup['con_var'], rtol=0.02)


def test_unfittable_column_falls_back_to_its_unconditional_law():
    """What `RIM_PortOp.sampling` substitutes when `find_theta` returns NaN:
    beta = 0 leaves the mean at mu_x, and the theta standing in for the NaN is
    irrelevant precisely because beta is 0 -- but it must be finite, since
    0*NaN is still NaN.
    """
    rng = np.random.default_rng(22)
    n_lags, d, n_sim = 3, 2, 4
    theta = np.where([False, True], 1.0, np.array([0.4, np.nan]))
    assert np.isfinite(theta).all()
    setup = dict(history=rng.normal(size=(6, d)),
                 uniforms=rng.uniform(0.001, 0.999, size=(3, n_sim, d)),
                 n_lags=n_lags, theta=theta, mu_x=np.array([0.1, -0.2]),
                 beta=np.array([0.6, 0.0]), con_var=np.array([0.05, 0.2]))
    path, _ = _recursion(**setup)
    assert np.isfinite(path).all()
    # the fallback column is centred on mu_x regardless of any signal
    draws = path[:, :, 1].ravel()
    assert draws.mean() == pytest.approx(setup['mu_x'][1], abs=0.5)
