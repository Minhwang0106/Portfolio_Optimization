"""Is one mean return really higher than another?

The third of the three objectives this project reports, alongside
`sharpe_inference` and `ce_inference`, and built the same way on purpose: a
delta method over sample means, a prewhitened QS HAC estimate of their long-run
covariance, and a studentized paired circular block bootstrap. `_psi_hac`,
`_block_indices` and `_psi_boot` come from `sharpe_inference` unchanged; they
were never specific to which moments they were handed, only to how many.

What is estimated. On the pair of plain means `v = (mu_x, mu_b)`,

    g(v) = v_0 - v_1,      grad g = (1, -1)

which is the degenerate case of the same machinery: the statistic is *linear*,
so the delta method is exact rather than a first-order approximation and the
gradient carries no estimated quantities. Everything still reduces to `Psi`, the
2x2 long-run covariance of `(R_xt, R_bt)`, and `s(Delta) = sqrt(grad' Psi grad
/ T)` collapses to `sqrt((Psi_00 - 2 Psi_01 + Psi_11)/T)` -- the long-run
variance of the paired difference, which is exactly what it should be.

That last identity is worth reading twice, because it is the reason a mean test
can be written two ways. Estimating `Psi` from the pair and contracting it with
`(1, -1)`, as here, and running the scalar HAC directly on `D_t = R_xt - R_bt`
are the same estimator asymptotically. They differ a little in finite samples --
a VAR(1) prewhitening filter against an AR(1) one, a bandwidth read off two
series against one, `T/(T-2)` against `T/(T-1)` -- and the pair route is taken
so that all three tests in one summary table are built from an identically
constructed `Psi`. A reader comparing `sharpe_pval`, `ce_pval` and `mean_pval`
should not have to wonder whether the difference between them is the statistic
or the plumbing. `tests/test_mean_inference.py` pins that the two routes agree
to well inside their own standard error.

**The two strategies are paired, and stay paired.** Same months, same market, so
`Psi`'s off-diagonal is most of the answer rather than a nuisance:
`SE = sqrt(SE_x^2 + SE_b^2)` would throw away the contemporaneous covariance and
inflate the standard error on two overlapping books, burying a real difference.
The bootstrap draws one set of block indices and applies it to both series for
the same reason.

**Excess or total returns, it does not matter here.** The risk-free rate cancels
in `R_xt - R_bt`, so unlike the Sharpe test (which needs excess) and the CE test
(which needs total, since CRRA utility is defined on what an investor actually
earns) this one is indifferent to the convention. Pass either, consistently.

**Both p-values are trustworthy here, which is not true of the other two.** A
linear statistic is the friendliest case for the normal approximation: there is
no non-linear transformation to skew the sampling distribution, so the only
thing left to get wrong is the HAC estimate of `Psi`. That shows up in the
level checks. On the heavy-tailed, persistent process that drove the CE test's
HAC to 0.18-0.20 against a nominal 0.10 -- t3 innovations, 9% monthly vol,
AR(1) at 0.2, the profile of a concentrated book -- this test measures 0.110 for
HAC and 0.100 for the bootstrap, and on moderate tails 0.100 and 0.083.
`tests/test_mean_inference.py` pins it. `pval_boot` is still the one to quote,
for consistency with the two companions and because it is the more robust
construction, but here the choice barely moves the answer.

The default is *two-sided*, matching both companions. `alternative='greater'`
tests `H0: mu_x <= mu_b` for a genuinely directional prior, and `'less'`
reverses it.

**A warning about what this test does and does not say.** The mean difference is
not risk-adjusted and not scale-invariant: lever any strategy up and its mean
difference against a fixed benchmark grows without its statistical significance
being penalised for the volatility that came with it. That is not a defect of
the test, it is the definition of the quantity, and it is precisely why the
Sharpe and CE tests exist. On this backtest the point is not hypothetical --
the proposed model runs at roughly twice `equal_weight`'s volatility, and the
mean test is the only one of the three that rejects. Report the three together
and say which question each answers; a significant mean difference alongside an
insignificant Sharpe difference is a finding about leverage, not about skill.
"""
import math
import numpy as np
import pandas as pd
from .sharpe_inference import _psi_hac, _block_indices, _psi_boot
# Shared with `ce_inference` rather than duplicated -- both files offer the same
# three alternatives and count the same tail; only the statistic differs.
from .ce_inference import _ALTERNATIVE, _tail_count

# `v = (mu_x, mu_b)`: one mean per strategy, and the pairing is the point.
_N_MOMENT: int = 2
MONTHS_PER_YEAR: int = 12
# `grad g` is constant, which is what makes the delta method exact here.
_GRADIENT: np.ndarray = np.array([1.0, -1.0])


def _delta (v: np.ndarray)-> float:
    """`g(v)`: the difference of the two mean returns."""
    return float(v[0]-v[1])


def _standard_error (psi: np.ndarray, n_obs: int)-> float:
    """`s(Delta) = sqrt(grad' Psi grad / T)`.

    With `grad = (1, -1)` this is the long-run variance of the paired
    difference, `Psi_00 - 2 Psi_01 + Psi_11`, over `T`. Written as the quadratic
    form anyway so the parallel with `sharpe_inference` and `ce_inference` is
    visible rather than something the reader has to reconstruct.
    """
    quad: float = float(_GRADIENT@psi@_GRADIENT)
    return math.sqrt(max(quad, 0.0)/n_obs)


def _centred (x: np.ndarray, b: np.ndarray, v: np.ndarray)-> np.ndarray:
    """`y_t = (R_xt - mu_x, R_bt - mu_b)`, centred at `v`.

    `v` need not be `x` and `b`'s own means -- the bootstrap centres its
    resamples at the *original* sample's moments, which is what makes the
    studentized statistic a pivot.
    """
    return np.column_stack([x-v[0], b-v[1]])


def _normal_pvalue (z: float, alternative: str)-> float:
    """Asymptotic p-value from `z ~ N(0, 1)`.

    `2*Phi(-|z|)` and `Phi(-z)` written with `erfc`, which is what `Phi` is
    implemented from anyway and saves a scipy import for one scalar.
    """
    if alternative == 'two-sided':
        return math.erfc(abs(z)/math.sqrt(2))
    if alternative == 'greater':
        return 0.5*math.erfc(z/math.sqrt(2))
    return 0.5*math.erfc(-z/math.sqrt(2))


def _bootstrap_pvalue (x: np.ndarray, b: np.ndarray, delta: float,
                       se_hac: float, block: int, n_boot: int,
                       rng: np.random.Generator, alternative: str,
                       null: float = 0.0)-> float:
    """Studentized circular block bootstrap p-value for the mean difference.

    The same construction as its two companions, and the same reason for it:
    each resample carries its own standard error, so the p-value is the tail
    probability of the pivot `(Delta* - Delta)/s(Delta*)` rather than of an
    unstudentized `Delta*` that would inherit whatever finite-sample skew the
    test is trying to correct for.

    `delta` is the point estimate and `null` the value under test. They are
    separate because the resamples are always centred at the point estimate --
    that is what makes the statistic a pivot -- while the numerator measures the
    distance from the null.

    Note:
        Blocks are drawn once and applied to `x` and `b` together, so every
        resample preserves the month-by-month pairing of the two strategies.
    """
    n_obs: int = len(x)
    v_hat: np.ndarray = np.array([x.mean(), b.mean()])
    idx: np.ndarray = _block_indices(n_obs, block, n_boot, rng)[:, :n_obs]
    # One index array, both series -- the pairing is preserved by construction.
    x_star: np.ndarray = x[idx]
    b_star: np.ndarray = b[idx]

    delta_star: np.ndarray = x_star.mean(axis=1)-b_star.mean(axis=1)
    y_star: np.ndarray = np.stack([x_star-v_hat[0], b_star-v_hat[1]], axis=2)
    psi_star: np.ndarray = _psi_boot(y_star, block)
    quad: np.ndarray = np.einsum('a,kab,b->k', _GRADIENT, psi_star, _GRADIENT)
    se_star: np.ndarray = np.sqrt(np.maximum(quad, 0.0)/n_obs)

    # A resample can be one block repeated, in which case the difference has no
    # variation and the pivot is undefined. It carries no information about the
    # tail, so it is dropped rather than allowed to become infinite.
    ok: np.ndarray = se_star > 0
    if not ok.any():
        return float('nan')
    stat: float = ((delta-null)/se_hac if se_hac > 0
                   else math.inf*np.sign(delta-null))
    centred: np.ndarray = (delta_star[ok]-delta)/se_star[ok]
    return (_tail_count(stat, centred, alternative)+1)/(int(ok.sum())+1)


def mean_difference_test (returns_x: pd.Series, returns_b: pd.Series,
                          block_size: int = 5, n_boot: int = 4999,
                          prewhite: bool = True, seed: int = 0,
                          periods_per_year: int = MONTHS_PER_YEAR,
                          alternative: str = 'two-sided')-> pd.Series:
    """Test that two mean returns are equal, robustly.

    Delta method over the pair of means, a prewhitened QS HAC long-run
    covariance, and a studentized paired circular block bootstrap. See the
    module docstring for the construction, for why `pval_boot` is the number to
    read, and -- more important than either -- for what a rejection here does
    not license you to say.

    Args:
        returns_x (pd.Series): The strategy's monthly returns. Excess or total,
            as long as `returns_b` uses the same convention: the risk-free rate
            cancels in the difference, so this test is the one place in the
            three where the choice does not change the answer.
        returns_b (pd.Series): The benchmark's, aligned on the shared index.
            Months missing from either side are dropped from both, which is what
            keeps the pair paired.
        block_size (int): Circular block length for the bootstrap. Defaults to
            5, matching `sharpe_inference` and `ce_inference` so the three tests
            in one table are not silently using different dependence
            assumptions.
        n_boot (int): Bootstrap resamples. Defaults to 4999. 0 skips the
            bootstrap and returns NaN for `pval_boot`, leaving only HAC.
        prewhite (bool): VAR(1)-prewhiten the HAC estimate. Defaults to True.
        seed (int): Seeds the resampling, so a rerun reproduces the p-value.
            Defaults to 0.
        periods_per_year (int): Scales `mean_diff` and `mean_diff_se` to annual
            by simple multiplication -- the arithmetic mean is linear in the
            horizon, so unlike the CE there is no compounding to fold in and
            unlike the Sharpe ratio there is no `sqrt`. The p-values are
            invariant to it. Defaults to 12; pass 1 for the monthly figure.
        alternative (str): `'two-sided'` tests `H0: mu_x = mu_b` and is the
            default, matching the other two tests; `'greater'` tests
            `H0: mu_x <= mu_b` against `H1: mu_x > mu_b`, and `'less'` reverses
            it.

    Returns:
        pd.Series: `mean_x`, `mean_b`, `mean_diff`, `mean_diff_se` (all on the
            `periods_per_year` scale, so annualised by default), `t_hac` -- the
            statistic the HAC p-value is read from, reported because a
            t-statistic is what a reader of a performance table expects to
            scan -- `pval_hac`, `pval_boot`, and `n_month`.

    Raises:
        ValueError: If `alternative` is not one of `'greater'`, `'less'`,
            `'two-sided'`, if fewer than `_N_MOMENT+2` months overlap, or if
            `block_size` is outside `[1, T]`.

    Example:
        >>> mean_difference_test(proposed_returns, ew_returns)[
        ...     ['mean_diff', 'pval_boot']]
        mean_diff    0.1868
        pval_boot    0.0038
        dtype: float64
    """
    if alternative not in _ALTERNATIVE:
        raise ValueError(f'alternative must be one of {_ALTERNATIVE}; '
                         f'got {alternative!r}')

    joined: pd.DataFrame = pd.concat(
        {'x': returns_x, 'b': returns_b}, axis=1).dropna()
    if len(joined) < _N_MOMENT+2:
        raise ValueError(f'need at least {_N_MOMENT+2} overlapping months; '
                         f'got {len(joined)}')
    x: np.ndarray = joined['x'].to_numpy(dtype=float)
    b: np.ndarray = joined['b'].to_numpy(dtype=float)
    n_obs: int = len(joined)

    v_hat: np.ndarray = np.array([x.mean(), b.mean()])
    delta: float = _delta(v_hat)
    psi: np.ndarray = _psi_hac(_centred(x, b, v_hat), prewhite=prewhite)
    se_hac: float = _standard_error(psi, n_obs)

    t_hac: float = delta/se_hac if se_hac > 0 else float('nan')
    pval_hac: float = (_normal_pvalue(t_hac, alternative)
                       if se_hac > 0 else float('nan'))

    pval_boot: float = float('nan')
    if n_boot > 0:
        if block_size < 1 or block_size > n_obs:
            raise ValueError(f'block_size must be in [1, {n_obs}]; '
                             f'got {block_size}')
        pval_boot = _bootstrap_pvalue(x, b, delta, se_hac, block_size, n_boot,
                                      np.random.default_rng(seed), alternative)

    scale: float = float(periods_per_year)
    return pd.Series({
        'mean_x': float(v_hat[0])*scale, 'mean_b': float(v_hat[1])*scale,
        'mean_diff': delta*scale, 'mean_diff_se': se_hac*scale,
        't_hac': t_hac, 'pval_hac': pval_hac, 'pval_boot': pval_boot,
        'n_month': float(n_obs)})
