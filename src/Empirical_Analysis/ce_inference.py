"""Is one CRRA certainty equivalent really higher than another?

The companion to `sharpe_inference` for the other objective this project reports
-- `metrics.crra_certainty_equivalent`, the number `PPP` and `port_optimize` are
actually fitted on. Same three ingredients as the Sharpe test, and deliberately
the same code for two of them: a delta method over sample means, a prewhitened
QS HAC estimate of their long-run covariance, and a studentized circular block
bootstrap. `_psi_hac`, `_block_indices` and `_psi_boot` are imported from
`sharpe_inference` rather than reimplemented -- they were never specific to which
moments they were handed, only to how many, and that is now read off the array.

What is estimated. With `gamma != 1` the CRRA utility of a monthly simple return
`R` is a power, so put

    X_it = (1 + R_it)^(1-gamma)

and the whole statistic is a smooth function of the *plain mean* of that:

    CE_i = mu_i^(k/(1-gamma)) - 1,      mu_i = E[X_it]

where `k` is the number of months the CE is quoted over -- `k=1` monthly, `k=12`
for the annualised figure `metrics.crra_certainty_equivalent` reports and that
the summary table prints. The annualisation is inside the exponent rather than
applied afterwards because compounding the monthly CE,
`(1 + (mu^(1/(1-g)) - 1))^12 - 1`, collapses to exactly `mu^(12/(1-g)) - 1`. That
matters for inference, not just tidiness: it keeps the estimator a single smooth
transformation of one mean, so the delta method covers the annualised number
directly instead of covering the monthly one and leaving the compounding step
un-propagated.

The difference of two such CEs, on the pair `v = (mu_A, mu_B)`, is

    h(v) = v_0^(k/(1-g)) - v_1^(k/(1-g))

with gradient `(c*v_0^(c-1), -c*v_1^(c-1))` at `c = k/(1-gamma)`, giving
`s(Delta) = sqrt(grad' Psi grad / T)` exactly as in the Sharpe case. Everything
again reduces to estimating `Psi`, here the 2x2 long-run covariance of
`(X_At, X_Bt)`.

**The two strategies are paired, and stay paired.** They are evaluated on the
same months, so `Psi`'s off-diagonal is not a nuisance term -- it is most of the
answer. Two strategies holding overlapping books move together, and
`SE = sqrt(SE_A^2 + SE_B^2)` would ignore that and inflate the standard error,
turning a real difference into an insignificant one. The bootstrap resamples
`(X_At, X_Bt)` as pairs inside one set of blocks for the same reason; resampling
the two series independently would destroy the very dependence the covariance
term is there to capture.

**Read `pval_boot`, not `pval_hac`.** Both are reported and the HAC one is
needed anyway to studentize, but the asymptotic normal p-value is the same
estimator this project already found liberal for the Sharpe difference at this
sample size -- Ledoit-Wolf's own simulations put it at 12.4% for a nominal 10%
test on heavy-tailed time-series data, and `tests/test_sharpe_inference.py`
reproduces it. Nothing about the certainty equivalent makes that better. If
anything it is worse: at `gamma=5` the exponent `k/(1-gamma)` is -3, so `X` is a
*negative* power of the gross return and the transformation puts its long left
tail exactly where a bad month is, while the concentrated books in this backtest
(`avg_weight_entropy` near 0.03 for the proposed model, i.e. `exp(entropy)` of
about one name) deliver monthly returns with close to single-stock tails. The
bootstrap held its nominal level across every process the paper tried and is
what `metrics` reports.

The default is *two-sided*, matching `sharpe_inference`. Both tests then answer
the same shape of question -- does this strategy's objective differ from the
benchmark's -- so a summary table carrying both is not quietly mixing a
directional test with a symmetric one, which is the kind of asymmetry a reader
has to be told about and would reasonably ask why. It is also the conservative
choice: a one-sided test at the same nominal level rejects on half the evidence,
and doing that only for the metric the proposed model happens to lead on is
hard to defend after the fact. `alternative='greater'` tests
`H0: CE_x <= CE_b` for a genuinely directional prior, and `'less'` reverses it.
"""
import math
import numpy as np
import pandas as pd
from constant import risk_aversion
from .sharpe_inference import _psi_hac, _block_indices, _psi_boot

# `v = (mu_A, mu_B)`: one mean per strategy, and the pairing is the point.
_N_MOMENT: int = 2
MONTHS_PER_YEAR: int = 12
_ALTERNATIVE: tuple[str, ...] = ('greater', 'less', 'two-sided')
# `mu` is a mean of strictly positive powers, so it cannot legitimately reach 0;
# a value this small means the series is degenerate rather than merely small.
_MU_TOL: float = 1e-300


def _utility (returns: np.ndarray, gamma: float)-> np.ndarray:
    """`X_t = (1+R_t)^(1-gamma)`, the quantity whose mean the CE is built on."""
    return (1.0+returns)**(1.0-gamma)


def _exponent (gamma: float, periods: int)-> float:
    """`c = periods/(1-gamma)`, the power that takes `mu` to `1+CE`.

    `periods=12` folds the annualisation into the exponent; see the module
    docstring for why that is done here rather than by compounding afterwards.
    """
    return periods/(1.0-gamma)


def _ce (mu: np.ndarray|float, c: float)-> np.ndarray|float:
    """`CE = mu^c - 1`."""
    return mu**c-1.0


def _delta (v: np.ndarray, c: float)-> float:
    """`h(v)`: the difference of the two certainty equivalents.

    The `-1` in each CE cancels, so this is a difference of two powers.
    """
    return float(v[0]**c-v[1]**c)


def _gradient (v: np.ndarray, c: float)-> np.ndarray:
    """`grad h(v) = (c*mu_A^(c-1), -c*mu_B^(c-1))`.

    The sign on the second entry is the whole reason the cross-covariance
    subtracts rather than adds in `_standard_error`.
    """
    return np.array([c*v[0]**(c-1.0), -c*v[1]**(c-1.0)])


def _standard_error (v: np.ndarray, psi: np.ndarray, n_obs: int,
                     c: float)-> float:
    """`s(Delta) = sqrt(grad' Psi grad / T)`, the delta method with HAC `Psi`."""
    grad: np.ndarray = _gradient(v, c)
    quad: float = float(grad@psi@grad)
    return math.sqrt(max(quad, 0.0)/n_obs)


def _centred (x: np.ndarray, b: np.ndarray, v: np.ndarray)-> np.ndarray:
    """`y_t = (X_At - mu_A, X_Bt - mu_B)`, centred at `v`.

    `v` need not be `x` and `b`'s own means -- the bootstrap centres its
    resamples at the *original* sample's moments, which is what makes the
    studentized statistic a pivot.
    """
    return np.column_stack([x-v[0], b-v[1]])


def _tail_count (stat: float, centred: np.ndarray, alternative: str)-> int:
    """How many resampled statistics are at least as extreme as `stat`."""
    if alternative == 'two-sided':
        return int((np.abs(centred) >= abs(stat)).sum())
    if alternative == 'greater':
        return int((centred >= stat).sum())
    return int((centred <= stat).sum())


def _bootstrap_pvalue (x: np.ndarray, b: np.ndarray, delta: float,
                       se_hac: float, c: float, block: int, n_boot: int,
                       rng: np.random.Generator, alternative: str,
                       null: float = 0.0)-> float:
    """Studentized circular block bootstrap p-value for the CE difference.

    The same construction as `sharpe_inference._bootstrap_pvalue`, and the same
    reason for it: resampling `Delta` itself would inherit the finite-sample
    skew the test is trying to correct for, so each resample carries its own
    standard error and the p-value is the tail probability of the pivot
    `(Delta* - Delta)/s(Delta*)`.

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

    v_star: np.ndarray = np.stack([x_star.mean(axis=1), b_star.mean(axis=1)],
                                  axis=1)
    # `X` is a strictly positive power of a strictly positive gross return, so
    # a resample mean can only fail to be positive through underflow.
    ok: np.ndarray = (v_star[:, 0] > _MU_TOL)&(v_star[:, 1] > _MU_TOL)
    if not ok.any():
        return float('nan')

    y_star: np.ndarray = np.stack([x_star-v_hat[0], b_star-v_hat[1]], axis=2)
    psi_star: np.ndarray = _psi_boot(y_star, block)

    delta_star: np.ndarray = v_star[:, 0]**c-v_star[:, 1]**c
    grad: np.ndarray = np.stack([c*v_star[:, 0]**(c-1.0),
                                 -c*v_star[:, 1]**(c-1.0)], axis=1)
    quad: np.ndarray = np.einsum('ka,kab,kb->k', grad, psi_star, grad)
    se_star: np.ndarray = np.sqrt(np.maximum(quad, 0.0)/n_obs)

    ok = ok&(se_star > 0)
    if not ok.any():
        return float('nan')
    stat: float = ((delta-null)/se_hac if se_hac > 0
                   else math.inf*np.sign(delta-null))
    centred: np.ndarray = (delta_star[ok]-delta)/se_star[ok]
    return (_tail_count(stat, centred, alternative)+1)/(int(ok.sum())+1)


def ce_difference_test (returns_x: pd.Series, returns_b: pd.Series,
                        gamma: float = risk_aversion,
                        block_size: int = 5, n_boot: int = 4999,
                        prewhite: bool = True, seed: int = 0,
                        periods_per_year: int = MONTHS_PER_YEAR,
                        alternative: str = 'two-sided')-> pd.Series:
    """Test that two CRRA certainty equivalents are equal, robustly.

    Delta method over the pair of utility means, a prewhitened QS HAC long-run
    covariance, and a studentized paired circular block bootstrap. See the
    module docstring for the construction and for why `pval_boot` is the number
    to read.

    Args:
        returns_x (pd.Series): The strategy's monthly *simple total* returns --
            not excess returns. CRRA utility is defined on the gross return an
            investor actually earns, so subtracting the risk-free rate here
            would evaluate a portfolio nobody held. This is the same series
            `metrics.crra_certainty_equivalent` is given.
        returns_b (pd.Series): The benchmark's, aligned on the shared index.
            Months missing from either side are dropped from both, which is what
            keeps the pair paired.
        gamma (float): Relative risk aversion. Must not be 1 -- log utility is a
            removable singularity this parameterisation does not cover, the same
            restriction `metrics.crra_certainty_equivalent` carries. Defaults to
            `constant.risk_aversion`.
        block_size (int): Circular block length for the bootstrap. Defaults to
            5, matching `sharpe_inference`'s default so the two tests in one
            table are not silently using different dependence assumptions.
        n_boot (int): Bootstrap resamples. Defaults to 4999. 0 skips the
            bootstrap and returns NaN for `pval_boot`, leaving only HAC.
        prewhite (bool): VAR(1)-prewhiten the HAC estimate. Defaults to True.
        seed (int): Seeds the resampling, so a rerun reproduces the p-value.
            Defaults to 0.
        periods_per_year (int): Months the CE is quoted over. Defaults to 12,
            i.e. the annualised figure the summary table reports. Pass 1 for the
            monthly CE. The p-values are *not* invariant to this -- unlike the
            Sharpe ratio's `sqrt` scaling, the exponent is inside a non-linear
            transformation -- though in practice they move very little.
        alternative (str): `'two-sided'` tests `H0: CE_x = CE_b` and is the
            default, matching `sharpe_inference`; `'greater'` tests
            `H0: CE_x <= CE_b` against `H1: CE_x > CE_b`, and `'less'` reverses
            it. See the module docstring on why the symmetric test is the
            default here.

    Returns:
        pd.Series: `ce_x`, `ce_b`, `ce_diff` (all on the `periods_per_year`
            scale, so annualised by default and directly comparable with
            `ann_crra_ce` in the summary table), `ce_diff_se`, `pval_hac`,
            `pval_boot`, and `n_month`.

    Raises:
        ValueError: If `gamma` is 1, if `alternative` is not one of
            `'greater'`, `'less'`, `'two-sided'`, if fewer than `_N_MOMENT+2`
            months overlap, if `block_size` is outside `[1, T]`, or if either
            series has a month at -100% or worse, where CRRA utility is
            undefined.

    Example:
        >>> ce_difference_test(proposed_returns, ew_returns)[['ce_diff', 'pval_boot']]
        ce_diff      0.0274
        pval_boot    0.7228
        dtype: float64
    """
    if gamma == 1:
        raise ValueError('gamma must not be 1; log utility is not covered')
    if alternative not in _ALTERNATIVE:
        raise ValueError(f'alternative must be one of {_ALTERNATIVE}; '
                         f'got {alternative!r}')

    joined: pd.DataFrame = pd.concat(
        {'x': returns_x, 'b': returns_b}, axis=1).dropna()
    if len(joined) < _N_MOMENT+2:
        raise ValueError(f'need at least {_N_MOMENT+2} overlapping months; '
                         f'got {len(joined)}')
    r_x: np.ndarray = joined['x'].to_numpy(dtype=float)
    r_b: np.ndarray = joined['b'].to_numpy(dtype=float)
    # Checked rather than returned as NaN: `crra_certainty_equivalent` already
    # reports NaN for such a series, so reaching here with one means the caller
    # is testing a number that was never computed in the first place.
    if (1.0+r_x <= 0).any() or (1.0+r_b <= 0).any():
        raise ValueError('a month is at -100% or worse; CRRA utility is '
                         'undefined there and the certainty equivalent with it')
    n_obs: int = len(joined)

    x: np.ndarray = _utility(r_x, gamma)
    b: np.ndarray = _utility(r_b, gamma)
    c: float = _exponent(gamma, periods_per_year)
    v_hat: np.ndarray = np.array([x.mean(), b.mean()])

    delta: float = _delta(v_hat, c)
    psi: np.ndarray = _psi_hac(_centred(x, b, v_hat), prewhite=prewhite)
    se_hac: float = _standard_error(v_hat, psi, n_obs, c)

    if se_hac > 0:
        z: float = delta/se_hac
        # 2*Phi(-|z|) and Phi(-z) written with erfc, which is what Phi is
        # implemented from anyway and saves a scipy import for one scalar.
        if alternative == 'two-sided':
            pval_hac: float = math.erfc(abs(z)/math.sqrt(2))
        elif alternative == 'greater':
            pval_hac = 0.5*math.erfc(z/math.sqrt(2))
        else:
            pval_hac = 0.5*math.erfc(-z/math.sqrt(2))
    else:
        pval_hac = float('nan')

    pval_boot: float = float('nan')
    if n_boot > 0:
        if block_size < 1 or block_size > n_obs:
            raise ValueError(f'block_size must be in [1, {n_obs}]; '
                             f'got {block_size}')
        pval_boot = _bootstrap_pvalue(x, b, delta, se_hac, c, block_size,
                                      n_boot, np.random.default_rng(seed),
                                      alternative)

    return pd.Series({
        'ce_x': float(_ce(v_hat[0], c)), 'ce_b': float(_ce(v_hat[1], c)),
        'ce_diff': delta, 'ce_diff_se': se_hac,
        'pval_hac': pval_hac, 'pval_boot': pval_boot,
        'n_month': float(n_obs)})
