"""Is one Sharpe ratio really higher than another, or is that the sample?

Ledoit, O. and Wolf, M. (2008), *Robust performance hypothesis testing with the
Sharpe ratio*, Journal of Empirical Finance 15, 850-859. Implemented here for
the difference between a strategy and a benchmark; `metrics.summarise` runs it
against `equal_weight`.

The test everybody reaches for -- Jobson-Korkie (1981), corrected by Memmel
(2003) -- assumes the two return series are i.i.d. bivariate normal. That
assumption enters as a specific `Omega` in the delta method, one whose lower
right block is `2 sigma^4` and whose off-diagonal blocks are zero. Both are
wrong for returns: the first needs `E[(r-mu)^4] = 3 sigma^4`, i.e. no excess
kurtosis, and the second needs mean and variance estimates to be asymptotically
uncorrelated, which fails under skew. Serial dependence breaks it again, since
the upper left entry is then `sigma^2 + 2 sum_l Cov(r_t, r_{t+l})` and not
`sigma^2`. The paper's simulations put the JK test's rejection rate at 22.5% for
a nominal 10% test on t6-VAR data. It is not a matter of a few percent.

What is estimated. With `mu` the mean and `gamma = E[r^2]` the uncentered second
moment of each series, the Sharpe difference is `f(v)` at
`v = (mu_x, mu_b, gamma_x, gamma_b)`, where

    f(a, b, c, d) = a/sqrt(c-a^2) - b/sqrt(d-b^2)

-- the paper's Eq. (2), written on uncentered moments precisely so that `v` is a
vector of plain means and the delta method applies to it. Its gradient, Eq. (4),
gives `s(Delta) = sqrt(grad' Psi grad / T)`, so everything reduces to estimating
`Psi`, the long-run covariance of those four means. Two ways to do that, and
they are the paper's two methods:

* **HAC** (`method='hac'`). A kernel estimate of `Psi`, prewhitened QS with
  Andrews' automatic bandwidth, then a normal p-value. Asymptotically valid and
  cheap. It is also *liberal in finite samples* -- the paper's own simulations
  put it at 12.4% for a nominal 10% test on t6-VAR data, and this backtest's
  T=132 is squarely in the range where that bites. Reported, but not the number
  to quote.
* **Bootstrap** (`method='boot'`, the default). The studentized circular block
  bootstrap of Section 3.2.2, which held 9.7-10.5% against that same nominal 10%
  across every data-generating process the paper tried, i.i.d. and time series
  alike. This is the one to read.

Studentizing is what makes the bootstrap worth the trouble: resampling `Delta`
itself would inherit the same finite-sample skew, so each resample gets *its
own* standard error, `d* = |Delta* - Delta| / s(Delta*)`, and the p-value is the
tail probability of that pivot, Eq. (9). Note the resampling is from the
observed data, not from a null-restricted distribution -- the null enters by
centering at `Delta`, which is why no artificial "make the Sharpe ratios equal"
transformation is needed.

Two places where this file departs from a literal reading of the paper, both
stated because the results move if you disagree:

* **`Psi*` is scaled by `1/l`, not the `1/T` printed in Section 3.2.2.** With
  `zeta_j` a block sum already divided by `sqrt(b)`, `1/T` is off by a factor of
  `b`: for i.i.d. data the estimator has to converge to the sample covariance,
  and `(1/l) sum zeta zeta'` does while `(1/T) sum zeta zeta'` converges to
  `1/b` of it. The two agree at `b=1`, which is the only case the paper's
  footnote 9 checks. `test_sharpe_inference.py` pins this by simulation.
* **The block size defaults to a fixed 5, not to the calibration of Algorithm
  3.1.** The algorithm is implemented, in `calibrate_block_size`, but it costs
  `K * len(grid) * n_boot` resamples and is not something to pay on every
  backtest. 5 sits between the `b=4` and `b=6` the paper's own two applications
  calibrated to at a comparable `T`; call `calibrate_block_size` when the
  p-value is near a threshold you care about.
"""
import math
import warnings
import numpy as np
import pandas as pd

# The 4-vector v whose estimation the paper's T/(T-4) factor offsets.
_N_MOMENT: int = 4
_QS_CONST: float = 1.3221    # Andrews (1991), table I, QS kernel.
_AR_CAP: float = 0.97        # Prewhitening cap, Andrews-Monahan (1992).
_MIN_BANDWIDTH: float = 1e-8
# `var = gamma - mu^2` is a difference of two nearly equal numbers for a series
# with little variation, so a flat one lands on a tiny positive float rather
# than on zero and has to be caught relative to the second moment.
_VAR_TOL: float = 1e-12
BLOCK_GRID: tuple[int, ...] = (1, 2, 4, 6, 8, 10)  # Section 3.2.2's own grid.


def _qs_kernel (x: np.ndarray)-> np.ndarray:
    """Quadratic spectral kernel, `k(0) = 1`.

    Andrews' (1991) optimal kernel, and the one Section 3.2.2 asks for. It has
    unbounded support, so unlike Bartlett or Parzen every lag gets a weight;
    that is what makes it worth the trigonometry.
    """
    out: np.ndarray = np.ones_like(x, dtype=float)
    nz: np.ndarray = x != 0
    z: np.ndarray = 6*np.pi*x[nz]/5
    out[nz] = 25/(12*np.pi**2*x[nz]**2)*(np.sin(z)/z-np.cos(z))
    return out


def _qs_bandwidth (y: np.ndarray)-> float:
    """Andrews' (1991) automatic bandwidth for the QS kernel.

    Fits a univariate AR(1) to each column and reads the bandwidth off the
    implied `alpha(2)`. The AR coefficients are capped at `_AR_CAP` because the
    formula divides by `(1-rho)^8`, which a unit root sends to infinity.
    """
    n_obs: int = len(y)
    num: float = 0.0
    den: float = 0.0
    for col in y.T:
        lag: np.ndarray = col[:-1]
        denom: float = float(lag@lag)
        rho: float = float(lag@col[1:])/denom if denom > 0 else 0.0
        rho = float(np.clip(rho, -_AR_CAP, _AR_CAP))
        resid: np.ndarray = col[1:]-rho*lag
        sig2: float = float(resid@resid)/len(resid)
        num += 4*rho**2*sig2**2/(1-rho)**8
        den += sig2**2/(1-rho)**4
    alpha2: float = num/den if den > 0 else 0.0
    return max(_QS_CONST*(alpha2*n_obs)**0.2, _MIN_BANDWIDTH)


def _prewhiten (y: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """Strip VAR(1) dependence before kernel smoothing; return residuals and `A`.

    Andrews-Monahan (1992). A kernel estimate on strongly autocorrelated data is
    badly biased at any feasible bandwidth, so the persistence is removed
    parametrically, the kernel handles what is left, and the estimate is recoloured
    by `(I-A)^-1`. `A`'s singular values are capped at `_AR_CAP` so the recolouring
    cannot blow up on a near-unit-root fit.
    """
    lag: np.ndarray = y[:-1]
    lead: np.ndarray = y[1:]
    coef: np.ndarray = np.linalg.lstsq(lag, lead, rcond=None)[0].T
    u, sv, vt = np.linalg.svd(coef)
    coef = (u*np.clip(sv, -_AR_CAP, _AR_CAP))@vt
    return lead-lag@coef.T, coef


def _psi_hac (y: np.ndarray, prewhite: bool = True)-> np.ndarray:
    """Long-run covariance of the four moment estimates, Section 3.1.

    Args:
        y (np.ndarray): `(T, 4)` of the paper's `y_t`, already centred.
        prewhite (bool): Apply the VAR(1) filter. Defaults to True, the paper's
            HAC_pw variant.

    Returns:
        np.ndarray: `(4, 4)` estimate of `Psi`, including the `T/(T-4)`
            small-sample factor.
    """
    n_obs: int = len(y)
    resid: np.ndarray = y
    coef: np.ndarray|None = None
    if prewhite and n_obs > _N_MOMENT+1:
        resid, coef = _prewhiten(y)

    bandwidth: float = _qs_bandwidth(resid)
    n_resid: int = len(resid)
    psi: np.ndarray = resid.T@resid/n_resid
    for lag_j in range(1, n_resid):
        weight: float = float(_qs_kernel(np.array([lag_j/bandwidth]))[0])
        # The QS kernel decays but never truncates, so stop once it stops
        # contributing rather than summing all T-1 lags for nothing.
        if abs(weight) < 1e-10:
            continue
        gamma: np.ndarray = resid[lag_j:].T@resid[:-lag_j]/n_resid
        psi = psi+weight*(gamma+gamma.T)

    if coef is not None:
        recolour: np.ndarray = np.linalg.inv(np.eye(_N_MOMENT)-coef)
        psi = recolour@psi@recolour.T
    return psi*n_obs/(n_obs-_N_MOMENT)


def _moments (x: np.ndarray, b: np.ndarray)-> np.ndarray:
    """`v = (mu_x, mu_b, gamma_x, gamma_b)`, means and uncentered second moments."""
    return np.array([x.mean(), b.mean(), (x**2).mean(), (b**2).mean()])


def _degenerate (v: np.ndarray)-> bool:
    """Is either series too flat for its Sharpe ratio to mean anything?"""
    return bool(v[2]-v[0]**2 <= abs(v[2])*_VAR_TOL
                or v[3]-v[1]**2 <= abs(v[3])*_VAR_TOL)


def _delta (v: np.ndarray)-> float:
    """`f(v)`, Eq. (2): the difference of the two Sharpe ratios."""
    var_x: float = v[2]-v[0]**2
    var_b: float = v[3]-v[1]**2
    return v[0]/np.sqrt(var_x)-v[1]/np.sqrt(var_b)


def _gradient (v: np.ndarray)-> np.ndarray:
    """`grad f(v)`, Eq. (4)."""
    var_x: float = v[2]-v[0]**2
    var_b: float = v[3]-v[1]**2
    return np.array([v[2]/var_x**1.5, -v[3]/var_b**1.5,
                     -0.5*v[0]/var_x**1.5, 0.5*v[1]/var_b**1.5])


def _standard_error (v: np.ndarray, psi: np.ndarray, n_obs: int)-> float:
    """`s(Delta)`, Eq. (5). Zero variance in `psi` gives 0, not a NaN."""
    grad: np.ndarray = _gradient(v)
    quad: float = float(grad@psi@grad)
    return math.sqrt(max(quad, 0.0)/n_obs)


def _centred (x: np.ndarray, b: np.ndarray, v: np.ndarray)-> np.ndarray:
    """The paper's `y_t`, centred at `v` -- which need not be `x` and `b`'s own moments."""
    return np.column_stack([x-v[0], b-v[1], x**2-v[2], b**2-v[3]])


def _block_indices (n_obs: int, block: int, n_draw: int,
                    rng: np.random.Generator)-> np.ndarray:
    """Circular block bootstrap indices, `(n_draw, n_obs)`.

    Circular rather than moving blocks (Politis-Romano 1992) so that every
    observation is drawn with equal probability -- which is what lets `_psi_boot`
    centre at the original sample's moments instead of having to correct for the
    edge effect that moving blocks introduce.
    """
    n_block: int = -(-n_obs//block)
    starts: np.ndarray = rng.integers(0, n_obs, size=(n_draw, n_block))
    idx: np.ndarray = (starts[:, :, None]+np.arange(block)[None, None, :])%n_obs
    return idx.reshape(n_draw, -1)


def _psi_boot (y: np.ndarray, block: int)-> np.ndarray:
    """`Psi*` from one resample's block structure, Section 3.2.2.

    Args:
        y (np.ndarray): `(n_draw, T, 4)` of `y*_t`, centred at the *original*
            sample's moments.
        block (int): The block length the resample was built from.

    Returns:
        np.ndarray: `(n_draw, 4, 4)`.

    Note:
        Scaled by `1/l` over the `l = floor(T/b)` complete blocks, not the `1/T`
        the paper prints. See the module docstring.
    """
    n_obs: int = y.shape[1]
    n_full: int = n_obs//block
    blocks: np.ndarray = y[:, :n_full*block].reshape(
        len(y), n_full, block, _N_MOMENT)
    zeta: np.ndarray = blocks.sum(axis=2)/math.sqrt(block)
    return np.einsum('kja,kjb->kab', zeta, zeta)/n_full


def _bootstrap_pvalue (x: np.ndarray, b: np.ndarray, delta: float,
                       se_hac: float, block: int, n_boot: int,
                       rng: np.random.Generator, null: float = 0.0)-> float:
    """Studentized bootstrap p-value, Eq. (9).

    `delta` is the point estimate and `null` the value under test. They are
    separate because the resamples are always centred at the point estimate --
    that is what makes the statistic a pivot -- while the numerator measures the
    distance from the null. `calibrate_block_size` needs a non-zero `null`.
    """
    n_obs: int = len(x)
    v_hat: np.ndarray = _moments(x, b)
    idx: np.ndarray = _block_indices(n_obs, block, n_boot, rng)[:, :n_obs]
    x_star: np.ndarray = x[idx]
    b_star: np.ndarray = b[idx]

    v_star: np.ndarray = np.stack(
        [x_star.mean(axis=1), b_star.mean(axis=1),
         (x_star**2).mean(axis=1), (b_star**2).mean(axis=1)], axis=1)
    var_x: np.ndarray = v_star[:, 2]-v_star[:, 0]**2
    var_b: np.ndarray = v_star[:, 3]-v_star[:, 1]**2
    # A resample can be degenerate -- one block repeated, zero variance. It
    # carries no information about the tail, so it is dropped rather than
    # allowed to produce an infinite studentized statistic.
    ok: np.ndarray = (var_x > 0)&(var_b > 0)
    if not ok.any():
        return float('nan')

    y_star: np.ndarray = np.stack(
        [x_star-v_hat[0], b_star-v_hat[1],
         x_star**2-v_hat[2], b_star**2-v_hat[3]], axis=2)
    psi_star: np.ndarray = _psi_boot(y_star, block)

    delta_star: np.ndarray = (v_star[:, 0]/np.sqrt(var_x)
                              -v_star[:, 1]/np.sqrt(var_b))
    grad: np.ndarray = np.stack(
        [v_star[:, 2]/var_x**1.5, -v_star[:, 3]/var_b**1.5,
         -0.5*v_star[:, 0]/var_x**1.5, 0.5*v_star[:, 1]/var_b**1.5], axis=1)
    quad: np.ndarray = np.einsum('ka,kab,kb->k', grad, psi_star, grad)
    se_star: np.ndarray = np.sqrt(np.maximum(quad, 0.0)/n_obs)

    ok = ok&(se_star > 0)
    stat: float = abs(delta-null)/se_hac if se_hac > 0 else float('inf')
    centred: np.ndarray = np.abs(delta_star[ok]-delta)/se_star[ok]
    return (float((centred >= stat).sum())+1)/(int(ok.sum())+1)


def sharpe_difference_test (excess_x: pd.Series, excess_b: pd.Series,
                            block_size: int = 5, n_boot: int = 4999,
                            prewhite: bool = True, seed: int = 0,
                            periods_per_year: int = 12)-> pd.Series:
    """Test that two Sharpe ratios are equal, robustly. Ledoit-Wolf (2008).

    Both p-values test `H0: SR_x - SR_b = 0` two-sided. Read `pval_boot`; see the
    module docstring for why `pval_hac` is reported but is the weaker of the two.

    Args:
        excess_x (pd.Series): The strategy's excess returns, already net of the
            risk-free rate. Aligned against `excess_b` on the shared index.
        excess_b (pd.Series): The benchmark's excess returns.
        block_size (int): Circular block length for the bootstrap. Defaults to 5;
            `calibrate_block_size` picks it from the data instead.
        n_boot (int): Bootstrap resamples, the paper's `M`. Defaults to 4999, as
            in its own applications. 0 skips the bootstrap and returns NaN for
            `pval_boot`, leaving only HAC.
        prewhite (bool): VAR(1)-prewhiten the HAC estimate. Defaults to True.
        seed (int): Seeds the resampling, so a rerun reproduces the p-value.
            Defaults to 0.
        periods_per_year (int): Scales `sr_diff` and `sr_diff_se` to annual by
            `sqrt`. The p-values are invariant to it. Defaults to 12.

    Returns:
        pd.Series: `sr_diff` (annualised `SR_x - SR_b`), `sr_diff_se` (its
            annualised HAC standard error), `pval_hac`, `pval_boot`, and
            `n_month` used.

    Raises:
        ValueError: If fewer than `_N_MOMENT+2` months overlap, or if either
            series has zero variance over them.

    Example:
        >>> sharpe_difference_test(epo_excess, ew_excess)[['sr_diff', 'pval_boot']]
        sr_diff     -0.0237
        pval_boot    0.8836
        dtype: float64
    """
    joined: pd.DataFrame = pd.concat(
        {'x': excess_x, 'b': excess_b}, axis=1).dropna()
    if len(joined) < _N_MOMENT+2:
        raise ValueError(f'need at least {_N_MOMENT+2} overlapping months; '
                         f'got {len(joined)}')
    x: np.ndarray = joined['x'].to_numpy(dtype=float)
    b: np.ndarray = joined['b'].to_numpy(dtype=float)
    n_obs: int = len(x)

    v_hat: np.ndarray = _moments(x, b)
    if _degenerate(v_hat):
        raise ValueError('a return series has zero variance; '
                         'its Sharpe ratio is undefined')

    delta: float = _delta(v_hat)
    psi: np.ndarray = _psi_hac(_centred(x, b, v_hat), prewhite=prewhite)
    se_hac: float = _standard_error(v_hat, psi, n_obs)
    # 2*Phi(-|d|) written with erfc, which is what Phi is implemented from
    # anyway and saves a scipy import for one scalar.
    pval_hac: float = (math.erfc(abs(delta)/se_hac/math.sqrt(2))
                       if se_hac > 0 else float('nan'))

    pval_boot: float = float('nan')
    if n_boot > 0:
        if block_size < 1 or block_size > n_obs:
            raise ValueError(f'block_size must be in [1, {n_obs}]; '
                             f'got {block_size}')
        pval_boot = _bootstrap_pvalue(x, b, delta, se_hac, block_size, n_boot,
                                      np.random.default_rng(seed))

    scale: float = math.sqrt(periods_per_year)
    return pd.Series({'sr_diff': delta*scale, 'sr_diff_se': se_hac*scale,
                      'pval_hac': pval_hac, 'pval_boot': pval_boot,
                      'n_month': float(n_obs)})


def _var1_pseudo (x: np.ndarray, b: np.ndarray, n_seq: int,
                  rng: np.random.Generator, avg_block: int = 5)-> np.ndarray:
    """Pseudo return series from the semi-parametric model of Algorithm 3.1.

    A VAR(1) fitted to the pair, driven by its own residuals resampled with the
    stationary bootstrap of Politis-Romano (1994). The parametric part carries
    the linear dependence; resampling the residuals rather than drawing normals
    is what keeps the tails and any leftover non-linear structure.

    Returns:
        np.ndarray: `(n_seq, T, 2)`.
    """
    data: np.ndarray = np.column_stack([x, b])
    n_obs: int = len(data)
    lag: np.ndarray = data[:-1]
    coef: np.ndarray = np.linalg.lstsq(lag, data[1:], rcond=None)[0].T
    u, sv, vt = np.linalg.svd(coef)
    coef = (u*np.clip(sv, -_AR_CAP, _AR_CAP))@vt
    resid: np.ndarray = data[1:]-lag@coef.T
    resid = resid-resid.mean(axis=0)
    mean: np.ndarray = data.mean(axis=0)

    # Stationary bootstrap: geometric block lengths, i.e. restart the index with
    # probability 1/avg_block at each step and otherwise walk forward.
    n_resid: int = len(resid)
    restart: np.ndarray = rng.random((n_seq, n_obs)) < 1/avg_block
    restart[:, 0] = True
    draw: np.ndarray = rng.integers(0, n_resid, size=(n_seq, n_obs))
    idx: np.ndarray = np.empty((n_seq, n_obs), dtype=int)
    idx[:, 0] = draw[:, 0]
    for t in range(1, n_obs):
        idx[:, t] = np.where(restart[:, t], draw[:, t],
                             (idx[:, t-1]+1)%n_resid)
    shocks: np.ndarray = resid[idx]

    out: np.ndarray = np.empty((n_seq, n_obs, 2))
    state: np.ndarray = np.tile(data[0]-mean, (n_seq, 1))
    for t in range(n_obs):
        state = state@coef.T+shocks[:, t]
        out[:, t] = state+mean
    return out


def calibrate_block_size (excess_x: pd.Series, excess_b: pd.Series,
                          grid: tuple[int, ...] = BLOCK_GRID,
                          alpha: float = 0.05, n_pseudo: int = 100,
                          n_boot: int = 499, seed: int = 0)-> tuple[int, pd.Series]:
    """Pick the block size by the coverage calibration of Algorithm 3.1.

    Fits a semi-parametric model to the observed pair, generates pseudo
    sequences from it, and for each candidate block size measures how often the
    bootstrap interval covers the observed `Delta` -- treated as the truth,
    which is what makes this computable at all. The winner is the block size
    whose realised coverage is closest to `1-alpha`.

    Costs `n_pseudo * len(grid) * n_boot` resamples, so it is deliberately not
    what `sharpe_difference_test` does by default.

    Args:
        excess_x (pd.Series): The strategy's excess returns.
        excess_b (pd.Series): The benchmark's excess returns.
        grid (tuple[int, ...]): Candidate block sizes. Defaults to the paper's
            own `(1, 2, 4, 6, 8, 10)`.
        alpha (float): Nominal level whose coverage is targeted. Defaults to 0.05.
        n_pseudo (int): Pseudo sequences, the paper's `K`. Defaults to 100 to
            keep this callable; the paper asks for at least 1000 on real data
            and the choice does wobble between neighbouring grid points below
            that, so raise it before quoting a calibrated block size.
        n_boot (int): Resamples per interval. Defaults to 499.
        seed (int): Seeds both layers. Defaults to 0.

    Returns:
        tuple[int, pd.Series]: The chosen block size, and realised coverage
            indexed by candidate -- worth looking at, since a flat curve means
            the choice does not matter and a monotone one means the grid is too
            narrow.

    Example:
        >>> block, coverage = calibrate_block_size(ppp_excess, ew_excess)
        >>> block
        6
    """
    joined: pd.DataFrame = pd.concat(
        {'x': excess_x, 'b': excess_b}, axis=1).dropna()
    x: np.ndarray = joined['x'].to_numpy(dtype=float)
    b: np.ndarray = joined['b'].to_numpy(dtype=float)
    rng: np.random.Generator = np.random.default_rng(seed)
    truth: float = _delta(_moments(x, b))
    pseudo: np.ndarray = _var1_pseudo(x, b, n_pseudo, rng)

    coverage: dict[int, float] = {}
    for block in grid:
        covered: int = 0
        for seq in pseudo:
            px, pb = seq[:, 0], seq[:, 1]
            v_seq: np.ndarray = _moments(px, pb)
            if _degenerate(v_seq):
                continue
            d_seq: float = _delta(v_seq)
            psi_seq: np.ndarray = _psi_hac(_centred(px, pb, v_seq))
            se_seq: float = _standard_error(v_seq, psi_seq, len(px))
            if se_seq <= 0:
                continue
            # The interval covers `truth` exactly when a test of `truth` would
            # not reject, so the p-value machinery answers this directly and
            # there is no interval to build.
            pval: float = _bootstrap_pvalue(px, pb, d_seq, se_seq, block,
                                            n_boot, rng, null=truth)
            covered += int(pval > alpha)
        coverage[block] = covered/len(pseudo)

    result: pd.Series = pd.Series(coverage, name='coverage')
    result.index.name = 'block_size'
    if result.isna().all():
        warnings.warn('block size calibration produced no usable coverage; '
                      'falling back to 5', RuntimeWarning)
        return 5, result
    return int((result-(1-alpha)).abs().idxmin()), result
