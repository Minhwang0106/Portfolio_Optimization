"""Ledoit-Wolf (2004) shrinkage of the sample covariance towards constant correlation.

The covariance estimator Bielstein & Hanauer (2019) pair with their
implied-cost-of-capital means: Ledoit & Wolf, "Honey, I shrunk the sample
covariance matrix", *Journal of Portfolio Management* 30(4), 2004, 110-119.
The target keeps every sample variance and sets every correlation to the
average sample correlation; the weight on it is the paper's estimate of the
optimal shrinkage intensity, `kappa / T` clipped to [0, 1], with
`kappa = (pi - rho) / gamma`.

Not scikit-learn's `LedoitWolf`, which is the same authors' other 2004
estimator (*Journal of Multivariate Analysis*), shrinking towards a scaled
identity. Over a single equity market the correlations are most of what an
optimiser has to work with, and the identity target discards them. That is also
what this module used to call, on an (N, T) array -- which estimated the T x T
covariance between months rather than the one between stocks.
"""
import numpy as np


def ledoit_wolf_cc (returns: np.ndarray)-> tuple[np.ndarray, float]:
    """Covariance matrix shrunk towards constant correlation.

    Normalised by T rather than T - 1, as in the paper. The difference is one
    common factor, which moves no maximum-Sharpe portfolio.

    Args:
        returns (np.ndarray): Complete panel, shape (T, N): observations down
            the rows, one column per asset, no NaN.

    Returns:
        tuple[np.ndarray, float]: The (N, N) shrunk covariance matrix, and the
            intensity delta in [0, 1] -- the weight on the target.

    Raises:
        ValueError: If the panel is not 2-D, has fewer than two observations or
            two assets, contains NaN, or has a column with no variance, which
            has no correlation to put in the target.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> sigma, delta = ledoit_wolf_cc(rng.normal(size=(60, 5)))
        >>> sigma.shape, bool(0 <= delta <= 1)
        ((5, 5), True)
    """
    x: np.ndarray = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f'returns must be 2-D (T, N), got shape {x.shape}')
    n_obs, n_asset = x.shape
    if n_obs < 2 or n_asset < 2:
        raise ValueError(f'need two observations on at least two assets, got '
                         f'shape {x.shape}')
    if np.isnan(x).any():
        raise ValueError('returns contain NaN; the estimator needs a complete '
                         'panel')
    x = x-x.mean(axis=0)
    sample: np.ndarray = x.T@x/n_obs
    var: np.ndarray = np.diag(sample).copy()
    if (var <= 0).any():
        raise ValueError(f'{int((var <= 0).sum())} column(s) have no variance')
    sd: np.ndarray = np.sqrt(var)

    # The target: sample variances on the diagonal, the average off-diagonal
    # sample correlation everywhere else.
    r_bar: float = float(((sample/np.outer(sd, sd)).sum()-n_asset)
                         /(n_asset*(n_asset-1)))
    prior: np.ndarray = r_bar*np.outer(sd, sd)
    np.fill_diagonal(prior, var)

    # pi: the summed asymptotic variances of the sample covariances,
    # pi_ij = mean_t[(x_it x_jt - s_ij)^2].
    sq: np.ndarray = x**2
    pi_mat: np.ndarray = sq.T@sq/n_obs-sample**2
    pi_hat: float = float(pi_mat.sum())

    # rho: the summed asymptotic covariances between target and sample entries.
    # The target's diagonal is the sample's, which contributes its own pi; off
    # it, theta_ii,ij = mean_t[(x_it^2 - s_ii)(x_it x_jt - s_ij)].
    theta: np.ndarray = (x**3).T@x/n_obs-var[:, None]*sample
    np.fill_diagonal(theta, 0.0)
    rho_hat: float = float(np.trace(pi_mat)
                           +r_bar*((sd[None, :]/sd[:, None])*theta).sum())

    # gamma: how far the target sits from the sample.
    gamma_hat: float = float(((sample-prior)**2).sum())
    if gamma_hat <= 0:
        # The sample already has constant correlation, as it always does at
        # N = 2: target and sample coincide and there is nothing to choose.
        return sample, 0.0
    delta: float = float(np.clip((pi_hat-rho_hat)/gamma_hat/n_obs, 0.0, 1.0))
    return delta*prior+(1-delta)*sample, delta
