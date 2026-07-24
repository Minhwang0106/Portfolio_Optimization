import numpy as np
from constant import risk_aversion

def crra_utility (r:np.ndarray, risk_aversion:float = risk_aversion):
    """Average CRRA (constant relative risk aversion) utility of a return path.

    Applies `u(r) = (1+r)**(1-gamma) / (1-gamma)` elementwise and averages over
    every element, which is the sample analogue of the expected utility the
    parametric portfolio policy maximises (Brandt-Santa-Clara-Valkanov 2009,
    eq. 3).

    Args:
        r (np.ndarray): Realised portfolio returns as simple (not log) decimals,
            one per period. Any shape; the mean is taken over the flattened
            array.
        risk_aversion (float): The gamma coefficient. Must not equal 1 (the
            log-utility case is a removable singularity this formula does not
            handle). Defaults to `constant.risk_aversion` (5.0).

    Returns:
        np.float64: Mean utility. Negative for gamma > 1, and closer to zero is
            better.

    Example:
        >>> crra_utility(np.array([0.01, -0.02, 0.03]), risk_aversion=5.0)
        np.float64(-0.24446941...)
    """
    utility: np.ndarray = (1+r)**(1-risk_aversion)/(1-risk_aversion)
    return np.mean(utility)

def ppp_weight (theta:np.ndarray, benchmark: np.ndarray, 
                char: np.ndarray):
    """Tilt a benchmark portfolio by the standardized characteristics.

    Implements the policy `w_it = w_bar_it + (1/N_t) * theta' * x_hat_it`: each
    stock's benchmark weight is nudged by its characteristic exposure, scaled by
    `1/N_t` so the tilt does not grow with the size of the universe. Because the
    characteristics are cross-sectionally standardized, the tilts sum to zero
    and the output still sums to one.

    Args:
        theta (np.ndarray): Policy coefficients, shape `(n_char,)`, one per
            characteristic in the same column order as `char`.
        benchmark (np.ndarray): Benchmark weights, shape `(1, n_t)` -- equal
            weighting `np.ones((1, n_t)) / n_t` in this project.
        char (np.ndarray): Standardized characteristics, either a single
            cross-section `(n_t, n_char)` or a stacked panel
            `(n_period, n_t, n_char)`. The stock axis is always `-2`.

    Returns:
        np.ndarray: Portfolio weights, shape `(1, n_t)` for a cross-section or
            `(n_period, n_t)` for a panel. Each row sums to one, and individual
            weights may be negative (the policy is unconstrained).

    Example:
        >>> theta = np.array([0.3, 0.0, 0.0])
        >>> benchmark = np.ones((1, 3)) / 3
        >>> char = np.array([[1., 0., 0.],
        ...                  [0., 1., 0.],
        ...                  [-1., -1., 0.]])
        >>> ppp_weight(theta, benchmark, char)
        array([[0.43333333, 0.33333333, 0.23333333]])
    """
    # Stock axis is -2, never 1: `char` arrives as a (n_t, n_char) cross-section
    # from portfolio_weight but as a (n_period, n_t, n_char) panel from
    # find_theta. shape[1] is n_t only in the 3-D case; in the 2-D case it is
    # the characteristic count, which would scale every tilt by n_t/n_char.
    n_t:int = char.shape[-2]
    return benchmark+char@theta/n_t

def ppp_op (theta:np.ndarray, benchmark: np.ndarray, 
                char: np.ndarray,r_arr:np.ndarray):
    """Objective function handed to `scipy.optimize.minimize` to fit theta.

    Forms the policy weights for every period in the estimation window, earns
    them against the realised returns, and returns the *negative* average CRRA
    utility -- so minimising this maximises expected utility.

    Args:
        theta (np.ndarray): Policy coefficients, shape `(n_char,)`. This is the
            variable `minimize` searches over.
        benchmark (np.ndarray): Benchmark weights, shape `(1, n_t)`;
            broadcast across every period of the panel.
        char (np.ndarray): Standardized characteristics panel, shape
            `(n_period, n_t, n_char)`.
        r_arr (np.ndarray): Realised returns over the same window, shape
            `(n_period, n_t)`, aligned column-for-column with `char`'s stock
            axis.

    Returns:
        np.float64: Negative mean CRRA utility of the policy's return series.

    Example:
        >>> theta = np.array([0.3, 0.0, 0.0])
        >>> benchmark = np.ones((1, 3)) / 3
        >>> char = np.array([[[1., 0., 0.],
        ...                   [0., 1., 0.],
        ...                   [-1., -1., 0.]]])          # 1 period, 3 stocks
        >>> r_arr = np.array([[0.02, -0.01, 0.03]])
        >>> ppp_op(theta, benchmark, char, r_arr)
        np.float64(0.23803...)

        Fitting it:

        >>> from scipy.optimize import minimize
        >>> minimize(ppp_op, theta, args=(benchmark, char, r_arr)).x
        array([...])
    """
    weight: np.ndarray = ppp_weight(theta,benchmark,char)
    r_vec: np.ndarray = np.sum(weight*r_arr, axis=1)
    return -crra_utility(r_vec)