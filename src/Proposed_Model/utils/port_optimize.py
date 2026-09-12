import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ...PPP.utils import crra_utility
from ...ICC_MVO.utils import max_sharpe_weight
from constant import risk_aversion


def objective_func (weight: np.ndarray,return_arr:np.ndarray,
                    risk_aversion:float=risk_aversion):
    """Negative mean CRRA utility of the portfolio these weights imply.

    Args:
        weight (np.ndarray): Shape (n_tickers,).
        return_arr (np.ndarray): Shape (n_simulation, n_tickers) -- simulations
            down the rows and assets across the columns, so `return_arr @ weight`
            gives one portfolio return per simulation.
        risk_aversion (float): The gamma coefficient. Defaults to
            `constant.risk_aversion`.

    Returns:
        np.float64: Negated, so that `minimize` maximises utility.
    """
    port_return: np.ndarray = return_arr@weight
    return -crra_utility(port_return,risk_aversion)

def objective_grad (weight: np.ndarray, return_arr: np.ndarray,
                    risk_aversion: float = risk_aversion)-> np.ndarray:
    """Gradient of `objective_func` with respect to the weights.

    `u'(r) = (1+r)^(-gamma)`, so the gradient of mean utility is each asset's
    simulated return weighted by the portfolio's marginal utility in that
    simulation. Handed to `minimize` so SLSQP does not difference the objective
    once per asset per step -- at 350 names that is 350 extra products of a
    10,000 x 350 matrix every iteration.

    Args:
        weight (np.ndarray): Shape (n_tickers,).
        return_arr (np.ndarray): Shape (n_simulation, n_tickers), as
            `objective_func`.
        risk_aversion (float): The gamma coefficient. Defaults to
            `constant.risk_aversion`.

    Returns:
        np.ndarray: Shape (n_tickers,), negated as the objective is.
    """
    port_return: np.ndarray = return_arr@weight
    marginal: np.ndarray = (1+port_return)**(-risk_aversion)
    return -(marginal@return_arr)/len(port_return)

def _usable_simulations (return_arr: np.ndarray|pd.DataFrame
                         )-> tuple[pd.Index, np.ndarray, np.ndarray]:
    """The simulated returns an optimiser can use, and where they came from.

    Args:
        return_arr (np.ndarray | pd.DataFrame): As `port_weight` takes it --
            indexed by ticker with one column per simulation.

    Returns:
        tuple[pd.Index, np.ndarray, np.ndarray]: `(labels, keep, matrix)` --
            every ticker label, the mask of the ones with any finite
            simulation, and those tickers' returns as (n_simulation, n_kept)
            over the simulations finite for all of them.

    Raises:
        ValueError: If no ticker is usable, or if no simulation is priced across
            every usable ticker.

    Note:
        Non-finite entries are removed rather than passed through.
        `joint_return` produces them for degenerate inputs, and a single NaN
        makes the whole objective NaN -- on which `minimize` does not fail, it
        reports success and hands back the equal-weight starting guess.
    """
    if isinstance(return_arr,pd.DataFrame):
        labels: pd.Index = return_arr.index
        values: np.ndarray = return_arr.to_numpy(dtype=float)
    else:
        values = np.asarray(return_arr,dtype=float)
        labels = pd.RangeIndex(values.shape[0])

    keep: np.ndarray = np.isfinite(values).any(axis=1)
    if not keep.any():
        raise ValueError('every ticker is non-finite across all simulations')
    usable: np.ndarray = values[keep]
    # Simulations are dropped whole rather than per ticker, so the survivors are
    # still compared on one common set of states.
    shared: np.ndarray = np.isfinite(usable).all(axis=0)
    if not shared.any():
        raise ValueError('no simulation is finite across every usable ticker')
    usable = usable[:,shared]

    # (n_simulation, n_tickers). `joint_return` hands back the transpose of
    # this, so unpacking its shape directly read the simulation count as the
    # number of assets -- 200 weights for a 273-name universe.
    return labels, keep, usable.T

def port_weight (return_arr:np.ndarray|pd.DataFrame,
                 risk_aversion:float=risk_aversion,
                 long_only: bool= True,
                 n_eff: float|None = None)->pd.Series:
    """Weights maximising expected CRRA utility over the simulated returns.

    Args:
        return_arr (np.ndarray | pd.DataFrame): Simulated returns as
            `RIM_PortOp.joint_return` returns them -- indexed by ticker with one
            column per simulation. A bare array is read the same way, as
            (n_tickers, n_simulation), and transposed internally since the
            objective contracts over simulations.
        risk_aversion (float): The gamma coefficient. Defaults to
            `constant.risk_aversion`.
        long_only (bool): Bound weights to [0, 1] rather than leaving them free.
            Defaults to True.
        n_eff (float | None): Smallest effective number of names the book may
            have, `1 / sum(w^2)`, imposed as `sum(w^2) <= 1 / n_eff` -- convex
            for a long-only book summing to one, and satisfied by the
            equal-weight starting point. None leaves breadth free, the model as
            specified. At or above the number of usable tickers the only
            feasible book is equal weight, which is returned without solving.
            Defaults to None.

    Returns:
        pd.Series: One weight per ticker, indexed by ticker when the input
            carried labels and positionally otherwise, summing to 1. A ticker
            dropped for non-finite simulations gets 0.

    Raises:
        ValueError: If no ticker is usable, if no simulation is priced across
            every usable ticker, if `n_eff` is given for a long-short book, or
            if the optimiser does not converge.

    Note:
        The effective-N floor is what matched-breadth comparisons hold this
        model to: without it the objective sees little risk in the simulated
        implied returns and concentrates on one or two names.
    """
    labels, keep, matrix = _usable_simulations(return_arr)
    N_s, n_tickers = matrix.shape

    constrains: list[dict] = [{'type':'eq', 'fun': lambda w: np.sum(w)-1,
                               'jac': lambda w: np.ones_like(w)}]
    if n_eff is not None:
        if not long_only:
            raise ValueError('the effective-N floor is only defined here for '
                             'the long-only book')
        if n_eff >= n_tickers:
            # Equal weight is the only book this broad, and SLSQP does not
            # reliably land on a feasible set squeezed down to one point.
            out: pd.Series = pd.Series(0.0,index=labels)
            out.loc[labels[keep]] = 1.0/n_tickers
            return out
        max_hhi: float = 1.0/float(n_eff)
        constrains.append({'type':'ineq', 'fun': lambda w: max_hhi-w@w,
                           'jac': lambda w: -2.0*w})

    if long_only:
        bounds: list[tuple] = [(0,1) for _ in range(n_tickers)]
    else:
        bounds: list[tuple] = [(None,None) for _ in range(n_tickers)]
    weight: np.ndarray = np.ones(shape=(n_tickers))/n_tickers

    res = minimize(objective_func, weight, args=(matrix, risk_aversion),
                   jac=objective_grad, method='SLSQP',
                   constraints=constrains, bounds=bounds,
                   options={'maxiter': 1000})
    if not res.success:
        raise ValueError(f'optimiser did not converge: {res.message}')

    # The whole vector: `res.x[0]` is one asset's weight, not the portfolio.
    out: pd.Series = pd.Series(0.0,index=labels)
    out.loc[labels[keep]] = res.x
    return out

def moment_port_weight (return_arr: np.ndarray|pd.DataFrame, rf: float = 0.0,
                        n_eff: float|None = None)-> pd.Series:
    """Maximum Sharpe ratio on the simulation's first two moments.

    Bielstein and Hanauer's portfolio rule (`ICC_MVO.utils.max_sharpe_weight`)
    applied to this model's distribution instead of to a point estimate: the
    mean of each ticker's simulated implied returns less `rf`, and the
    covariance of the same simulations. Everything above the second moment is
    discarded, which is the point. Beside `port_weight` on the same draws, the
    gap is what the rest of the distribution is worth to a CRRA investor;
    beside B&H, it is what fundamentals-derived moments are worth against a
    point estimate paired with a price-history covariance.

    No weight cap and no dust floor. Breadth is held by `n_eff` instead, and a
    floor-and-renormalise step afterwards would nudge the book off it.

    Args:
        return_arr (np.ndarray | pd.DataFrame): As `port_weight`.
        rf (float): Risk-free rate on the simulations' own scale -- annual, the
            implied returns being annualised. The maximum-Sharpe book is
            invariant to rescaling the mean and covariance together, but not to
            what is subtracted from the mean. Defaults to 0.
        n_eff (float | None): As `port_weight`. Defaults to None.

    Returns:
        pd.Series: As `port_weight`.

    Raises:
        ValueError: As `_usable_simulations`, and whatever `max_sharpe_weight`
            raises -- no book with a positive expected excess return, or a
            solve that does not converge.
    """
    labels, keep, matrix = _usable_simulations(return_arr)
    mu: np.ndarray = matrix.mean(axis=0)-rf
    sigma: np.ndarray = np.atleast_2d(np.cov(matrix, rowvar=False))
    weight: np.ndarray = max_sharpe_weight(mu, sigma, cap=1.0, min_weight=0.0,
                                           n_eff=n_eff)
    out: pd.Series = pd.Series(0.0, index=labels)
    out.loc[labels[keep]] = weight
    return out
