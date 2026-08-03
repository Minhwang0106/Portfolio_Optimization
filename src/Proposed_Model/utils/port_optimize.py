import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ...PPP.utils import crra_utility
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

def port_weight (return_arr:np.ndarray|pd.DataFrame,
                 risk_aversion:float=risk_aversion,
                 long_only: bool= True)->pd.Series:
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

    Returns:
        pd.Series: One weight per ticker, indexed by ticker when the input
            carried labels and positionally otherwise, summing to 1. A ticker
            dropped for non-finite simulations gets 0.

    Raises:
        ValueError: If no ticker is usable, if no simulation is priced across
            every usable ticker, or if the optimiser does not converge.

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
    matrix: np.ndarray = usable.T
    N_s, n_tickers = matrix.shape

    constrains: dict = {'type':'eq', 'fun': lambda w: np.sum(w)-1}

    if long_only:
        bounds: list[tuple] = [(0,1) for _ in range(n_tickers)]
    else:
        bounds: list[tuple] = [(None,None) for _ in range(n_tickers)]
    weight: np.ndarray = np.ones(shape=(n_tickers))/n_tickers

    res = minimize(objective_func, weight,
                   args=(matrix, risk_aversion),
                   constraints=constrains, bounds=bounds)
    if not res.success:
        raise ValueError(f'optimiser did not converge: {res.message}')

    # The whole vector: `res.x[0]` is one asset's weight, not the portfolio.
    out: pd.Series = pd.Series(0.0,index=labels)
    out.loc[labels[keep]] = res.x
    return out