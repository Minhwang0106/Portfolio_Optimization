"""The maximum-Sharpe solve `model.Icc_Mvo` calls, and a small panel helper.

Kept out of `model.py` so that module holds only `Icc_Mvo` itself: these two
functions take plain arrays and a plain frame, know nothing about the panels or
the class, and are what `tests/test_icc_mvo.py` exercises directly against
closed-form answers. `Proposed_Model.utils.port_optimize.moment_port_weight`
calls `max_sharpe_weight` too, to put the proposed model's simulated moments
through B&H's own rule.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from constant import icc_weight_cap, icc_weight_floor


def _best_return_book (mu: np.ndarray, cap: float)-> np.ndarray:
    """The feasible long-only book with the highest expected return.

    The cap on each of the best names in turn, and whatever is left on the next
    one. Its expected return is the highest any capped, fully invested book can
    reach, so if it is not positive no such book has a positive Sharpe ratio.
    """
    weight: np.ndarray = np.zeros(len(mu))
    left: float = 1.0
    for i in np.argsort(-mu):
        weight[i] = min(cap, left)
        left -= weight[i]
        if left <= 1e-15:
            break
    return weight


def _broad_best_book (mu: np.ndarray, cap: float, n_eff: float)-> np.ndarray:
    """Equal weight on the best names, as many as the cap and the breadth floor need.

    `k` equal weights have an effective N of `k` and a largest weight of `1/k`,
    so the smallest `k` with `k >= n_eff` and `1/k <= cap` satisfies both. A
    feasible starting point for the solve, not its answer.
    """
    k: int = min(len(mu), int(np.ceil(max(n_eff, 1.0/cap)-1e-9)))
    weight: np.ndarray = np.zeros(len(mu))
    weight[np.argsort(-mu)[:k]] = 1.0/k
    return weight


def _floor_and_cap (weight: np.ndarray, cap: float, min_weight: float
                    )-> np.ndarray:
    """Zero the weights under `min_weight`, renormalise, and keep the cap exact.

    Renormalising after the floor lifts every surviving weight by the few basis
    points it removed, which would take a name sitting on the cap a hair over
    it. The excess goes to the names below the cap in proportion to their
    weight, repeated until none is over.

    Raises:
        ValueError: If the capped names hold everything and there is nowhere
            to put the excess -- which the solve's own feasibility rules out.
    """
    weight = np.where(weight < min_weight, 0.0, weight)
    weight = weight/weight.sum()
    for _ in range(len(weight)):
        over: np.ndarray = weight > cap
        if not over.any():
            break
        excess: float = float((weight[over]-cap).sum())
        weight[over] = cap
        room: np.ndarray = (weight > 0) & (weight < cap)
        if not room.any():
            raise ValueError('every held name is at the cap; nowhere to put '
                             'the excess')
        weight[room] += excess*weight[room]/weight[room].sum()
    return weight


def max_sharpe_weight (mu: np.ndarray, sigma: np.ndarray,
                       cap: float = icc_weight_cap,
                       min_weight: float = icc_weight_floor,
                       long_only: bool = True,
                       n_eff: float|None = None)-> np.ndarray:
    """B&H's portfolio: maximum Sharpe ratio, long-only, capped, dust set to zero.

    Solved as the convex quadratic program the Sharpe ratio becomes under the
    substitution `y = w / (mu'w)`:

        min y' Sigma y   s.t.   mu'y = 1,   y >= 0,   y_i <= cap * sum(y)

    then `w = y / sum(y)`. The cap is homogeneous of degree one in `w`, which is
    what lets it pass through the substitution. The fractional objective itself
    is not concave, so solving it directly would leave the answer to the
    starting point; this form has one optimum. SLSQP with analytic gradients.

    An effective-N floor passes through the same way. `sum(w^2) <= 1/N` with
    `w = y / sum(y)` is `||y|| <= sum(y) / sqrt(N)`, a second-order cone, so
    the program stays convex. It is written as the cone rather than squared:
    the squared form is a difference of convex terms and not concave.

    Args:
        mu (np.ndarray): Expected excess returns, shape (n,).
        sigma (np.ndarray): Covariance matrix, shape (n, n), positive definite.
        cap (float): Largest weight on one name. Defaults to
            `constant.icc_weight_cap`; 1 leaves the book uncapped.
        min_weight (float): Weights below this are set to zero and the rest
            renormalised. Defaults to `constant.icc_weight_floor`.
        long_only (bool): False returns the unconstrained tangency portfolio,
            `Sigma^-1 mu` scaled to sum to one, which takes no cap. Defaults
            to True.
        n_eff (float | None): Smallest effective number of names, `1 /
            sum(w^2)`. None leaves breadth to the cap alone, B&H's rule. At or
            above `len(mu)` the only feasible book is equal weight, returned
            without solving. Long-only only. Defaults to None.

    Returns:
        np.ndarray: Weights summing to one, in `mu`'s order.

    Raises:
        ValueError: If no capped long-only book has a positive expected excess
            return (so none has a positive Sharpe ratio to maximise); if there
            are too few names to be fully invested under the cap; if the
            unconstrained tangency portfolio has a non-positive sum; if `n_eff`
            is given without `long_only`; or if the solver does not converge.
            `engine.backtest` holds the previous book on any of these.

    Example:
        >>> max_sharpe_weight(np.array([0.01, 0.02]), np.diag([0.04, 0.04]),
        ...                   cap=1.0).round(3)
        array([0.333, 0.667])
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)
    n: int = len(mu)
    if sigma.shape != (n, n):
        raise ValueError(f'{n} means against a covariance matrix of shape '
                         f'{sigma.shape}')
    if not long_only:
        if cap < 1 or n_eff is not None:
            raise ValueError('the weight cap and the effective-N floor are '
                             'only implemented for the long-only book')
        raw: np.ndarray = np.linalg.solve(sigma, mu)
        if not raw.sum() > 0:
            raise ValueError("the tangency portfolio's weights sum to a "
                             'non-positive number; it has no fully invested '
                             'form')
        return raw/raw.sum()
    if cap*n < 1-1e-12:
        raise ValueError(f'{n} names cannot make a fully invested book under a '
                         f'{cap:.2%} cap')
    if n_eff is not None and n_eff >= n:
        # Equal weight is the only book this broad, and SLSQP does not reliably
        # land on a feasible set squeezed down to one point.
        return np.full(n, 1.0/n)
    if not (mu > 0).any():
        raise ValueError('no long-only book has a positive expected excess '
                         'return')

    # Neither rescaling moves the optimum -- a positive multiple of mu or of
    # Sigma changes no Sharpe ranking -- and both put the solver on numbers of
    # order one. Monthly means near 0.005 against variances near 0.005 would
    # otherwise leave `y` in the hundreds.
    m: np.ndarray = mu/np.abs(mu).max()
    s: np.ndarray = sigma/np.diag(sigma).mean()
    breadth: float|None = None if n_eff is None else min(float(n_eff), n)
    start: np.ndarray = (_best_return_book(m, cap) if breadth is None
                         else _broad_best_book(m, cap, breadth))
    lead: float = float(m@start)
    if not lead > 0:
        raise ValueError('no long-only book has a positive expected excess '
                         'return' + ('' if breadth is None
                                     else f' at an effective N of {breadth:g}'))

    constraints: list[dict] = [{'type': 'eq', 'fun': lambda y: m@y-1.0,
                                'jac': lambda y: m}]
    if cap < 1:
        # cap * sum(y) - y_i >= 0, one row per name.
        bound: np.ndarray = cap*np.ones((n, n))-np.eye(n)
        constraints.append({'type': 'ineq', 'fun': lambda y: bound@y,
                            'jac': lambda y: bound})
    if breadth is not None:
        # sum(y)/sqrt(N) - ||y|| >= 0. `m'y = 1` keeps y off the origin, where
        # the norm's gradient would be undefined.
        root: float = float(np.sqrt(breadth))
        constraints.append({
            'type': 'ineq',
            'fun': lambda y: y.sum()/root-np.linalg.norm(y),
            'jac': lambda y: np.ones_like(y)/root-y/np.linalg.norm(y)})
    res = minimize(lambda y: y@s@y, start/lead, jac=lambda y: 2.0*(s@y),
                   method='SLSQP', bounds=[(0.0, None)]*n,
                   constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-12})
    if not res.success:
        raise ValueError(f'maximum-Sharpe solve did not converge: {res.message}')
    y: np.ndarray = np.clip(res.x, 0.0, None)
    return _floor_and_cap(y/y.sum(), cap, min_weight)


def _at (frame: pd.DataFrame, when: pd.Timestamp, names: pd.Index)-> pd.Series:
    """One row of a wide panel as a Series on `names`; NaN where absent."""
    return frame.reindex(index=[when], columns=names).iloc[0]
