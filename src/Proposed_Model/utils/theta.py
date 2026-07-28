"""Fitting the decay of eq (10).

Split from `sampling_distribution`, which holds the model itself -- the signal,
the lag moments, and the conditional law eqs (17) and (18) describe. This module
holds only the estimation of theta on top of that: the likelihood the model
implies, and the search that minimises it. Nothing here is needed to *draw* from
the model, and the split runs along that line.

The dependency points one way, `theta` -> `sampling_distribution`, and must stay
that way; nothing in the model needs to know how its parameter was chosen.
"""

import pandas as pd
import numpy as np
import warnings
from collections.abc import Callable, Mapping, Sequence

from .sampling_distribution import (
    LagMoments,
    _as_panel,
    conditional_mean,
    conditional_moments,
    lag_moments,
    rolling_signal,
)


def objective_func (theta:float|np.ndarray, moments:LagMoments,
                    data:pd.DataFrame|pd.Series|np.ndarray, n_lags:int,
                    var_eli:np.ndarray|float|None=None)->np.ndarray:
    """Mean negative log likelihood per observation, under eqs (17) and (18).

    Normalised two ways, neither of which moves the fitted theta but both of
    which make the returned number mean something on its own:

    * divided by the observations behind it, so columns with different amounts
      of history are on one scale. Raw sums are not: two columns drawn from the
      same process but 300 and 100 quarters long score -1015.8 and -326.8, and
      the gap is history, not fit. Per observation they are -3.397 and -3.404.
    * carrying the `log(2*pi)/2` the Gaussian density actually has, so this is a
      log likelihood rather than one up to an unstated constant, and feeds an
      AIC or a comparison across `n_lags` directly.

    Args:
        theta (float | np.ndarray): Decay in [0, 1], scalar or one per column.
        moments (LagMoments): Output of `lag_moments` on the same `data`.
        data (pd.DataFrame | pd.Series | np.ndarray): The observations.
        n_lags (int): Horizon `T` of eq (10).
        var_eli (np.ndarray | float | None): The `sigma_eli**2` of the `where`
            block. Defaults to `moments.var`.

    Returns:
        np.ndarray: `(d,)` mean negative log likelihood per observation. A
            column whose eq (18) is infeasible, or which has no residual left to
            score, is `inf` -- that walks the search away from it instead of
            stopping.

    Note:
        The residual count does not depend on theta, so the normalisation is a
        constant within a column and leaves every `find_theta` result unchanged
        (checked against the unnormalised objective on clean and on sparse
        panels). It is the reported value that changes, not the fit.

        The variance is the eq (18) plug-in rather than the residual variance
        concentrated out of the likelihood. The two agree to within about half a
        percent on data the model itself generated, and fitting either way
        recovers the same theta, so eq (18) is kept -- it is what the paper
        writes, and it keeps eqs (17) and (18) tied to one theta.
    """
    X, _, _ = _as_panel(data)
    beta, con_var = conditional_moments(moments.corr,moments.var,theta,var_eli)
    signal: np.ndarray = rolling_signal(X,n_lags,theta)
    resid: np.ndarray = X[1:] - conditional_mean(signal,moments.mean,beta)
    with np.errstate(invalid='ignore',divide='ignore'):
        n_obs: np.ndarray = np.sum(~np.isnan(resid),axis=0).astype(np.float64)
        mean_sq: np.ndarray = np.nansum(resid**2,axis=0)/n_obs
        score: np.ndarray = 0.5*(np.log(2*np.pi) + np.log(con_var)
                                 + mean_sq/con_var)
    return np.where(np.isfinite(score),score,np.inf)

def _fit_on_unit_interval (score:Callable[[np.ndarray],np.ndarray], d:int,
                           n_grid:int, tol:float)->np.ndarray:
    """Minimise `score` over [0, 1], every column stepping together.

    A grid pass followed by golden-section refinement, so the whole panel is
    evaluated on every step and nothing leaves numpy. A per-column optimiser
    call would cost more in its own scaffolding than in the objective.

    Args:
        score (Callable): Maps a `(d,)` theta to a `(d,)` objective. Must
            return `inf`, not raise, where a theta is infeasible.
        d (int): Number of columns being fitted at once.
        n_grid (int): Points in the opening sweep.
        tol (float): Width the bracket is refined to.

    Returns:
        np.ndarray: `(d,)` minimiser, NaN for a column whose objective is
            infeasible right across the interval.
    """
    # Opening sweep, so the refinement starts inside the right basin.
    grid: np.ndarray = np.linspace(0.0,1.0,n_grid)
    swept: np.ndarray = np.stack([score(np.full(d,g)) for g in grid])
    best: np.ndarray = np.argmin(swept,axis=0)
    lo: np.ndarray = grid[np.maximum(best-1,0)]
    hi: np.ndarray = grid[np.minimum(best+1,n_grid-1)]

    # Golden section, all columns stepping together.
    phi: float = (np.sqrt(5.0)-1.0)/2.0
    while np.max(hi-lo)>tol:
        left: np.ndarray = hi - phi*(hi-lo)
        right: np.ndarray = lo + phi*(hi-lo)
        take_left: np.ndarray = score(left)<score(right)
        hi = np.where(take_left,right,hi)
        lo = np.where(take_left,lo,left)

    theta: np.ndarray = (lo+hi)/2.0
    return np.where(np.isfinite(swept[best,np.arange(d)]),theta,np.nan)

def find_theta (panels:pd.DataFrame|pd.Series|np.ndarray|
                Sequence[pd.DataFrame|pd.Series|np.ndarray]|
                Mapping[str,pd.DataFrame|pd.Series|np.ndarray],
                n_lags:int=4,
                var_eli:np.ndarray|float|Sequence[np.ndarray|float|None]|
                None=None,
                n_grid:int=65, tol:float=1e-8)->pd.Series|float:
    """Fit the decay of eq (10) by maximum likelihood, every column at once.

    Searched rather than handed to an optimiser: theta is one bounded scalar
    per column, so a grid pass followed by golden-section refinement evaluates
    everything on every step and never leaves numpy. A per-column optimiser
    call costs more in its own scaffolding than in the objective.

    Given several panels the objective is summed over them before it is
    minimised, so one decay per column explains the whole universe rather than
    each ticker carrying its own. That is what a short training window forces.
    Theta enters eq (10) only through weights on `n_lags` lags, and a
    `constant.n_quarter` window leaves barely more residuals than that to
    identify it -- fitted per ticker it lands on 0 or 1 more often than not,
    which is the search reporting that the data cannot separate the interior
    values rather than a finding about persistence. Pooling backs each estimate
    with the whole universe while still letting the characteristics differ,
    which is where persistence genuinely varies.

    Args:
        panels (pd.DataFrame | pd.Series | np.ndarray | Sequence | Mapping):
            One `(n,)` or `(n, d)` panel of observations in time order, or a
            collection of them -- one per ticker, all carrying the same columns
            in the same order. A mapping is accepted so a `{ticker: frame}`
            dict goes in as it is; only its values are read. A single panel is
            detected rather than iterated, since iterating a DataFrame would
            yield its column labels. Nulls are dropped from a Series or an
            ndarray; a DataFrame is left as it is, since dropping a row for one
            column's sake would discard the others. The caller's objects are
            never modified.
        n_lags (int): Horizon `T` of eq (10). Defaults to 4.
        var_eli (np.ndarray | float | Sequence | None): The `sigma_eli**2` of
            the `where` block -- one value for a single panel, or one per panel
            in the order they are given. Defaults to each column's own variance.
        n_grid (int): Points in the opening sweep of [0, 1]. Defaults to 65.
        tol (float): Width the bracket is refined to. Defaults to 1e-8.

    Returns:
        pd.Series | float: A float for a single 1-D panel, otherwise one theta
            per column indexed by the column labels. A collection always gives
            a Series -- with several panels in play there is no single flat
            input to mirror. A column with no feasible theta anywhere on [0, 1]
            is NaN.

    Raises:
        ValueError: If `panels` is empty, if the panels disagree on their
            columns, if `var_eli` is given but does not match `panels` in
            length, or if no panel is long enough to score.

    Note:
        Which criterion the paper fits theta on is not settled here -- this is
        the likelihood implied by eqs (17) and (18), which is the natural
        reading but not one the paper states.

        Panels with `n_lags` or fewer observations cannot be scored at all and
        are dropped with a warning rather than taking the fit down; a universe
        normally holds a few tickers too newly listed to contribute.

        Panels are weighted by the residuals they actually carry, so a ticker
        with twice the history counts twice. Averaging the per-panel objectives
        instead would let a ticker with five usable quarters pull as hard as
        one with fifty.

        What pooling buys is variance, not accuracy. Measured on an AR(4) built
        from these very weights at a known interior theta, 25 panels of 24
        observations each, over independent draws of the universe: the pooled
        estimate has a standard deviation of 0.06 against 0.26 fitted one panel
        at a time, and stops landing on the endpoints. But both sit well above
        the long-sample answer -- 0.73 and 0.77 against 0.49 -- so the short
        panel bias survives pooling, which averages it rather than cancelling
        it, every panel carrying the same one. It falls away with panel length
        (0.78, 0.59, 0.54, 0.51 at 24, 40, 200 and 800 observations against a
        long-sample 0.49), not with universe size. Read a pooled theta as a
        stable summary of a short window, not as an unbiased one.
    """
    single: bool = isinstance(panels,(pd.DataFrame,pd.Series,np.ndarray))
    items: list = ([panels] if single
                   else list(panels.values()) if isinstance(panels,Mapping)
                   else list(panels))          # type: ignore[union-attr]
    if not items:
        raise ValueError('panels is empty, nothing to fit')
    scales: list = ([var_eli] if single else
                    [None]*len(items) if var_eli is None else list(var_eli))
    if len(scales)!=len(items):
        raise ValueError(f'var_eli holds {len(scales)} entries for '
                         f'{len(items)} panels')

    prepared: list[tuple[np.ndarray,LagMoments,np.ndarray,
                         np.ndarray|float|None]] = []
    label: list[str]|None = None
    flat: bool = False
    short: list[int] = []
    for k, data in enumerate(items):
        X, cols, was_flat = _as_panel(data)
        if was_flat:
            X = X[~np.isnan(X).any(axis=1)]
        if label is None:
            label, flat = cols, was_flat
        elif cols!=label:
            raise ValueError(f'panel {k} carries columns {cols}, but the first '
                             f'carries {label}; pooling needs one layout')
        if X.shape[0]<=n_lags:
            short.append(k)
            continue
        # Residual count, which `objective_func` notes is free of theta, so it
        # is the same weight at every step of the search.
        seen: np.ndarray = np.sum(
            ~np.isnan(X[1:]) & ~np.isnan(rolling_signal(X,n_lags,0.5)),
            axis=0).astype(np.float64)
        prepared.append((X,lag_moments(X,n_lags),seen,scales[k]))
    if short:
        warnings.warn(f'{len(short)} of {len(items)} panels have {n_lags} or '
                      f'fewer observations and cannot be scored; they are left '
                      f'out of the fit', RuntimeWarning)
    if not prepared or label is None:
        raise ValueError(f'no panel has more than n_lags={n_lags} observations')

    d: int = prepared[0][0].shape[1]
    total: np.ndarray = np.sum([seen for _,_,seen,_ in prepared],axis=0)

    def score (th:np.ndarray)->np.ndarray:
        acc: np.ndarray = np.zeros(d,dtype=np.float64)
        for X, moments, seen, ve in prepared:
            part: np.ndarray = objective_func(th,moments,X,n_lags,ve)
            # `seen` is 0 where a column contributed no residual, and 0*inf is
            # NaN, which would read as feasible. Drop those outright.
            acc = acc + np.where(seen>0,seen*part,0.0)
        with np.errstate(invalid='ignore',divide='ignore'):
            pooled: np.ndarray = acc/total
        return np.where(np.isfinite(pooled),pooled,np.inf)

    theta: np.ndarray = _fit_on_unit_interval(score,d,n_grid,tol)
    if single and flat:
        return float(theta[0])
    return pd.Series(theta,index=label,name='theta')
