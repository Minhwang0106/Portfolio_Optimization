import pandas as pd
import numpy as np
from typing import NamedTuple
from collections.abc import Sequence
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats


class LagMoments(NamedTuple):
    """Everything about a panel that eqs (17) and (18) need, computed once.

    None of it depends on theta, so a fit evaluates this once and then only
    re-weights it. `corr` is laid out with row/column 0 as the unshifted series
    and row/column `i` as lag `i`, which is the block structure the `where` of
    eq (17) indexes into.

    Attributes:
        corr (np.ndarray): `(d, n_lags+1, n_lags+1)` correlation matrices.
        var (np.ndarray): `(d,)` unconditional variance of each series, the
            `sigma**2` of eq (18).
        mean (np.ndarray): `(d,)` unconditional mean, the `mu_eli` of eq (17).
        n_obs (np.ndarray): `(d, n_lags+1, n_lags+1)` observations behind each
            correlation. These differ entry to entry -- see `lag_moments`.
        name (list[str]): Column labels, in the order of axis 0.
    """
    corr: np.ndarray
    var: np.ndarray
    mean: np.ndarray
    n_obs: np.ndarray
    name: list[str]


def _as_panel (data:pd.DataFrame|pd.Series|np.ndarray
               )->tuple[np.ndarray,list[str],bool]:
    """Normalise any of the accepted inputs to `(n, d)` plus its labels."""
    if isinstance(data,pd.Series):
        return data.to_numpy(dtype=np.float64)[:,None], [str(data.name)], True
    if isinstance(data,pd.DataFrame):
        return (data.to_numpy(dtype=np.float64), [str(c) for c in data.columns],
                False)
    arr: np.ndarray = np.asarray(data,dtype=np.float64)
    if arr.ndim==1:
        return arr[:,None], ['0'], True
    if arr.ndim!=2:
        raise ValueError(f'data must be 1- or 2-dimensional, got {arr.ndim}')
    return arr, [str(i) for i in range(arr.shape[1])], False

def signal_weights (n_lags:int, theta:float|np.ndarray)->np.ndarray:
    """Weights of eq (10), most recent lag first.

    Entry `j` multiplies `x_{t-1-j}`, so the last entry carries the
    `(1-theta)**(n_lags-1)` tail term on `x_{t-n_lags}`. The weights sum to 1
    for every theta, which is what makes `E[y_t] == E[x_t]` and lets eq (17)
    use one mean for both.

    Args:
        n_lags (int): Horizon `T` of eq (10). Must be at least 1.
        theta (float | np.ndarray): Decay in [0, 1]. A scalar gives one weight
            vector; a `(d,)` array gives one row per series, so a panel can
            carry a different theta per column.

    Returns:
        np.ndarray: `(n_lags,)` for scalar theta, `(d, n_lags)` for an array.
            Each row sums to 1.

    Raises:
        ValueError: If `n_lags` is less than 1.
    """
    if n_lags<1:
        raise ValueError('n_lags must be at least 1')
    th: np.ndarray = np.asarray(theta,dtype=np.float64)
    scalar: bool = th.ndim==0
    col: np.ndarray = np.atleast_1d(th).reshape(-1,1)
    i: np.ndarray = np.arange(1,n_lags)
    w: np.ndarray = np.empty((col.shape[0],n_lags),dtype=np.float64)
    w[:,:n_lags-1] = col*(1-col)**(i-1)
    w[:,n_lags-1] = ((1-col)**(n_lags-1)).ravel()
    return w[0] if scalar else w

def signal_cal (window:np.ndarray|Sequence[np.ndarray], n_lags:int,
                theta:float|np.ndarray)->np.ndarray|np.float64:
    """Eq (10) on one window, contracted along time.

    Axis 0 is time, most recent observation **last**, and every other axis is
    carried through untouched. So a `(T, n_simulation, d)` block of simulated
    paths collapses to the `(n_simulation, d)` signal that stands against the
    next period -- which is the shape `simulation.sample_joint` produces one
    period at a time. The periods may equally arrive as a sequence of separate
    blocks rather than one array; see the second note.

    Args:
        window (np.ndarray | Sequence[np.ndarray]): `(k, ...)` observations in
            time order, or `k` separate blocks in that same order, each of the
            shape that one period carries. Only the last `n_lags` are read; a
            shorter window is allowed and renormalises eq (10) onto the horizon
            actually available.
        n_lags (int): Horizon `T` of eq (10).
        theta (float | np.ndarray): Decay in [0, 1]. A scalar applies to
            everything; a `(d,)` array broadcasts against the **last** axis of
            `window`, so each characteristic can carry its own theta.

    Returns:
        np.ndarray | np.float64: One period's worth of signal, i.e.
            `window.shape[1:]` for an array and the shape of one block for a
            sequence -- a scalar for a 1-D window either way.

    Raises:
        ValueError: If `window` holds nothing to read along time.

    Note:
        The contraction is written out rather than left to `np.dot`, which
        reduces against the *second-to-last* axis of a multi-dimensional
        operand. On a `(T, n_simulation, d)` block that raises whenever
        `n_simulation != n_lags` -- and silently returns the wrong sum when
        they happen to be equal.

        The sequence form is what the forward recursion needs. Eq (10) reads
        *realised* x, so on a simulated path period `t`'s signal cannot be built
        until `t-1` has been through `sample_conditional`: there is no finished
        `(T, n_simulation, d)` array to slice into, only the blocks drawn so
        far. Those can be kept in a list -- or a `deque` of `maxlen=n_lags`, or
        seeded with the tail of real history broadcast to `(n_simulation, d)` --
        and handed straight over. Stacking them costs one copy of
        `n_lags*n_simulation*d` floats per period, which is nothing beside the
        copula draws that filled them. Only the last `n_lags` entries are
        stacked, so handing over an ever-growing history does not pay for its
        own length.
    """
    if isinstance(window,np.ndarray):
        block: np.ndarray = window
    else:
        seq: list[np.ndarray] = list(window)
        keep: int = min(len(seq),n_lags)
        if not keep:
            raise ValueError('window is empty')
        block = np.stack(seq[len(seq)-keep:])

    true_lags: int = min(block.shape[0],n_lags)
    if not true_lags:
        raise ValueError('window is empty')

    recent: np.ndarray = block[-1:-true_lags-1:-1]       # most recent first
    w: np.ndarray = signal_weights(true_lags,theta)
    if w.ndim==1:
        shaped: np.ndarray = w.reshape((true_lags,)+(1,)*(recent.ndim-1))
    else:
        # (d, k) -> (k, 1..., d): line the weights up with the trailing axis so
        # each characteristic is weighted by its own theta.
        shaped = w.T.reshape((true_lags,)+(1,)*(recent.ndim-2)+(w.shape[0],))
    return np.sum(shaped*recent,axis=0)

def rolling_signal (data:pd.DataFrame|pd.Series|np.ndarray, n_lags:int,
                    theta:float|np.ndarray)->np.ndarray:
    """Eq (10) at every date, for a whole panel, aligned to predict `data[1:]`.

    Row `k` is the signal built from `data[:k+1]`, i.e. the one that stands
    against `data[k+1]`. The signal never reads its own target.

    Args:
        data (pd.DataFrame | pd.Series | np.ndarray): `(n,)` or `(n, d)`
            observations in time order.
        n_lags (int): Horizon `T` of eq (10).
        theta (float | np.ndarray): Decay in [0, 1], scalar or one per column.

    Returns:
        np.ndarray: `(n-1, d)`, empty if there are fewer than 2 observations.

    Note:
        The first `n_lags-1` rows are built from windows shorter than `n_lags`,
        where eq (10) is renormalised onto the horizon actually available
        rather than zero padded. `signal_moments` assumes a full window
        throughout, so those rows carry moments that are slightly off; measured
        on AR(1) draws the effect on the fitted theta is within the sampling
        noise from about 80 observations up.
    """
    X, _, _ = _as_panel(data)
    n, d = X.shape
    if n<2:
        return np.empty((0,d),dtype=np.float64)
    th: np.ndarray = np.broadcast_to(
        np.atleast_1d(np.asarray(theta,dtype=np.float64)),(d,))

    out: np.ndarray = np.empty((n-1,d),dtype=np.float64)
    # Full windows, as one contraction: window `i` ends at k = i+n_lags-1, and
    # the last one we can use is k = n-2, hence the n-n_lags of them.
    if n>n_lags:
        block: np.ndarray = sliding_window_view(X,n_lags,axis=0)[...,::-1]
        out[n_lags-1:] = np.einsum('mdi,di->md',block[:n-n_lags],
                                   signal_weights(n_lags,th))
    # Short windows at the head. Only n_lags-1 of them, each still vectorised
    # across the panel, so the loop does not grow with n or d.
    for k in range(min(n_lags-1,n-1)):
        out[k] = np.einsum('di,id->d',signal_weights(k+1,th),X[k::-1])
    return out

def lag_moments (data:pd.DataFrame|pd.Series|np.ndarray, n_lags:int=4
                 )->LagMoments:
    """Lag correlations, variances and means for every column, in one pass.

    Reproduces `DataFrame.corr()` on the lag-shifted frame exactly, including
    its *pairwise* deletion: each entry is scored on the rows where its own two
    columns are both present, so `Cor(x_t, x_{t-1})` sees more observations
    than `Cor(x_t, x_{t-n_lags})`. That matters here -- the training window is
    only `constant.n_quarter` quarters long, and dropping rows that are
    complete across all `n_lags+1` columns instead would throw away most of it
    (at 30% missing and n=20 it leaves nothing at all).

    Args:
        data (pd.DataFrame | pd.Series | np.ndarray): `(n,)` or `(n, d)`
            observations in time order. NaNs are handled, not dropped.
        n_lags (int): Horizon `T` of eq (10). Defaults to 4.

    Returns:
        LagMoments: See that class. Axis 0 of every field indexes the columns.

    Note:
        Pairwise deletion buys sample size at the cost of positive
        semi-definiteness: entries scored on different rows need not form a
        consistent matrix, and eq (18) can then come out negative. Measured on
        AR(1) draws that never happens at `n_lags=4` even with 30% missing, but
        it reaches 262 cases in 300 at `n_lags=12` with 40% missing. If you
        raise `n_lags` on sparse data, check `conditional_moments` for NaN.
    """
    X, name, _ = _as_panel(data)
    n, d = X.shape
    if n<=n_lags:
        raise ValueError(f'need more than n_lags={n_lags} observations, got {n}')

    # Column k is lag k. The head padding reproduces what `series.shift(k)`
    # leaves behind, so the pairwise counts match pandas row for row.
    padded: np.ndarray = np.vstack([np.full((n_lags,d),np.nan),X])
    Z: np.ndarray = sliding_window_view(padded,n_lags+1,axis=0)[...,::-1]

    mask: np.ndarray = (~np.isnan(Z)).astype(np.float64)
    Z0: np.ndarray = np.where(mask>0,Z,0.0)

    # Every (i, j) pair scored on the rows where both are present. Each einsum
    # is one pass over the (n, d, n_lags+1) block.
    count: np.ndarray = np.einsum('mdi,mdj->dij',mask,mask)
    sum_i: np.ndarray = np.einsum('mdi,mdj->dij',Z0,mask)
    sq_i: np.ndarray = np.einsum('mdi,mdj->dij',Z0*Z0,mask)
    cross: np.ndarray = np.einsum('mdi,mdj->dij',Z0,Z0)

    with np.errstate(invalid='ignore',divide='ignore'):
        mean_i: np.ndarray = sum_i/count
        mean_j: np.ndarray = np.swapaxes(sum_i,1,2)/count
        cov: np.ndarray = cross/count - mean_i*mean_j
        var_i: np.ndarray = sq_i/count - mean_i*mean_i
        var_j: np.ndarray = np.swapaxes(sq_i,1,2)/count - mean_j*mean_j
        corr: np.ndarray = cov/np.sqrt(var_i*var_j)

    return LagMoments(corr=corr,
                      var=np.nanvar(X,axis=0),
                      mean=np.nanmean(X,axis=0),
                      n_obs=count,
                      name=name)

def signal_moments (corr:np.ndarray, var:np.ndarray|float,
                    theta:float|np.ndarray, var_eli:np.ndarray|float|None=None
                    )->tuple[np.ndarray,np.ndarray]:
    """`sigma_y` and `sigma_xy` of the `where` block, for every column at once.

    Both are just the weight vector contracted against the lag correlations:
    with `w = signal_weights(n_lags, theta)`, `R` the correlations among lags
    1..T and `r` the correlations of `x_t` against those lags,

        sigma_y  = w' R w * var        (the block's A + 2B + C)
        sigma_xy = w . r   * var

    Writing the A/B/C sums out term by term gives the same numbers -- verified
    to machine precision for `n_lags` 2 through 16 across theta -- but as a
    quadratic form it is one contraction instead of a double loop, and it
    broadcasts over the panel for free.

    Takes the two arrays it reads rather than a `LagMoments`, so a correlation
    matrix from anywhere -- shrunk, hand-written, imposed by an AR model -- goes
    in without inventing the means and counts that the rest of that tuple wants.
    `lag_moments` is one way to get them, not the only one.

    Args:
        corr (np.ndarray): `(n_lags+1, n_lags+1)` for a single series, or
            `(d, n_lags+1, n_lags+1)` for a panel. Row/column 0 is the
            unshifted series and row/column `i` is lag `i`, the layout the
            `where` of eq (17) indexes into. Not checked for symmetry or
            positive semi-definiteness -- see `conditional_moments`, which
            reports an eq (18) that comes out non-positive as NaN.
        var (np.ndarray | float): `(d,)` or scalar unconditional variance of
            each series, the `sigma**2` of eq (18). Only used as the default
            scale, so it is ignored outright when `var_eli` is given.
        theta (float | np.ndarray): Decay in [0, 1], scalar or one per series.
        var_eli (np.ndarray | float | None): The `sigma_eli**2` scaling every
            correlation term. Defaults to `var`.

    Returns:
        tuple[np.ndarray, np.ndarray]: `sigma_y` and `sigma_xy`, both `(d,)`.
            A 2-D `corr` is treated as a panel of one, so both come back
            shaped `(1,)` rather than as scalars.

    Raises:
        ValueError: If `corr` is not square in its last two axes, is neither
            2- nor 3-dimensional, or is smaller than 2x2.
        TypeError: If handed a `LagMoments` instead of its two fields.
    """
    if isinstance(corr,LagMoments):
        raise TypeError('signal_moments takes corr and var, not a LagMoments -- '
                        'pass moments.corr and moments.var')
    R: np.ndarray = np.asarray(corr,dtype=np.float64)
    if R.ndim==2:
        R = R[None]
    if R.ndim!=3 or R.shape[1]!=R.shape[2]:
        raise ValueError('corr must be (n_lags+1, n_lags+1) or '
                         f'(d, n_lags+1, n_lags+1), got {np.shape(corr)}')
    d: int = R.shape[0]
    n_lags: int = R.shape[1]-1
    if n_lags<1:
        raise ValueError('corr must be at least 2x2, i.e. one lag')

    scale: np.ndarray = np.broadcast_to(
        np.atleast_1d(np.asarray(
            var if var_eli is None else var_eli,dtype=np.float64)),(d,))
    th: np.ndarray = np.broadcast_to(
        np.atleast_1d(np.asarray(theta,dtype=np.float64)),(d,))

    w: np.ndarray = signal_weights(n_lags,th)
    lag_corr: np.ndarray = R[:,1:,1:]
    cross_corr: np.ndarray = R[:,0,1:]
    sig_var: np.ndarray = np.einsum('di,dij,dj->d',w,lag_corr,w)*scale
    sig_cov: np.ndarray = np.einsum('di,di->d',w,cross_corr)*scale
    return sig_var, sig_cov

def conditional_moments (corr:np.ndarray, var:np.ndarray|float,
                         theta:float|np.ndarray,
                         var_eli:np.ndarray|float|None=None
                         )->tuple[np.ndarray,np.ndarray]:
    """Regression slope and conditional variance of eqs (17) and (18).

    Returns `sigma_xy/sigma_y` rather than a mean, since the mean also needs
    the realised signal; `conditional_mean` closes that gap.

    Takes the two arrays it reads rather than a `LagMoments`, for the same
    reason `signal_moments` does: the means and counts in that tuple are no
    part of eqs (17) and (18), and a caller whose variance is elicited rather
    than measured has none of them to hand over.

    Args:
        corr (np.ndarray): `(n_lags+1, n_lags+1)` or `(d, n_lags+1, n_lags+1)`
            lag correlations, e.g. `lag_moments(...).corr`.
        var (np.ndarray | float): `(d,)` or scalar unconditional variance of
            each series, the `sigma**2` of eq (18).
        theta (float | np.ndarray): Decay in [0, 1], scalar or one per column.
        var_eli (np.ndarray | float | None): The `sigma_eli**2` of the `where`
            block. Defaults to `var`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The slope of eq (17) and the variance of
            eq (18), both `(d,)`. A column whose eq (18) falls outside
            `(0, inf)` is returned as NaN rather than raising, so one bad
            column does not take the panel down with it.

    Note:
        The slope is invariant to `var_eli` -- it scales `sigma_xy` and
        `sigma_y` alike and cancels -- but eq (18) is not, since only the
        subtracted term carries it. Getting `var_eli` wrong therefore leaves
        conditional means intact and biases every conditional variance.

        At the default `var_eli == var` the scale drops out of eq (18)
        altogether: it reduces to `var * (1 - (w.r)**2/(w'Rw))`, whose sign is
        a property of `corr` and `theta` alone. Cauchy-Schwarz then holds it
        positive for any `corr` that is positive semi-definite, so the NaN
        above is reachable only through the pairwise deletion of `lag_moments`
        -- and even then rarely, since `w` is a smooth positive direction.
        Decoupling `var_eli` from `var` withdraws that guarantee outright:
        measured on clean AR(1) correlations, `var_eli` at five times `var`
        already drives eq (18) negative.
    """
    var_x: np.ndarray = np.asarray(var,dtype=np.float64)
    sig_var, sig_cov = signal_moments(corr,var,theta,var_eli)
    with np.errstate(invalid='ignore',divide='ignore'):
        beta: np.ndarray = sig_cov/sig_var
        con_var: np.ndarray = var_x - sig_cov**2/sig_var
    # sigma_xy and sigma_y come from correlations scored on different rows, so
    # Cauchy-Schwarz does not hold automatically and eq (18) can leave (0, inf).
    ok: np.ndarray = np.isfinite(con_var) & (con_var>0) & np.isfinite(beta)
    return np.where(ok,beta,np.nan), np.where(ok,con_var,np.nan)

def conditional_mean (signal:np.ndarray|float, mu_x:np.ndarray|float,
                      beta:np.ndarray|float)->np.ndarray|float:
    """Eq (17): `mu + beta*(y_t - mu)`, broadcast over the signal."""
    return mu_x + beta*(signal-mu_x)

def sample_conditional (u:np.ndarray|float, signal:np.ndarray|float,
                        mu_x:np.ndarray|float, beta:np.ndarray|float,
                        con_var:np.ndarray|float)->np.ndarray|float:
    """Eq (11): map uniform draws through the conditional normal's inverse CDF.

    The uniforms are the copula's output, so this is the step that carries a
    draw off the unit square and back onto the scale of `x_t`.

    Args:
        u (np.ndarray | float): Uniform draws in the open interval (0, 1). Any
            shape; it broadcasts against `signal`, so a `(n_path, d)` block of
            uniforms against a `(d,)` signal gives one path per row.
        signal (np.ndarray | float): `y_t` from `rolling_signal` or
            `signal_cal`.
        mu_x (np.ndarray | float): Unconditional mean, the `mu_eli` of eq (17).
        beta (np.ndarray | float): First element of `conditional_moments`.
        con_var (np.ndarray | float): Second element of `conditional_moments`.

    Returns:
        np.ndarray | float: Draws of `x_t`, broadcast to the common shape.

    Raises:
        ValueError: If any `u` falls outside (0, 1), which the inverse CDF
            would send to an infinity, or if any `con_var` is not positive.
    """
    u_arr: np.ndarray = np.asarray(u,dtype=np.float64)
    if np.any((u_arr<=0)|(u_arr>=1)) or not np.all(np.isfinite(u_arr)):
        raise ValueError('u must lie strictly inside (0, 1)')
    cv: np.ndarray = np.asarray(con_var,dtype=np.float64)
    if not np.all(np.isfinite(cv)) or np.any(cv<=0):
        raise ValueError('con_var must be positive and finite everywhere')
    return conditional_mean(signal,mu_x,beta) + np.sqrt(cv)*stats.norm.ppf(u_arr)
