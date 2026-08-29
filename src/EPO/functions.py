import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from constant import risk_n_day, risk_day_per_name

# Smallest denominator `cross_sectional_momentum` will divide by. The positive
# half of a demeaned cross-section sums to zero only if every name shares the
# same trailing return, which no real universe does -- but a one-name universe
# reaches it exactly, and dividing by that turns a degenerate date into infinite
# weights rather than a flat book.
_XSMOM_FLOOR: float = 1e-12


def cal_return(price_path: Path, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Compute simple returns per ticker, wide format.

    price_path is expected to hold long-format columns ['ticker', 'date', 'adj_close'].
    The frequency of the result is the frequency of the file: pass the daily price
    panel and the returns are daily, pass the monthly one and they are monthly.

    Kept separate from `cal_excess_return` because the risk model wants raw returns
    while the signal and the P&L want excess ones (§2 of
    `Instruction/epo_equity4_implementation.md`); subtracting a rate common to every
    name barely moves a covariance matrix, but there is no reason to subtract it.

    Args:
        price_path: CSV of adjusted close prices, long format.
        start_date: Inclusive lower bound (YYYY-MM-DD) applied to the result's date index.

    Returns:
        DataFrame indexed by date, columns are tickers, values are simple returns
        (decimal, not percent).
    """
    df_p: pd.DataFrame = pd.read_csv(price_path)
    df_p['date'] = pd.to_datetime(df_p['date'])
    df_p.set_index(['ticker', 'date'], inplace=True)
    return_df: pd.DataFrame = df_p['adj_close'].unstack(level='ticker').pct_change()
    return_df = return_df[return_df.index >= start_date]
    return_df.index.name = 'date'
    return_df.columns.name = 'ticker'
    return return_df


def cov_window(n_candidate: int, n_day: int = risk_n_day,
               day_per_name: int = risk_day_per_name) -> int:
    """Length of the trailing daily window the covariance is estimated on.

    `max(n_day, day_per_name * n_candidate)`. Table 1 of the paper fixes this at
    120 days, which suits its 49 industry portfolios; this universe runs 172 to
    381 names, where a flat 120 would leave the sample covariance rank-deficient
    at every date. See `constant.risk_n_day` for the full rationale.

    Sized off the *candidate* pool rather than the surviving one on purpose. The
    survivors are the candidates that additionally have a complete window, so
    `n_t <= n_candidate` always, and the guarantee `K >= day_per_name * n_t`
    holds without iterating between the window length and the universe it screens.

    Args:
        n_candidate (int): Names in the candidate pool at this date, before the
            data-completeness screen.
        n_day (int): Floor, in trading days. Defaults to `constant.risk_n_day`.
        day_per_name (int): Observations required per name. Defaults to
            `constant.risk_day_per_name`.

    Returns:
        int: Window length in trading days.

    Example:
        >>> cov_window(49)
        120
        >>> cov_window(381)
        762
    """
    return max(n_day, day_per_name*n_candidate)


def cross_sectional_momentum(cum_return: pd.Series) -> pd.Series:
    """XSMOM: the paper's cross-sectional momentum signal, eq. (18).

    Demeans the trailing cumulative returns, then scales so the positive entries
    sum to +1 -- which puts the negative entries at -1 automatically, since
    demeaning makes the two halves equal in magnitude. One scalar does both
    sides; normalising the legs separately is a different signal and a common
    way to get this wrong.

    The result is a self-financing long-short portfolio over the cross-section,
    so its entries shrink as the universe grows: over 49 assets a typical entry
    is ~0.02, over 381 it is ~0.005.

    Args:
        cum_return (pd.Series): Trailing cumulative excess return per ticker,
            over `constant.risk_n_month` months.

    Returns:
        pd.Series: The signal, indexed like `cum_return`, summing to zero with
            `sum(max(x, 0)) == 1`.

    Example:
        >>> x = cross_sectional_momentum(pd.Series({'A': 0.3, 'B': 0.1, 'C': -0.1}))
        >>> float(x[x > 0].sum()), float(x.sum())
        (1.0, 0.0)
    """
    demeaned: pd.Series = cum_return-cum_return.mean()
    upside: float = float(demeaned[demeaned > 0].sum())
    return demeaned/max(upside, _XSMOM_FLOOR)


def cal_excess_return(price_path: Path, rf_path: Path, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Compute excess returns (simple return minus risk-free rate) per ticker.

    price_path is expected to hold long-format columns ['ticker', 'date', 'adj_close'].
    rf_path is expected to hold Kenneth French factor data with a 'date' column
    in YYYYMMDD (daily) or YYYYMM (monthly) format and an 'RF' column given in
    percent (e.g. 0.01 == 0.01%).

    Args:
        price_path: CSV of adjusted close prices, long format.
        rf_path: CSV of Fama-French factors, must contain 'date' and 'RF'.
        start_date: Inclusive lower bound (YYYY-MM-DD) applied to the result's date index.

    Returns:
        DataFrame indexed by date, columns are tickers, values are excess returns
        (decimal, not percent).
    """
    return_df: pd.DataFrame = cal_return(price_path, start_date)

    df_rf: pd.DataFrame = pd.read_csv(rf_path)
    if 'date' not in df_rf.columns:
        df_rf.rename(columns={df_rf.columns[0]: 'date'}, inplace=True)
    date_str = df_rf['date'].astype(str)
    if date_str.str.len().eq(6).all():
        # monthly FF5 file: YYYYMM, no day-of-month -> snap to month end to match price dates
        df_rf['date'] = pd.to_datetime(date_str, format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        df_rf['date'] = pd.to_datetime(date_str, format='%Y%m%d')
    df_rf.set_index('date', inplace=True)
    rf_rate: pd.Series = (df_rf['RF'] / 100).rename('RF_rate')  # RF is a stock ticker too, avoid name collision

    return_df = pd.concat([return_df, rf_rate], axis=1)
    return_df = return_df[return_df.index >= start_date]
    return_df = return_df[return_df.index <= rf_rate.index.max()]  # RF often lags price data; trim to avoid NaN rows

    tickers: list = [c for c in return_df.columns if c != 'RF_rate']
    return_arr: np.ndarray = return_df[tickers].to_numpy()
    rf_arr: np.ndarray = return_df[['RF_rate']].to_numpy()  # kept 2D (n,1) to broadcast against (n, n_tickers)

    excess_return: pd.DataFrame = pd.DataFrame(
        return_arr - rf_arr, columns=tickers, index=return_df.index
    )
    excess_return.index.name = 'date'
    excess_return.columns.name = 'ticker'
    return excess_return


def _mean_variance (weight: np.ndarray, signal: np.ndarray,
                    Sigma: np.ndarray, gamma: float)-> float:
    """Negated mean-variance objective, `-(w'mu - gamma/2 w'Sigma w)`.

    The same objective the unconstrained `Sigma^-1 @ signal / gamma` maximises,
    written out so it can be handed to a constrained solver.
    """
    return float(-(weight@signal-0.5*gamma*(weight@Sigma@weight)))


def _mean_variance_jac (weight: np.ndarray, signal: np.ndarray,
                        Sigma: np.ndarray, gamma: float)-> np.ndarray:
    """Gradient of `_mean_variance`, `-(mu - gamma*Sigma@w)`.

    Supplied explicitly because SLSQP would otherwise difference the objective
    once per variable per iteration, and there are ~150 of them at every
    formation date.
    """
    return -(signal-gamma*(Sigma@weight))


def _long_only_weight (Sigma: np.ndarray, signal: np.ndarray,
                       gamma: float)-> np.ndarray:
    """Maximise the mean-variance objective over the long-only simplex.

    Solves `max w'mu - gamma/2 w'Sigma w` subject to `w >= 0` and `sum(w) == 1`.
    Unlike clipping the unconstrained solution, this leaves the covariance
    matrix doing real work -- a name can still be held down because of what it
    is correlated with rather than only because of its own signal.

    Args:
        Sigma (np.ndarray): Shrunk covariance, shape `(n, n)`.
        signal (np.ndarray): Expected excess returns, shape `(n,)`.
        gamma (float): Risk aversion.

    Returns:
        np.ndarray: Weights, shape `(n,)`, non-negative and summing to one.

    Raises:
        ValueError: If the solver does not converge, which the caller records as
            a failed formation date rather than passing an arbitrary vector on.
    """
    n_stock: int = signal.shape[0]
    start: np.ndarray = np.ones(n_stock)/n_stock
    res = minimize(_mean_variance, start, args=(signal, Sigma, gamma),
                   jac=_mean_variance_jac, method='SLSQP',
                   bounds=[(0.0, 1.0)]*n_stock,
                   constraints={'type': 'eq',
                                'fun': lambda w: float(w.sum()-1.0),
                                'jac': lambda w: np.ones_like(w)},
                   options={'maxiter': 500, 'ftol': 1e-12})
    if not res.success:
        raise ValueError(f'EPO long-only solve did not converge: {res.message}')
    # Clip before renormalising: SLSQP satisfies its bounds to solver tolerance,
    # so a weight can land at -1e-17 and turn `n_holdings` into a count of
    # rounding noise.
    weight: np.ndarray = np.clip(res.x, 0.0, None)
    return weight/weight.sum()