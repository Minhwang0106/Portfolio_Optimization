"""Gebhardt, Lee & Swaminathan's (2001) implied cost of capital, and the
expected return Bielstein & Hanauer (2019) build from it.

GLS value a share as its book value plus the present value of the residual
income it is forecast to earn, and read the implied cost of capital (ICC) off
the price as the discount rate at which the two agree:

    P0 = B0 + sum_{t=1}^{T-1} (ROE_t - r) B_{t-1} / (1+r)^t
            + (ROE_T - r) B_{T-1} / (r (1+r)^{T-1})

with T = 12. The last term is year-T residual income held flat for ever, as a
perpetuity. The forecast path the sum runs over has three parts, and none of
them depends on `r` -- so `gls_path` builds it once per firm and `icc_gls` only
searches the discount rate:

* Explicit years, 1 to `constant.icc_explicit_years` (3): ROE_t = EPS_t /
  B_{t-1}. GLS and B&H take EPS from analyst consensus; this repo has none and
  uses *realised* EPS instead, which is lookahead -- see `model.Icc_Mvo`.
* The fade, to year T: ROE moves in a straight line from ROE_3 to the industry
  median ROE, reaching it in year T.
* Book value throughout: clean surplus at a constant payout ratio k,
  B_t = B_{t-1} + EPS_t (1 - k), with EPS_t = ROE_t B_{t-1} in the fade years.

B&H then turn the ICC into an expected excess return by adding momentum on the
ICC's own scale and subtracting the risk-free rate; see
`expected_excess_return`.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from constant import icc_horizon, icc_winsor

# Discount rates the root is searched over. The terminal value divides by `r`,
# so the bracket cannot start at zero; at 0.01% the value is already far above
# any price for a firm whose ROE fades to a positive target, which the median of
# profitable firm-years always is. 100% a year is beyond any ICC GLS report, so
# a firm with no root under it is priced below what its own earnings path is
# worth at every sensible rate -- that is reported as NaN rather than pinned to
# the bracket's edge.
ICC_BRACKET: tuple[float, float] = (1e-4, 1.0)

MONTHS_PER_YEAR: int = 12


def gls_path (eps: np.ndarray, b0: np.ndarray, payout: np.ndarray,
              roe_target: np.ndarray, horizon: int = icc_horizon
              )-> tuple[np.ndarray, np.ndarray]:
    """ROE and beginning-of-year book value along GLS's forecast path.

    Args:
        eps (np.ndarray): Explicit EPS, shape (n_firms, n_explicit), year 1 in
            the first column. Every year must be filled in; `inputs.complete_eps`
            extends the ones the panel does not reach.
        b0 (np.ndarray): Book value per share at the start, shape (n_firms,).
        payout (np.ndarray): Dividend payout ratio k, in [0, 1].
        roe_target (np.ndarray): The industry median ROE the fade ends at.
        horizon (int): T, the year the fade reaches the target and the last
            one before the perpetuity. Defaults to `constant.icc_horizon`.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(roe, book)`, each (n_firms, T).
            `roe[:, t-1]` is ROE_t and `book[:, t-1]` is B_{t-1}, the book
            value that year's ROE is earned on -- so column `t-1` of the pair
            is exactly one residual-income term. A non-positive book value
            anywhere makes that firm's ROE meaningless; `icc_gls` returns NaN
            for it rather than solving.

    Raises:
        ValueError: If the horizon leaves no fade years after the explicit
            ones, or the inputs disagree on the number of firms.

    Example:
        GLS's worked example for General Motors (their Appendix A):

        >>> roe, book = gls_path(np.array([[6.75, 7.73, 8.29]]),
        ...                      np.array([17.01]), np.array([0.196]),
        ...                      np.array([0.16]))
        >>> round(float(book[0, 1]), 3), round(float(roe[0, 2]), 3)
        (22.437, 0.289)
    """
    eps = np.atleast_2d(np.asarray(eps, dtype=float))
    n_firm, n_explicit = eps.shape
    b0 = np.asarray(b0, dtype=float).reshape(-1)
    retain: np.ndarray = 1.0-np.asarray(payout, dtype=float).reshape(-1)
    target: np.ndarray = np.asarray(roe_target, dtype=float).reshape(-1)
    if not len(b0) == len(retain) == len(target) == n_firm:
        raise ValueError(f'{n_firm} rows of EPS but {len(b0)} book values, '
                         f'{len(retain)} payout ratios and {len(target)} '
                         'target ROEs')
    if horizon <= n_explicit:
        raise ValueError(f'a horizon of {horizon} years leaves no fade after '
                         f'{n_explicit} explicit ones')

    roe: np.ndarray = np.empty((n_firm, horizon))
    book: np.ndarray = np.empty((n_firm, horizon))
    book[:, 0] = b0
    # A zero or negative book value divides into ROE below; the firm is
    # dropped by `icc_gls`, so the warning would only be noise.
    with np.errstate(divide='ignore', invalid='ignore'):
        for col in range(horizon):              # column `col` is year col+1
            if col < n_explicit:
                earned: np.ndarray = eps[:, col]
                roe[:, col] = earned/book[:, col]
            else:
                # Linear in the year: ROE_3 at year 3, the target at year T.
                step: float = (col+1-n_explicit)/(horizon-n_explicit)
                last: np.ndarray = roe[:, n_explicit-1]
                roe[:, col] = last+step*(target-last)
                earned = roe[:, col]*book[:, col]
            if col+1 < horizon:
                book[:, col+1] = book[:, col]+earned*retain
    return roe, book


def gls_value (r: float|np.ndarray, roe: np.ndarray, book: np.ndarray
               )-> np.ndarray:
    """The price a forecast path implies at discount rate `r`.

    Args:
        r (float | np.ndarray): Annual discount rate, a scalar or one per firm.
        roe (np.ndarray): ROE_1..ROE_T, shape (T,) or (n_firms, T).
        book (np.ndarray): B_0..B_{T-1}, the same shape. See `gls_path`.

    Returns:
        np.ndarray: The value per share -- 0-d for a single firm, one per firm
            otherwise.

    Example:
        >>> roe, book = gls_path(np.array([[6.75, 7.73, 8.29]]),
        ...                      np.array([17.01]), np.array([0.196]),
        ...                      np.array([0.16]))
        >>> round(float(gls_value(0.1394, roe[0], book[0])), 1)
        48.5
    """
    rate: np.ndarray = np.asarray(r, dtype=float)
    roe = np.asarray(roe, dtype=float)
    book = np.asarray(book, dtype=float)
    horizon: int = roe.shape[-1]
    years: np.ndarray = np.arange(1, horizon)
    per_year: np.ndarray = rate[..., None]
    explicit: np.ndarray = ((roe[..., :-1]-per_year)*book[..., :-1]
                            /(1.0+per_year)**years).sum(axis=-1)
    # Year-T residual income from year T on, discounted back: the sum over
    # t >= T of X/(1+r)^t is X/(r (1+r)^{T-1}).
    terminal: np.ndarray = ((roe[..., -1]-rate)*book[..., -1]
                            /(rate*(1.0+rate)**(horizon-1)))
    return book[..., 0]+explicit+terminal


def icc_gls (price: np.ndarray, roe: np.ndarray, book: np.ndarray,
             bracket: tuple[float, float] = ICC_BRACKET)-> np.ndarray:
    """Implied cost of capital per firm: the `r` at which `gls_value` is the price.

    One bracketed root per firm, as GLS solve it. Each firm's price pins down
    its own rate, so there is nothing to gain from a joint fit, and one bad
    firm cannot move another's answer.

    Args:
        price (np.ndarray): Share price, shape (n_firms,).
        roe (np.ndarray): ROE paths, (n_firms, T), from `gls_path`.
        book (np.ndarray): Book value paths, (n_firms, T), from `gls_path`.
        bracket (tuple[float, float]): Rates to search between. Defaults to
            `ICC_BRACKET`.

    Returns:
        np.ndarray: The ICC per firm, annual, in decimals. NaN where the price
            is not positive, the path is not finite or reaches a non-positive
            book value (GLS drop negative-book firms, and ROE is undefined on
            one), or the value does not cross the price inside the bracket.

    Example:
        >>> roe, book = gls_path(np.array([[6.75, 7.73, 8.29]]),
        ...                      np.array([17.01]), np.array([0.196]),
        ...                      np.array([0.16]))
        >>> round(float(icc_gls(np.array([48.50]), roe, book)[0]), 4)
        0.1394
    """
    price = np.asarray(price, dtype=float).reshape(-1)
    roe = np.atleast_2d(np.asarray(roe, dtype=float))
    book = np.atleast_2d(np.asarray(book, dtype=float))
    lo, hi = bracket
    out: np.ndarray = np.full(len(price), np.nan)
    usable: np.ndarray = (np.isfinite(price) & (price > 0)
                          & np.isfinite(roe).all(axis=1)
                          & np.isfinite(book).all(axis=1)
                          & (book > 0).all(axis=1))
    for i in np.flatnonzero(usable):
        def gap (rate: float, i: int = i)-> float:
            return float(gls_value(rate, roe[i], book[i]))-price[i]
        # No sign change: the value stays above the price at 100% or below it
        # at 0.01%. Neither is a cost of capital.
        if gap(lo)*gap(hi) > 0:
            continue
        out[i] = brentq(gap, lo, hi, xtol=1e-12)
    return out


def winsorize (x: pd.Series, q: float = icc_winsor)-> pd.Series:
    """Clip a cross-section at its `q` and `1 - q` quantiles.

    Args:
        x (pd.Series): One value per ticker.
        q (float): Fraction clipped in each tail. Defaults to
            `constant.icc_winsor`.

    Returns:
        pd.Series: `x` with the tails pulled in to the quantiles.

    Raises:
        ValueError: If `q` is not in [0, 0.5).
    """
    if not 0 <= q < 0.5:
        raise ValueError(f'q must be in [0, 0.5), got {q}')
    return x.clip(lower=x.quantile(q), upper=x.quantile(1-q))


def rescaled_momentum (mom: pd.Series, icc: pd.Series)-> pd.Series:
    """Momentum on the ICC's scale: standardised, then times the ICC's dispersion.

    B&H's device for adding a return anomaly to the ICC without letting its
    units decide its weight: momentum is a cumulative return with a much wider
    cross-section than any cost of capital, so it is z-scored and given the
    ICC's own cross-sectional standard deviation. The result has mean zero, so
    it re-ranks names without moving the average expected return.

    Args:
        mom (pd.Series): Momentum per ticker.
        icc (pd.Series): ICC per ticker, the same cross-section.

    Returns:
        pd.Series: On `mom`'s index.

    Raises:
        ValueError: If momentum has no cross-sectional dispersion to divide by.
    """
    spread: float = float(mom.std())
    if not spread > 0:
        raise ValueError('momentum has no cross-sectional dispersion')
    return (mom-mom.mean())/spread*float(icc.std())


def expected_excess_return (icc: pd.Series, mom: pd.Series, rf: float,
                            winsor: float = icc_winsor)-> pd.Series:
    """Bielstein & Hanauer's expected excess return, per month.

    Both inputs are winsorised across the cross-section first; momentum is then
    rescaled onto the (winsorised) ICC's dispersion and added to it, and the
    risk-free rate comes off: `ICC + MOM* - rf`, annual.

    Converted to a monthly rate by dividing by 12, not by compounding, because
    the covariance matrix it is paired with is of monthly simple returns and the
    maximum Sharpe portfolio is unchanged by scaling every mean by one positive
    constant. Division keeps the vector exactly proportional to B&H's annual
    one; compounding each name separately would bend it.

    Args:
        icc (pd.Series): Annual ICC per ticker.
        mom (pd.Series): 12-1 momentum per ticker, on the same index.
        rf (float): Annual risk-free rate at the formation date.
        winsor (float): Fraction winsorised in each tail. Defaults to
            `constant.icc_winsor`.

    Returns:
        pd.Series: Expected monthly excess return per ticker.

    Raises:
        ValueError: If the two inputs are not on the same index.
    """
    if not icc.index.equals(mom.index):
        raise ValueError('icc and mom must be on the same index')
    icc_w: pd.Series = winsorize(icc, winsor)
    mom_w: pd.Series = winsorize(mom, winsor)
    annual: pd.Series = icc_w+rescaled_momentum(mom_w, icc_w)-rf
    return (annual/MONTHS_PER_YEAR).rename('expected_excess_return')
