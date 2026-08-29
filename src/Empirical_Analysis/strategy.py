"""One calling convention for three models that do not share one.

`engine.backtest` needs a single callable `(date, tickers) -> pd.Series` indexed
by ticker. None of the three models offers that:

* `EPO.portfolio_weight` takes no universe at all -- it looks its own up by date
  -- and returns a bare array ordered by `EPO.correl[date].columns`. It is
  unnormalised by default, and sums to one under `long_only=True`.
* `PPP.portfolio_weight` returns a `(1, n_t)` array ordered by `self.ticker`,
  which is *narrower* than the list passed in and is only readable off the
  instance afterwards.
* `RIM_PortOp.weight` already returns a labelled Series, and is the only one
  that does.

The adapters here put all three on the Series convention and nothing else. They
do not change any model's weighting rule; the one thing they do decide is EPO's
scale, which the model genuinely leaves open -- see `normalise`.
"""
import warnings
import numpy as np
import pandas as pd
from ..EPO.model import EPO
from ..PPP.model import PPP
from ..Proposed_Model.model import RIM_PortOp

# Below this the normalising denominator is treated as zero rather than divided
# by. A gross exposure this small means the solver returned essentially no
# position, and dividing by it turns rounding noise into a levered book.
_SCALE_FLOOR: float = 1e-12


def normalise (weight: pd.Series, how: str = 'gross')-> pd.Series:
    """Put a weight vector on a stated scale.

    Only EPO needs this, and by default EPO always does. `PPP` and `RIM_PortOp`
    both already sum to one -- the first because the standardized tilts cancel,
    the second because `port_weight` constrains them to -- as does EPO under
    `long_only=True`. Unconstrained, which is now EPO's default, it returns
    `Sigma_w^-1 @ signal / gamma`, whose magnitude is whatever the covariance
    matrix and the signal scale happen to imply. That number is not a portfolio
    weight in the same sense as the other two, and comparing a Sharpe ratio
    computed on it against theirs compares two different amounts of capital.

    Args:
        weight (pd.Series): Raw weights, indexed by ticker.
        how (str): One of

            * 'gross' -- scale to `sum(|w|) == 1`, i.e. one unit of capital
              deployed across the book. The right choice for a long-short
              vector, whose net exposure is not a meaningful denominator.
            * 'net' -- scale to `sum(w) == 1`, i.e. fully invested. Only
              meaningful when the net exposure is reliably away from zero.
            * 'none' -- return the weights untouched.

    Returns:
        pd.Series: Scaled weights. All zeros if the denominator is below
            `_SCALE_FLOOR`, which is a flat book rather than a levered one.

    Raises:
        ValueError: If `how` is not one of the three above.

    Example:
        >>> w = pd.Series({'A': 2.0, 'B': -1.0})
        >>> normalise(w, 'gross')
        A    0.666667
        B   -0.333333
        dtype: float64
    """
    if how == 'none':
        return weight.astype(float)
    if how == 'gross':
        scale: float = float(weight.abs().sum())
    elif how == 'net':
        scale = float(weight.sum())
    else:
        raise ValueError(f"how must be 'gross', 'net' or 'none', got {how!r}")
    if abs(scale) < _SCALE_FLOOR:
        warnings.warn(f'normalise({how!r}): denominator {scale:.3e} is at the '
                      'floor; returning a flat book', RuntimeWarning)
        return pd.Series(0.0, index=weight.index)
    return weight/scale


def epo_weight (date: pd.Timestamp, tickers: list[str]|None = None,
                how: str = 'gross')-> pd.Series:
    """EPO weights at one formation date, labelled and scaled.

    `EPO.Config()` must have run first; it precomputes the per-date vol,
    correlation and signal inputs over the whole of `constant.testing_period` in
    a process pool, and `EPO.portfolio_weight` raises rather than doing it
    lazily. Whether the book is long-only was fixed there, by
    `Config(long_only=...)`, so there is nothing to pass per date.

    Args:
        date (pd.Timestamp): Formation date. Keyed into `EPO.correl` as
            `str(date)`, which is the form `EPO.Config` stored -- so it has to
            be a date `Config` covered, i.e. one of `constant.testing_period`.
        tickers (list[str] | None): Accepted for interface uniformity and
            intersected with EPO's own universe if given. EPO reads
            `applicable_ticker.csv` itself in `EPO.utils.compute_date`, so under
            the default backtest configuration this is the same list and the
            intersection is a no-op.
        how (str): Scale, passed to `normalise`. Defaults to 'gross', which is
            what makes an unconstrained book one unit of deployed capital.
            Under `EPO.long_only` the model already returns a book summing to
            one, so all three settings agree up to the renormalisation that
            follows the intersection with `tickers`.

    Returns:
        pd.Series: One weight per ticker EPO priced at `date`.

    Raises:
        RuntimeError: If `EPO.Config()` has not been called.
        KeyError: If `Config` did not cover `date`.
        ValueError: If the long-only solve does not converge.

    Example:
        >>> EPO.Config()
        >>> w = epo_weight(pd.Timestamp('2020-03-31'))
        >>> float(w.abs().sum())
        1.0
    """
    key: str = str(date)
    raw: np.ndarray = EPO(key).portfolio_weight()
    # The column order of the correlation matrix is the only thing that says
    # which ticker each element belongs to; `portfolio_weight` returns a bare
    # array, so reading it against any other list scrambles the cross-section.
    weight: pd.Series = pd.Series(raw, index=EPO.correl[key].columns,
                                  dtype=float)
    weight.index.name = 'ticker'
    if tickers is not None:
        keep: list[str] = [t for t in weight.index if t in set(tickers)]
        weight = weight.loc[keep]
    return normalise(weight, how)


def ppp_weight (model: PPP, date: pd.Timestamp, tickers: list[str],
                n_month: int = 60, long_only: bool = True)-> pd.Series:
    """Parametric-policy weights at one formation date, labelled.

    Args:
        model (PPP): A live `PPP` instance. Reused across dates rather than
            rebuilt, since it also accumulates `theta` and `weight` per date and
            those are worth keeping for inspection.
        date (pd.Timestamp): Formation date; a month end.
        tickers (list[str]): Candidate universe. `find_theta` narrows it to the
            names with a complete characteristic record over the whole
            estimation window, so the result is generally shorter.
        n_month (int): Estimation window length in months. Defaults to 60.
        long_only (bool): Truncate the short leg and renormalise, fitting theta
            under the same constraint. Defaults to True.

    Returns:
        pd.Series: One weight per surviving ticker, summing to one.
            Non-negative under `long_only`; otherwise individual weights may be
            negative, the policy being unconstrained.

    Example:
        >>> PPP.Config()
        >>> w = ppp_weight(PPP(), pd.Timestamp('2020-03-31'), ['AAPL', 'MSFT'])
        >>> float(w.sum())
        1.0
    """
    raw: np.ndarray = model.portfolio_weight(date, tickers, n_month,
                                             long_only=long_only)
    # `(1, n_t)`, not `(n_t,)`, and ordered by `model.ticker` -- which is set as
    # a side effect of the call above and is not the list passed in.
    return pd.Series(np.asarray(raw).reshape(-1), index=model.ticker,
                     dtype=float).rename_axis('ticker')


def proposed_weight (date: pd.Timestamp, tickers: list[str],
                     n_samples: int = 10000, n_lags: int = 4,
                     forward: bool = False, common_theta: bool = True,
                     seed: int|None = None, long_only: bool = True
                     )-> pd.Series:
    """Residual-income model weights at one formation date.

    The only adapter that is close to a pass-through: `RIM_PortOp.weight`
    already returns a ticker-indexed Series summing to one.

    Args:
        date (pd.Timestamp): Formation date.
        tickers (list[str]): Candidate universe, narrowed by `RIM_PortOp` to the
            names with at least `constant.min_char_obs` complete quarters.
        n_samples (int): Simulated paths per ticker. Defaults to 10000.
        n_lags (int): Lags the persistence fit reads. Defaults to 4.
        forward (bool): Elicit the unconditional moments from the *realised*
            future window rather than from the training window. Defaults to
            False.

            **This is lookahead and the returns it produces are not
            tradeable.** `take_training_data` hands `RIM_PortOp` a future window
            running `constant.n_quarter_ahead` quarters past the formation date,
            and under `forward=True` the mean and variance each characteristic
            is simulated around come from that window -- so the portfolio formed
            at `t` has already seen the accounting figures through `t + 3
            years`. Its purpose is to decompose the model's error: run against
            `forward=False` it separates how much of the shortfall is the
            simulation machinery and how much is simply not knowing the future
            moments. Read it as an upper bound on what perfect moment
            forecasting would buy, never as a strategy.

        common_theta (bool): Pool the decay estimate across the universe.
            Defaults to True.
        seed (int | None): Seed for the copula draw. Pass one for a reproducible
            backtest; `run` derives a per-date seed from it so that two
            formation dates do not draw the same paths. Defaults to None.
        long_only (bool): Bound weights to [0, 1]. Defaults to True.

    Returns:
        pd.Series: One weight per ticker, summing to one. A ticker whose
            simulations were all non-finite gets exactly 0.

    Example:
        >>> RIM_PortOp.Config()
        >>> w = proposed_weight(pd.Timestamp('2020-03-31'), tickers, seed=0)
        >>> float(w.sum())
        1.0
    """
    model: RIM_PortOp = RIM_PortOp(date, tickers)
    return model.weight(n_samples=n_samples, n_lags=n_lags, forward=forward,
                        common_theta=common_theta, seed=seed,
                        long_only=long_only).astype(float)


def equal_weight (date: pd.Timestamp, tickers: list[str])-> pd.Series:
    """Equal weights over the candidate universe: the benchmark the tilts start from.

    Worth running alongside the three models rather than quoting from an index
    provider, because it is the same universe on the same dates -- so the
    difference between it and a model is the weighting rule alone, with the
    stock selection screen held fixed. It is also literally `PPP`'s benchmark
    `w_bar`, which makes PPP's contribution readable as the gap between the two.

    Args:
        date (pd.Timestamp): Formation date. Unused; present so the signature
            matches the other adapters.
        tickers (list[str]): Universe to spread the capital over.

    Returns:
        pd.Series: `1/N` on every ticker.

    Raises:
        ValueError: If `tickers` is empty.

    Example:
        >>> equal_weight(pd.Timestamp('2020-03-31'), ['A', 'B'])
        ticker
        A    0.5
        B    0.5
        dtype: float64
    """
    if not tickers:
        raise ValueError(f'{date.date()}: empty universe')
    return pd.Series(1.0/len(tickers), index=pd.Index(tickers, name='ticker'))