"""The backtest loop: turn a per-date weighting rule into a return series.

One loop for all three models, because the thing being compared is the
weighting rule and everything around it has to be identical or the comparison
measures the harness. Specifically:

* **No lookahead.** A weight formed at month end `t` earns the return of the
  month ending at `t + 1` and never the month ending at `t`. Each model applies
  its own publication lag to its own inputs upstream; the engine's job is only
  to not undo that by paying it the return it was fitted on.
* **Holdings drift.** Between rebalances the book is held, not re-set to the
  last target: a position that doubles is twice the weight next month. Skipping
  this quietly rebalances to target every month, which would give a quarterly
  strategy a monthly strategy's turnover and a free contrarian tilt.
* **Turnover is measured against the drifted book**, not against the previous
  target, for the same reason.

A date whose weighting rule raises is recorded and skipped rather than taking
the run down -- across 132 formation dates and a universe that turns over, some
model will fail somewhere, and losing eleven years of results to it is worse
than a gap of one month.
"""
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Below this the book's growth factor `1 + r_p` is not divided by when drifting
# holdings. Reachable only for a levered long-short book that lost its whole
# capital base in one month, where "weight as a fraction of NAV" has stopped
# meaning anything; the positions are carried at their grown value instead.
_NAV_FLOOR: float = 1e-8

# A position smaller than this is not counted as a holding. `!= 0` is the
# obvious test and it is wrong: `port_optimize.port_weight` solves with SLSQP
# under a [0, 1] bound, and the solver leaves the names it wants nothing in at
# ~1e-12 rather than at exactly zero. Counted that way the residual income model
# reports ~138 holdings on a book whose largest weight is 95% and whose real
# breadth is two names -- which is the one number a reader would use to judge
# whether the strategy is diversified.
_WEIGHT_TOLERANCE: float = 1e-6

WeightFn = Callable[[pd.Timestamp, list[str]], pd.Series]


@dataclass
class BacktestResult:
    """Everything one strategy's run produced.

    Attributes:
        name (str): Label carried into the summary table.
        returns (pd.Series): Gross monthly return, indexed by the month end it
            was *earned* in -- one month after the formation date that set the
            weights.
        net_returns (pd.Series): The same, after the transaction cost charged on
            that month's rebalance. Identical to `returns` when `cost_bps` is 0.
        diagnostics (pd.DataFrame): Per-month `net_exposure`, `gross_exposure`,
            `turnover`, `cost`, `n_holdings`, `effective_n`, `max_weight`,
            `entropy` and `unpriced_weight`, on the same index as `returns`.
            Read `entropy` or `effective_n` rather than `n_holdings` to judge
            breadth -- see `_weight_entropy` and `_WEIGHT_TOLERANCE`.
        weights (pd.DataFrame): Target weights at each *formation* date, one
            column per ticker, NaN where not held. Rebalance dates only.
        failures (dict[pd.Timestamp, str]): Formation dates whose weighting rule
            raised, and what it said.
    """
    name: str
    returns: pd.Series
    net_returns: pd.Series
    diagnostics: pd.DataFrame
    weights: pd.DataFrame
    failures: dict[pd.Timestamp, str] = field(default_factory=dict)


def _weight_entropy (held: pd.Series, gross: float)-> float:
    """Shannon entropy of the book, in nats: `-sum p ln p` on `p = |w|/gross`.

    A breadth measure that answers the same question as `effective_n` but
    weights the tail differently: the inverse Herfindahl is driven by the
    largest positions, while entropy also notices whether the remaining capital
    is spread evenly or itself concentrated. An equal-weighted book of `N` names
    scores `ln N`, so `exp(entropy)` is readable as a name count and is what
    makes the two comparable. A single-name book scores 0.

    Taken on absolute weights over gross exposure so that it is defined for a
    long-short book too, where a signed weight has no reading as a probability.
    That makes it a measure of where the *risk* is spread, not of net position.

    Args:
        held (pd.Series): The book, signed weights.
        gross (float): `held.abs().sum()`, passed in since the caller has it.

    Returns:
        float: Entropy in nats, 0 for an empty or fully concentrated book.

    Example:
        >>> import numpy as np
        >>> np.isclose(_weight_entropy(pd.Series([0.25]*4), 1.0), np.log(4))
        True
    """
    if gross <= 0 or not len(held):
        return 0.0
    share: np.ndarray = (held.abs()/gross).to_numpy(dtype=float)
    # Only the non-zero terms: `p ln p` tends to 0 as p -> 0, but `log(0)` is
    # -inf and would take the sum with it.
    share = share[share > 0]
    return float(-(share*np.log(share)).sum())


def _drift (held: pd.Series, realised: pd.Series, port_return: float
            )-> pd.Series:
    """Carry a book forward one month without trading it.

    A position grows by its own return while the whole book grows by the
    portfolio return, so its share of NAV goes to `w(1+r)/(1+r_p)`. Unpriced
    names are carried flat rather than dropped -- dropping them would silently
    reallocate their capital to everything else.
    """
    grown: pd.Series = held*(1+realised.reindex(held.index).fillna(0.0))
    nav: float = 1.0+port_return
    if abs(nav) < _NAV_FLOOR:
        return grown
    return grown/nav


def backtest (weight_fn: WeightFn, dates: Iterable[pd.Timestamp],
              return_df: pd.DataFrame, universe: dict[pd.Timestamp, list[str]],
              rebalance: Iterable[pd.Timestamp]|None = None,
              cost_bps: float = 0.0, name: str = 'strategy',
              verbose: bool = False)-> BacktestResult:
    """Run one weighting rule over a sequence of formation dates.

    Args:
        weight_fn (WeightFn): `(date, tickers) -> pd.Series` indexed by ticker.
            The adapters in `strategy` put each model on this signature; bind
            their extra arguments with `functools.partial`.
        dates (Iterable[pd.Timestamp]): Formation dates, ascending. Every one is
            a *holding* month; `rebalance` decides which are also trading
            months. Defaults elsewhere to `constant.testing_period`.
        return_df (pd.DataFrame): Realised monthly returns from
            `data.monthly_return`, indexed by month end with one column per
            ticker.
        universe (dict[pd.Timestamp, list[str]]): Candidate tickers per date,
            from `data.universe`. A date absent from this mapping is skipped.
        rebalance (Iterable[pd.Timestamp] | None): Subset of `dates` on which
            `weight_fn` is called; on the others the existing book is held and
            drifted. None means rebalance on every date. The first date that
            produces a book is always a rebalance regardless, since there is
            nothing to hold yet.
        cost_bps (float): One-way transaction cost in basis points of traded
            notional, charged against the month following each trade. 10 bps is
            a reasonable large-cap assumption. Defaults to 0.
        name (str): Label for the result. Defaults to 'strategy'.
        verbose (bool): Print each formation date as it is reached. Worth it for
            the proposed model, where a single date takes minutes. Defaults to
            False.

    Returns:
        BacktestResult: See that class. `returns` is indexed by the month the
            return was earned, which is one month later than the formation date
            that set the weights.

    Raises:
        ValueError: If no formation date produced a return at all -- that is a
            wiring problem (wrong dates, empty universe, a weighting rule that
            raises everywhere), not a result.

    Note:
        A ticker held into a month it has no price for contributes nothing to
        that month's return, and its weight is reported in
        `diagnostics['unpriced_weight']` rather than being reallocated. The
        universe screen requires 75 months of trailing prices, so this is a
        delisting between formation and settlement and should stay near zero;
        a large value means the price panel is short, not that the strategy
        did anything.

    Example:
        >>> res = backtest(equal_weight, testing_period, monthly_return(),
        ...                universe(), name='equal weight')
        >>> res.returns.head(2)
        date
        2015-02-28    0.055...
        2015-03-31   -0.013...
    """
    dates = [pd.Timestamp(d) for d in dates]
    rebal: set[pd.Timestamp] = (set(dates) if rebalance is None
                                else {pd.Timestamp(d) for d in rebalance})

    held: pd.Series|None = None
    index: list[pd.Timestamp] = []
    gross_r: list[float] = []
    rows: list[dict] = []
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    failures: dict[pd.Timestamp, str] = {}

    for date in dates:
        # The return the book earns is next month's, never this month's. If the
        # panel does not carry it, the formation date is the last one that can
        # be evaluated and the loop stops rather than dropping a date out of the
        # middle of a compounded series.
        earn_date: pd.Timestamp = date+pd.offsets.MonthEnd(1)
        if earn_date not in return_df.index:
            break
        if verbose:
            print(f'  {name}: {date.date()}', flush=True)

        turnover: float = 0.0
        if date in rebal or held is None:
            tickers: list[str]|None = universe.get(date)
            if not tickers:
                # No candidates: nothing to form. An existing book is held on
                # rather than liquidated, since an empty screen is a data gap
                # and not a signal to go to cash.
                if held is None:
                    continue
            else:
                try:
                    target: pd.Series = weight_fn(date, list(tickers))
                except Exception as e:                       # noqa: BLE001
                    # Deliberately broad. Every failure mode here is a
                    # data-shaped one -- a copula with no association, an
                    # optimiser that will not converge, a ticker with no rows --
                    # and they surface as half a dozen different exception
                    # types. Naming them individually would mean the run dies on
                    # the first one nobody anticipated.
                    failures[date] = f'{type(e).__name__}: {e}'
                    warnings.warn(f'{name} {date.date()}: weighting failed '
                                  f'({type(e).__name__}: {e}); '
                                  f'{"skipping" if held is None else "holding previous book"}',
                                  RuntimeWarning)
                    if held is None:
                        continue
                else:
                    target = target.astype(float)
                    target = target[target.notna()]
                    # Against the *drifted* book, so a strategy is not credited
                    # with the trades it avoided by not rebalancing. `sub` with
                    # fill_value aligns two different universes, which is the
                    # normal case: each model screens its own names.
                    prev: pd.Series = (held if held is not None
                                       else pd.Series(dtype=float))
                    turnover = float(target.sub(prev, fill_value=0.0).abs().sum())
                    target_by_date[date] = target
                    held = target

        assert held is not None
        realised: pd.Series = return_df.loc[earn_date]  #type:ignore
        aligned: pd.Series = realised.reindex(held.index)
        priced: pd.Series = aligned.notna()
        port_return: float = float((held[priced]*aligned[priced]).sum())
        cost: float = turnover*cost_bps/1e4

        gross: float = float(held.abs().sum())
        # Inverse Herfindahl, which answers "how many names is this really?"
        # rather than "how many are non-zero". A book of N equal weights scores
        # N; a book that is 95% one name scores ~1.1 however many rows carry a
        # non-zero. For the residual income model the two differ by two orders
        # of magnitude, and only this one is honest about the concentration.
        sq: float = float((held**2).sum())
        index.append(earn_date)
        entropy: float = _weight_entropy(held, gross)
        gross_r.append(port_return)
        rows.append({'net_exposure': float(held.sum()),
                     'gross_exposure': gross,
                     'turnover': turnover,
                     'cost': cost,
                     'n_holdings': int((held.abs() > _WEIGHT_TOLERANCE).sum()),
                     'effective_n': gross**2/sq if sq > 0 else 0.0,
                     'max_weight': float(held.abs().max()) if len(held) else 0.0,
                     'entropy': entropy,
                     'unpriced_weight': float(held[~priced].abs().sum())})

        held = _drift(held, realised, port_return)

    if not index:
        raise ValueError(f'{name}: no formation date produced a return; check '
                         'the dates, the universe and the weighting rule')

    idx: pd.DatetimeIndex = pd.DatetimeIndex(index, name='date')
    returns: pd.Series = pd.Series(gross_r, index=idx, name=name)
    diagnostics: pd.DataFrame = pd.DataFrame(rows, index=idx)
    net_returns: pd.Series = (returns-diagnostics['cost']).rename(name)
    weights: pd.DataFrame = (
        pd.DataFrame(target_by_date).T.rename_axis('date').sort_index()
        if target_by_date else pd.DataFrame())
    return BacktestResult(name=name, returns=returns, net_returns=net_returns,
                          diagnostics=diagnostics, weights=weights,
                          failures=failures)


def quarter_ends (dates: Iterable[pd.Timestamp])-> list[pd.Timestamp]:
    """The quarter ends among `dates` -- the rebalance schedule for a quarterly model.

    The residual income model fits a copula per ticker and draws ten thousand
    paths for each, so a monthly schedule is a hundred and thirty-two of those.
    Its inputs are quarterly accounting figures besides, and
    `data_generator.accounting_cutoff` backs a mid-quarter formation date off
    two quarters rather than one -- so eleven of every twelve monthly
    rebalances would be re-solving against an accounting panel that had not
    moved.

    Args:
        dates (Iterable[pd.Timestamp]): Candidate dates, typically
            `constant.testing_period`.

    Returns:
        list[pd.Timestamp]: Those that fall on a quarter end, in order.

    Example:
        >>> quarter_ends(pd.date_range('2020-01-31', periods=6, freq='ME'))
        [Timestamp('2020-03-31 00:00:00'), Timestamp('2020-06-30 00:00:00')]
    """
    return [d for d in (pd.Timestamp(x) for x in dates) if d.is_quarter_end]