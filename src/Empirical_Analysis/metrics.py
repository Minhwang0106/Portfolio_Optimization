"""Performance statistics for a monthly return series.

Two things here are not the textbook formula, and both are because the three
strategies are not the same kind of portfolio:

* **Excess return is net of `net_exposure * rf`, not of `rf`.** PPP and the
  residual income model are fully invested, so their cash drag is one unit of
  the risk-free rate and the usual `r - rf` is right. EPO under gross
  normalisation is a self-financing long-short book with a net exposure near
  zero: its longs are funded by its shorts, not by capital, so charging it a
  full risk-free rate would subtract a financing cost it never paid and
  understate its Sharpe by roughly `rf/vol`. Scaling by the realised net
  exposure gives the right answer in both cases and needs no per-strategy flag.

* **The certainty equivalent is reported alongside the Sharpe ratio.** Two of
  the three models maximise expected CRRA utility explicitly (`PPP.find_theta`
  and `port_optimize.port_weight`, both at `constant.risk_aversion`), so the
  CE at that same gamma is the objective they were actually fitted on. A Sharpe
  ratio ranks them on a criterion neither one optimises, and the two disagree
  exactly when the return distribution is skewed -- which for a long-short book
  and a concentrated long-only book is most of the time.

A ranking of Sharpe ratios says nothing on its own about whether the gaps are
larger than sampling error, so `summarise` also tests each strategy against a
benchmark -- `equal_weight` by default. The test is Ledoit and Wolf (2008),
*Robust performance hypothesis testing with the Sharpe ratio*, JEF 15, 850-859,
and it lives in `sharpe_inference`; read that module's docstring before quoting
a p-value from this one, in particular on which of its two p-values to believe.
"""
import warnings
import numpy as np
import pandas as pd
from constant import risk_aversion
from .engine import BacktestResult
from .sharpe_inference import sharpe_difference_test

MONTHS_PER_YEAR: int = 12


def _annualise_geometric (returns: pd.Series)-> float:
    """Compound annual growth rate. NaN if the series ever wipes out.

    Compounding through a month of -100% or worse gives a wealth path that is
    zero or negative, and no real growth rate reaches it. Reported as NaN with a
    warning rather than as a complex number or a silently sign-flipped one.
    """
    growth: pd.Series = 1+returns
    if (growth <= 0).any():
        warnings.warn('return series reaches a non-positive wealth level; '
                      'no real annualised return exists', RuntimeWarning)
        return float('nan')
    total: float = float(growth.prod())
    return total**(MONTHS_PER_YEAR/len(returns))-1


def max_drawdown (returns: pd.Series)-> float:
    """Largest peak-to-trough fall in compounded wealth, as a positive fraction.

    Args:
        returns (pd.Series): Monthly simple returns.

    Returns:
        float: In [0, 1] for a series that stays solvent; NaN if wealth reaches
            zero or below, where the ratio to the running peak is meaningless.

    Example:
        >>> max_drawdown(pd.Series([0.1, -0.2, 0.05]))
        0.2
    """
    growth: pd.Series = 1+returns
    if (growth <= 0).any():
        return float('nan')
    wealth: pd.Series = growth.cumprod()
    return float((1-wealth/wealth.cummax()).max())


def crra_certainty_equivalent (returns: pd.Series,
                               gamma: float = risk_aversion)-> float:
    """Annualised certain return an investor with this gamma would swap the series for.

    The objective `PPP` and `port_optimize` are fitted on, read back on the
    realised series and put on a return scale so it can sit next to the CAGR.
    Inverts `PPP.utils.crra_utility`: from `u = E[(1+r)^(1-g)]/(1-g)`, the
    monthly CE is `((1-g)u)^(1/(1-g)) - 1`, compounded to a year.

    Args:
        returns (pd.Series): Monthly simple returns.
        gamma (float): Relative risk aversion. Must not be 1 -- the log case is
            a removable singularity this formula does not cover, the same
            restriction `crra_utility` carries. Defaults to
            `constant.risk_aversion`.

    Returns:
        float: Annualised certainty equivalent. NaN if any month is -100% or
            worse, where CRRA utility is undefined.

    Raises:
        ValueError: If `gamma` is 1.

    Example:
        >>> crra_certainty_equivalent(pd.Series([0.02]*12), gamma=5.0)
        0.26824...
    """
    if gamma == 1:
        raise ValueError('gamma must not be 1; log utility is not covered')
    growth: np.ndarray = 1+returns.to_numpy(dtype=float)
    if (growth <= 0).any():
        return float('nan')
    utility: float = float(np.mean(growth**(1-gamma))/(1-gamma))
    monthly_ce: float = ((1-gamma)*utility)**(1/(1-gamma))-1
    return (1+monthly_ce)**MONTHS_PER_YEAR-1


def excess_return (returns: pd.Series, rf: pd.Series|None = None,
                   net_exposure: pd.Series|float = 1.0)-> pd.Series:
    """Returns net of the risk-free charge actually incurred: `r - net_exposure*rf`.

    Factored out because the Sharpe ratio in `performance` and the Ledoit-Wolf
    test in `summarise` have to be measuring the same series -- a test of a
    difference in Sharpe ratios that used a different excess-return convention
    from the Sharpe ratios above it would be answering a question about numbers
    nobody reported. See the module docstring for why the charge is scaled.

    Args:
        returns (pd.Series): Monthly simple returns, indexed by month end.
        rf (pd.Series | None): Monthly risk-free rate, reindexed onto `returns`.
            None treats it as zero.
        net_exposure (pd.Series | float): Capital actually at risk each month.
            Defaults to 1, a fully invested portfolio.

    Returns:
        pd.Series: Excess returns on `returns`' index.

    Example:
        >>> excess_return(pd.Series([0.02]), pd.Series([0.005]), 0.5).iloc[0]
        0.0175
    """
    if isinstance(net_exposure, pd.Series):
        exposure: pd.Series = net_exposure.reindex(returns.index).fillna(1.0)
    else:
        exposure = pd.Series(float(net_exposure), index=returns.index)
    rf_ser: pd.Series = (pd.Series(0.0, index=returns.index) if rf is None
                         else rf.reindex(returns.index).fillna(0.0))
    return returns-exposure*rf_ser


def performance (returns: pd.Series, rf: pd.Series|None = None,
                 net_exposure: pd.Series|float = 1.0,
                 gamma: float = risk_aversion)-> pd.Series:
    """Summary statistics for one monthly return series.

    Args:
        returns (pd.Series): Monthly simple returns, indexed by month end.
        rf (pd.Series | None): Monthly risk-free rate from `data.risk_free`,
            reindexed onto `returns` here. None treats it as zero, which makes
            the Sharpe ratio a return-to-vol ratio rather than a Sharpe ratio.
        net_exposure (pd.Series | float): The capital actually at risk each
            month, from `BacktestResult.diagnostics['net_exposure']`. The
            risk-free charge is scaled by it -- see the module docstring.
            Defaults to 1, i.e. a fully invested portfolio.
        gamma (float): Relative risk aversion for the certainty equivalent.
            Defaults to `constant.risk_aversion`.

    Returns:
        pd.Series: Named after `returns`, with entries

            * `n_month`, `ann_return`, `ann_vol`, `sharpe`, `sortino`
            * `max_drawdown`, `calmar`
            * `hit_rate`, `skew`, `excess_kurtosis`
            * `ann_crra_ce`, `worst_month`, `best_month`

    Raises:
        ValueError: If `returns` is empty.

    Example:
        >>> performance(res.returns, risk_free(),
        ...             res.diagnostics['net_exposure'])['sharpe']
        np.float64(0.61...)
    """
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError('empty return series')

    excess: pd.Series = excess_return(returns, rf, net_exposure)

    ann_vol: float = float(returns.std(ddof=1))*np.sqrt(MONTHS_PER_YEAR)
    ann_return: float = _annualise_geometric(returns)
    # On excess returns, and annualised by the same root the vol was, so the
    # ratio is the arithmetic mean over the standard deviation rather than a
    # geometric mean over one -- mixing the two is the usual way this number
    # ends up flattering a high-vol strategy.
    sharpe: float = (float(excess.mean())*MONTHS_PER_YEAR/ann_vol
                     if ann_vol > 0 else float('nan'))
    # Target semideviation: the squared shortfalls are averaged over *every*
    # month, not only the losing ones. Averaging over the losers alone is a
    # common variant and it makes the Sortino ratio smaller than the Sharpe
    # ratio for almost any series, which defeats the point of quoting both.
    shortfall: pd.Series = excess.clip(upper=0.0)
    down_vol: float = float(np.sqrt((shortfall**2).mean()))*np.sqrt(
        MONTHS_PER_YEAR)
    sortino: float = (float(excess.mean())*MONTHS_PER_YEAR/down_vol
                      if down_vol > 0 else float('nan'))
    mdd: float = max_drawdown(returns)

    return pd.Series({
        'n_month': float(len(returns)),
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': mdd,
        'calmar': ann_return/mdd if mdd and mdd > 0 else float('nan'),
        'hit_rate': float((returns > 0).mean()),
        'skew': float(returns.skew()),
        'excess_kurtosis': float(returns.kurtosis()),
        'ann_crra_ce': crra_certainty_equivalent(returns, gamma),
        'worst_month': float(returns.min()),
        'best_month': float(returns.max()),
    }, name=returns.name)


def summarise (results: dict[str, BacktestResult]|list[BacktestResult],
               rf: pd.Series|None = None, net: bool = True,
               gamma: float = risk_aversion,
               benchmark: str|None = 'equal_weight',
               block_size: int = 5, n_boot: int = 4999,
               seed: int = 0)-> pd.DataFrame:
    """Stack `performance` across strategies, plus diagnostics and the SR test.

    Args:
        results (dict[str, BacktestResult] | list[BacktestResult]): The runs to
            compare. A list is keyed by each result's own `name`.
        rf (pd.Series | None): Monthly risk-free rate from `data.risk_free`.
        net (bool): Measure on `net_returns` (after transaction costs) rather
            than `returns`. Defaults to True -- gross-of-cost figures flatter
            whichever strategy trades most, which under the default schedule is
            not the same strategy for all three.
        gamma (float): Relative risk aversion for the certainty equivalent.
            Defaults to `constant.risk_aversion`.
        benchmark (str | None): Strategy every other column's Sharpe ratio is
            tested against, by `sharpe_inference.sharpe_difference_test`.
            Defaults to `'equal_weight'`. None skips the test, as does a run
            that does not contain the named benchmark -- with a warning, since
            the usual cause is `--only` having dropped it rather than a
            deliberate choice. The benchmark's own column is NaN in those rows,
            not zero: it has no difference from itself to test.
        block_size (int): Block length for the bootstrap. Defaults to 5; see
            `sharpe_inference.calibrate_block_size` to pick it from the data.
        n_boot (int): Bootstrap resamples. Defaults to 4999. 0 leaves only the
            HAC p-value, which is the fast way to iterate.
        seed (int): Seeds the bootstrap so the p-values reproduce. Defaults to 0.

    Returns:
        pd.DataFrame: One column per strategy, statistics down the rows.
            `performance`'s rows, then `ann_turnover`, `avg_gross_exposure`,
            `avg_net_exposure`, `avg_n_holdings` and `n_failed_date` -- the last
            being formation dates whose weighting rule raised, which is a caveat
            on the column above it rather than a performance figure. Then, when
            a benchmark is tested, `sr_diff` and `sr_diff_se` (annualised) and
            the two p-values `sr_diff_pval_hac` and `sr_diff_pval_boot`.

    Example:
        >>> summarise({'EPO': epo_res, 'PPP': ppp_res}, risk_free()).loc['sharpe']
        EPO    0.58...
        PPP    0.41...
        Name: sharpe, dtype: float64
    """
    if isinstance(results, list):
        results = {r.name: r for r in results}

    excess: dict[str, pd.Series] = {
        label: excess_return(res.net_returns if net else res.returns, rf,
                             res.diagnostics['net_exposure'])
        for label, res in results.items()}
    if benchmark is not None and benchmark not in results:
        warnings.warn(f'benchmark {benchmark!r} is not in this run; the Sharpe '
                      f'ratio difference test is skipped', RuntimeWarning)
        benchmark = None

    columns: dict[str, pd.Series] = {}
    for label, res in results.items():
        series: pd.Series = res.net_returns if net else res.returns
        stats: pd.Series = performance(series, rf,
                                       res.diagnostics['net_exposure'], gamma)
        diag: pd.DataFrame = res.diagnostics
        stats = pd.concat([stats, pd.Series({
            # Annualised by summing over the year, not by scaling the mean:
            # turnover is zero on a hold month and the mean over all months
            # would read a quarterly strategy as trading a third as much per
            # trade rather than a third as often.
            'ann_turnover': float(diag['turnover'].sum())*MONTHS_PER_YEAR
                            /len(diag),
            'avg_gross_exposure': float(diag['gross_exposure'].mean()),
            'avg_net_exposure': float(diag['net_exposure'].mean()),
            'avg_n_holdings': float(diag['n_holdings'].mean()),
            # The breadth figure worth reading. `avg_n_holdings` counts every
            # name the optimiser left off exact zero, which for a bounded SLSQP
            # solve is most of the universe however concentrated the book is.
            'avg_effective_n': float(diag['effective_n'].mean()),
            'avg_max_weight': float(diag['max_weight'].mean()),
            'avg_unpriced_weight': float(diag['unpriced_weight'].mean()),
            'n_failed_date': float(len(res.failures)),
        })])

        if benchmark is not None:
            stats = pd.concat([stats, _sharpe_test_rows(
                label, excess, benchmark, block_size, n_boot, seed)])
        columns[label] = stats
    return pd.DataFrame(columns)


def _sharpe_test_rows (label: str, excess: dict[str, pd.Series], benchmark: str,
                       block_size: int, n_boot: int, seed: int)-> pd.Series:
    """One strategy's Ledoit-Wolf rows, or NaNs where there is nothing to test.

    Every strategy gets the same `seed`, so the resampling is common across
    columns. That is deliberate: the p-values are then differences in the data
    rather than differences in which months each column happened to draw.
    """
    rows: list[str] = ['sr_diff', 'sr_diff_se', 'sr_diff_pval_hac',
                       'sr_diff_pval_boot']
    if label == benchmark:
        return pd.Series(float('nan'), index=rows)
    try:
        test: pd.Series = sharpe_difference_test(
            excess[label], excess[benchmark], block_size=block_size,
            n_boot=n_boot, seed=seed)
    except ValueError as exc:
        # A degenerate column -- too few overlapping months, or a flat series --
        # should cost that column its p-values, not the whole table.
        warnings.warn(f'{label}: Sharpe difference test skipped ({exc})',
                      RuntimeWarning)
        return pd.Series(float('nan'), index=rows)
    return pd.Series({'sr_diff': test['sr_diff'],
                      'sr_diff_se': test['sr_diff_se'],
                      'sr_diff_pval_hac': test['pval_hac'],
                      'sr_diff_pval_boot': test['pval_boot']})


def cumulative_wealth (results: dict[str, BacktestResult]|list[BacktestResult],
                       net: bool = True)-> pd.DataFrame:
    """Wealth path of one unit of capital, one column per strategy.

    Args:
        results: As `summarise`.
        net (bool): Compound the after-cost series. Defaults to True.

    Returns:
        pd.DataFrame: Indexed by month end, starting from the first month any
            strategy earned a return. Strategies that start later are NaN until
            they do, rather than being padded with 1 -- padding would show a
            flat stretch that reads as a period of no return.

    Example:
        >>> cumulative_wealth([epo_res, ppp_res]).iloc[-1]
        EPO    2.31...
        PPP    1.98...
        dtype: float64
    """
    if isinstance(results, list):
        results = {r.name: r for r in results}
    paths: dict[str, pd.Series] = {
        label: (1+(res.net_returns if net else res.returns)).cumprod()
        for label, res in results.items()}
    return pd.DataFrame(paths).sort_index()