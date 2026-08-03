"""Tests for `src.Empirical_Analysis`.

Everything here runs on synthetic panels. The point of the backtest harness is
that it is the *same* harness for every model, so what has to be pinned down is
the harness's own arithmetic -- when a weight earns a return, how a held book
drifts, what turnover is measured against -- and none of that needs the real
data to check. Wiring it to the real panels instead would make the suite depend
on the CSVs being present and current, and would hide an off-by-one month behind
eleven years of noise.

The `test_regression_*` cases cover the defects this harness exists to prevent,
and are named so a refactor reintroducing one fails loudly rather than quietly
producing a better-looking Sharpe ratio.
"""

import numpy as np
import pandas as pd
import pytest

from src.Empirical_Analysis.engine import backtest, quarter_ends
from src.Empirical_Analysis.metrics import (
    crra_certainty_equivalent,
    max_drawdown,
    performance,
    summarise,
)
from src.Empirical_Analysis.strategy import equal_weight, normalise

TICKERS = ['AAA', 'BBB', 'CCC']


@pytest.fixture
def dates() -> list[pd.Timestamp]:
    """Six month ends starting at a quarter end."""
    return list(pd.date_range('2020-03-31', periods=6, freq='ME'))


@pytest.fixture
def returns(dates) -> pd.DataFrame:
    """A return panel covering one month past the last formation date.

    The extra row is what a weight formed on the last date would earn, so its
    presence is what decides whether the loop evaluates that date at all.
    """
    index = list(dates) + [dates[-1] + pd.offsets.MonthEnd(1)]
    data = np.tile(np.array([0.10, 0.00, -0.05]), (len(index), 1))
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name='date'),
                        columns=TICKERS)


@pytest.fixture
def uni(dates) -> dict[pd.Timestamp, list[str]]:
    return {d: list(TICKERS) for d in dates}


# --------------------------------------------------------------------------
# normalise
# --------------------------------------------------------------------------

def test_normalise_gross_scales_to_unit_absolute_sum():
    w = normalise(pd.Series({'A': 2.0, 'B': -1.0}), 'gross')
    assert w.abs().sum() == pytest.approx(1.0)
    assert w['A'] == pytest.approx(2 / 3)


def test_normalise_net_scales_to_unit_sum():
    w = normalise(pd.Series({'A': 2.0, 'B': -1.0}), 'net')
    assert w.sum() == pytest.approx(1.0)


def test_normalise_none_is_identity():
    raw = pd.Series({'A': 2.0, 'B': -1.0})
    pd.testing.assert_series_equal(normalise(raw, 'none'), raw)


def test_normalise_rejects_unknown_scheme():
    with pytest.raises(ValueError, match='gross'):
        normalise(pd.Series({'A': 1.0}), 'sideways')


def test_normalise_flat_book_rather_than_dividing_by_zero():
    """A net-zero vector under 'net' must not be levered up by rounding noise."""
    with pytest.warns(RuntimeWarning, match='floor'):
        w = normalise(pd.Series({'A': 1.0, 'B': -1.0}), 'net')
    assert (w == 0).all()


# --------------------------------------------------------------------------
# engine: the no-lookahead contract
# --------------------------------------------------------------------------

def test_regression_weight_earns_the_following_month(dates, returns, uni):
    """A weight formed at `t` earns the month ending at `t+1`, never at `t`.

    Paying it the return of the month it was fitted on is the single defect that
    would make every strategy here look good, so it is pinned twice: the return
    index is shifted a month past the formation dates, and the value is the one
    from the later month.
    """
    returns.loc[dates[0]] = [99.0, 99.0, 99.0]        # must never be earned
    returns.loc[dates[0] + pd.offsets.MonthEnd(1)] = [0.02, 0.02, 0.02]
    res = backtest(equal_weight, dates, returns, uni)

    assert res.returns.index[0] == dates[0] + pd.offsets.MonthEnd(1)
    assert res.returns.iloc[0] == pytest.approx(0.02)


def test_stops_when_the_next_month_is_unpriced(dates, returns, uni):
    """The final formation date is dropped rather than earning a partial month."""
    truncated = returns.drop(index=returns.index[-1])
    res = backtest(equal_weight, dates, truncated, uni)
    assert len(res.returns) == len(dates) - 1


def test_raises_when_nothing_could_be_evaluated(dates, returns, uni):
    early = returns.loc[:dates[0] - pd.offsets.MonthEnd(1)]
    with pytest.raises(ValueError, match='no formation date'):
        backtest(equal_weight, dates, early, uni)


# --------------------------------------------------------------------------
# engine: holdings drift
# --------------------------------------------------------------------------

def test_regression_held_book_drifts_between_rebalances(dates, returns, uni):
    """A position that outperforms is a bigger weight next month.

    Re-setting to the target every month instead would give a quarterly
    strategy a monthly strategy's turnover and a free contrarian tilt, and the
    return series would silently be a different strategy's.
    """
    res = backtest(equal_weight, dates, returns, uni,
                   rebalance=[dates[0]])                # rebalance once, then hold

    first = res.returns.iloc[0]
    assert first == pytest.approx((0.10 + 0.00 - 0.05) / 3)

    # AAA grew 10% while the book grew `first`, so its share of NAV rises.
    expected_aaa = (1 / 3) * 1.10 / (1 + first)
    second = res.returns.iloc[1]
    drifted = pd.Series({'AAA': expected_aaa,
                         'BBB': (1 / 3) * 1.00 / (1 + first),
                         'CCC': (1 / 3) * 0.95 / (1 + first)})
    assert second == pytest.approx(float((drifted * [0.10, 0.0, -0.05]).sum()))
    assert res.diagnostics['net_exposure'].iloc[1] == pytest.approx(1.0)


def test_rebalance_schedule_limits_when_the_rule_is_called(dates, returns, uni):
    calls: list[pd.Timestamp] = []

    def counting(date, tickers):
        calls.append(date)
        return equal_weight(date, tickers)

    quarterly = quarter_ends(dates)
    backtest(counting, dates, returns, uni, rebalance=quarterly)
    assert calls == quarterly


def test_quarter_ends_picks_only_quarter_ends(dates):
    assert quarter_ends(dates) == [pd.Timestamp('2020-03-31'),
                                   pd.Timestamp('2020-06-30')]


# --------------------------------------------------------------------------
# engine: turnover and cost
# --------------------------------------------------------------------------

def test_regression_turnover_is_measured_against_the_drifted_book(dates, returns,
                                                                  uni):
    """Not against the previous *target*.

    Measured against the target, a strategy would be charged nothing for the
    trades it needs to undo its own drift -- so a buy-and-hold book would report
    zero turnover at a rebalance that actually has to sell its winners back to
    equal weight.
    """
    res = backtest(equal_weight, dates, returns, uni,
                   rebalance=[dates[0], dates[2]])

    first_trade = res.diagnostics['turnover'].iloc[0]
    assert first_trade == pytest.approx(1.0)          # buying the book from cash

    # Two months of drift away from equal weight, then a trade back to it.
    rebalance_trade = res.diagnostics['turnover'].iloc[2]
    assert rebalance_trade > 0
    assert res.diagnostics['turnover'].iloc[1] == 0.0  # a hold month


def test_cost_hits_the_month_the_trade_is_held_into(dates, returns, uni):
    res = backtest(equal_weight, dates, returns, uni,
                   rebalance=[dates[0]], cost_bps=100.0)
    charged = res.diagnostics['cost'].iloc[0]
    assert charged == pytest.approx(1.0 * 100.0 / 1e4)
    assert res.net_returns.iloc[0] == pytest.approx(res.returns.iloc[0] - charged)
    # A hold month trades nothing and is charged nothing.
    assert res.diagnostics['cost'].iloc[1] == 0.0
    assert res.net_returns.iloc[1] == pytest.approx(res.returns.iloc[1])


# --------------------------------------------------------------------------
# engine: degenerate inputs
# --------------------------------------------------------------------------

def test_unpriced_holding_contributes_nothing_and_is_reported(dates, returns,
                                                              uni):
    """A delisting is not reallocated to the survivors behind the caller's back."""
    earn = dates[0] + pd.offsets.MonthEnd(1)
    returns.loc[earn, 'CCC'] = np.nan
    res = backtest(equal_weight, dates, returns, uni)

    assert res.returns.iloc[0] == pytest.approx((0.10 + 0.00) / 3)
    assert res.diagnostics['unpriced_weight'].iloc[0] == pytest.approx(1 / 3)


def test_failing_rule_holds_the_previous_book_and_is_recorded(dates, returns,
                                                              uni):
    def flaky(date, tickers):
        if date == dates[2]:
            raise RuntimeError('no association available')
        return equal_weight(date, tickers)

    with pytest.warns(RuntimeWarning, match='weighting failed'):
        res = backtest(flaky, dates, returns, uni)

    assert dates[2] in res.failures
    assert 'no association available' in res.failures[dates[2]]
    # Eleven years must not be lost to one bad date.
    assert len(res.returns) == len(dates)


def test_failing_first_date_is_skipped_not_held(dates, returns, uni):
    def flaky(date, tickers):
        if date == dates[0]:
            raise RuntimeError('nothing to hold yet')
        return equal_weight(date, tickers)

    with pytest.warns(RuntimeWarning):
        res = backtest(flaky, dates, returns, uni)
    assert len(res.returns) == len(dates) - 1


def test_empty_universe_is_skipped(dates, returns, uni):
    del uni[dates[0]]
    res = backtest(equal_weight, dates, returns, uni)
    assert res.returns.index[0] == dates[1] + pd.offsets.MonthEnd(1)


def test_equal_weight_rejects_an_empty_universe(dates):
    with pytest.raises(ValueError, match='empty universe'):
        equal_weight(dates[0], [])


# --------------------------------------------------------------------------
# engine: breadth diagnostics
# --------------------------------------------------------------------------

def test_regression_solver_dust_is_not_counted_as_a_holding(dates, returns, uni):
    """`held != 0` reports a one-stock book as fully diversified.

    `port_optimize.port_weight` bounds its SLSQP solve to [0, 1] and leaves the
    names it wants nothing in at ~1e-12 rather than at exactly zero, so the
    naive count made the residual income model read as ~138 holdings on a book
    whose largest weight was 95%.
    """
    def concentrated(date, tickers):
        return pd.Series({'AAA': 1.0 - 2e-12, 'BBB': 1e-12, 'CCC': 1e-12})

    res = backtest(concentrated, dates, returns, uni)
    assert res.diagnostics['n_holdings'].iloc[0] == 1
    assert res.diagnostics['effective_n'].iloc[0] == pytest.approx(1.0)
    assert res.diagnostics['max_weight'].iloc[0] == pytest.approx(1.0)


def test_effective_n_is_the_name_count_for_an_equal_weighted_book(dates, returns,
                                                                  uni):
    res = backtest(equal_weight, dates, returns, uni)
    assert res.diagnostics['effective_n'].iloc[0] == pytest.approx(len(TICKERS))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_max_drawdown_known_path():
    assert max_drawdown(pd.Series([0.10, -0.20, 0.05])) == pytest.approx(0.20)


def test_max_drawdown_is_nan_once_wealth_is_wiped_out():
    assert np.isnan(max_drawdown(pd.Series([0.10, -1.50])))


def test_annualised_return_is_nan_rather_than_complex_after_a_wipeout():
    with pytest.warns(RuntimeWarning, match='non-positive wealth'):
        stats = performance(pd.Series([0.10, -1.50],
                                      index=pd.date_range('2020-01-31',
                                                          periods=2, freq='ME')))
    assert np.isnan(stats['ann_return'])


def test_crra_certainty_equivalent_of_a_constant_series_is_that_return():
    """With no dispersion there is no risk premium to give up."""
    constant = pd.Series([0.02] * 12)
    assert crra_certainty_equivalent(constant, gamma=5.0) == pytest.approx(
        1.02 ** 12 - 1)


def test_crra_certainty_equivalent_is_below_the_mean_when_returns_disperse():
    risky = pd.Series([0.12, -0.08] * 6)
    flat = pd.Series([0.02] * 12)
    assert (crra_certainty_equivalent(risky, gamma=5.0)
            < crra_certainty_equivalent(flat, gamma=5.0))


def test_crra_rejects_log_utility():
    with pytest.raises(ValueError, match='gamma'):
        crra_certainty_equivalent(pd.Series([0.01]), gamma=1)


def test_regression_riskfree_charge_scales_with_net_exposure():
    """A self-financing long-short book is not charged a full risk-free rate.

    Its longs are funded by its shorts rather than by capital, so subtracting
    `rf` outright would take off a financing cost it never paid and understate
    its Sharpe ratio by roughly `rf / vol`.
    """
    index = pd.date_range('2020-01-31', periods=24, freq='ME')
    rets = pd.Series(np.tile([0.01, 0.02], 12), index=index)
    rf = pd.Series(0.005, index=index)

    invested = performance(rets, rf, net_exposure=1.0)
    neutral = performance(rets, rf, net_exposure=0.0)
    assert neutral['sharpe'] > invested['sharpe']
    # The neutral book is charged nothing, so its Sharpe is the raw ratio.
    assert neutral['sharpe'] == pytest.approx(
        performance(rets, None)['sharpe'])


def test_summarise_reports_one_column_per_strategy(dates, returns, uni):
    res = backtest(equal_weight, dates, returns, uni, name='ew')
    # No benchmark: one strategy has nothing to be tested against. The Sharpe
    # test has its own file.
    table = summarise({'ew': res}, rf=None, benchmark=None)
    assert list(table.columns) == ['ew']
    for row in ('ann_return', 'sharpe', 'ann_crra_ce', 'ann_turnover',
                'n_failed_date'):
        assert row in table.index


def test_annualised_turnover_counts_trades_not_months(dates, returns, uni):
    """A quarterly strategy trades a third as *often*, not a third as *much*."""
    monthly = backtest(equal_weight, dates, returns, uni, name='m')
    quarterly = backtest(equal_weight, dates, returns, uni,
                         rebalance=quarter_ends(dates), name='q')
    table = summarise([monthly, quarterly], rf=None, benchmark=None)
    assert table.loc['ann_turnover', 'm'] > table.loc['ann_turnover', 'q']