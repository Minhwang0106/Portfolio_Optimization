"""Tests for `run.rebuild`, the summary-only path.

Its whole value is that it reproduces what a real run would have written
without running the models, so that is what these check: same numbers as
`save` wrote, and a loud failure when the inputs no longer match the run.
"""
import numpy as np
import pandas as pd
import pytest

from src.Empirical_Analysis.engine import backtest
from src.Empirical_Analysis.metrics import summarise
from src.Empirical_Analysis.run import rebuild, save
from src.Empirical_Analysis.strategy import equal_weight

TICKERS = ['AAA', 'BBB', 'CCC']


@pytest.fixture
def dates()-> list[pd.Timestamp]:
    return list(pd.date_range('2020-03-31', periods=6, freq='ME'))


@pytest.fixture
def returns(dates)-> pd.DataFrame:
    index = list(dates)+[dates[-1]+pd.offsets.MonthEnd(1)]
    rng = np.random.default_rng(0)
    data = rng.normal(0.01, 0.05, size=(len(index), len(TICKERS)))
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name='date'),
                        columns=TICKERS)


@pytest.fixture
def uni(dates)-> dict[pd.Timestamp, list[str]]:
    return {d: list(TICKERS) for d in dates}


@pytest.fixture
def saved_run (tmp_path, dates, returns, uni, monkeypatch):
    """A finished run on disk, with the panels patched to the fixtures."""
    import src.Empirical_Analysis.run as run_mod
    monkeypatch.setattr(run_mod, 'monthly_return', lambda: returns)
    monkeypatch.setattr(run_mod, 'universe', lambda: uni)
    monkeypatch.setattr(run_mod, 'risk_free',
                        lambda: pd.Series(0.001, index=returns.index))

    res = backtest(equal_weight, dates, returns, uni, cost_bps=10.0,
                   name='equal_weight')
    results = {'equal_weight': res}
    table = summarise(results, pd.Series(0.001, index=returns.index),
                      benchmark=None)
    save(results, table, pd.Series(0.001, index=returns.index), tmp_path)
    return tmp_path, table


def test_rebuild_reproduces_the_saved_summary(saved_run, dates):
    out, original = saved_run
    _, rebuilt = rebuild(result_dir=out, dates=dates, sr_benchmark=None)
    pd.testing.assert_frame_equal(rebuilt, original)


def test_rebuild_reproduces_the_saved_returns_exactly(saved_run, dates):
    out, _ = saved_run
    before = pd.read_csv(out/'monthly_returns.csv', index_col=0,
                         parse_dates=True)
    rebuild(result_dir=out, dates=dates, sr_benchmark=None)
    after = pd.read_csv(out/'monthly_returns.csv', index_col=0,
                        parse_dates=True)
    pd.testing.assert_frame_equal(before, after)


def test_rebuild_raises_when_the_cost_no_longer_matches_the_run(saved_run,
                                                               dates):
    """Silently re-costing a run would make every number downstream wrong."""
    out, _ = saved_run
    with pytest.raises(ValueError, match='does not reproduce'):
        rebuild(result_dir=out, dates=dates, cost_bps=250.0,
                sr_benchmark=None)


def test_rebuild_allows_a_deliberate_recost(saved_run, dates):
    out, original = saved_run
    _, table = rebuild(result_dir=out, dates=dates, cost_bps=250.0,
                       sr_benchmark=None, verify=False)
    # A heavier cost has to show up as a worse return, or nothing was recomputed.
    assert (table.loc['ann_return', 'equal_weight']
            < original.loc['ann_return', 'equal_weight'])


def test_rebuild_carries_failures_across(saved_run, dates):
    """A lookup rule cannot fail, so the record has to come from the file."""
    out, _ = saved_run
    pd.DataFrame([{'strategy': 'equal_weight', 'date': dates[0],
                   'error': 'boom'}]).to_csv(out/'failures.csv', index=False)
    results, _ = rebuild(result_dir=out, dates=dates, sr_benchmark=None)
    assert results['equal_weight'].failures[dates[0]] == 'boom'
    # And it survives the round trip back to disk.
    assert (out/'failures.csv').exists()


def test_rebuild_needs_a_run_to_rebuild_from(tmp_path, dates):
    with pytest.raises(FileNotFoundError, match='no weights'):
        rebuild(result_dir=tmp_path, dates=dates)
