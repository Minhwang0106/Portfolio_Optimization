"""The matched-breadth rows: the effective-N floor in both solvers, the target
each matched row is held to, the one-sided Sharpe test, and the difference
panels the tables build from them.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.Proposed_Model.utils.port_optimize import (
    port_weight, moment_port_weight, objective_func, objective_grad,
)
from src.ICC_MVO.utils import max_sharpe_weight
from src.Empirical_Analysis.run import effective_n_target
from src.Empirical_Analysis.sharpe_inference import sharpe_difference_test
from src.Empirical_Analysis.tables import _difference_block, render_latex


def _draws (n_ticker: int = 30, n_sim: int = 2000, seed: int = 0
            )-> pd.DataFrame:
    """Simulated implied returns in `joint_return`'s (ticker, simulation) shape.

    Means spread well apart and little risk anywhere -- the case the floor
    exists for, where the unconstrained CRRA solve holds a single name.
    """
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.02, 0.30, n_ticker)
    draws = mu[:, None]+0.02*rng.standard_normal((n_ticker, n_sim))
    return pd.DataFrame(draws, index=[f'T{i}' for i in range(n_ticker)])


def _effective_n (w)-> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum()**2/(w**2).sum())


def _pair (seed: int = 0, n: int = 132, edge: float = 0.01
           )-> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    b = rng.normal(0.008, 0.04, n)
    x = b+rng.normal(edge, 0.02, n)
    idx = pd.date_range('2015-01-31', periods=n, freq='ME')
    return pd.Series(x, index=idx), pd.Series(b, index=idx)


# --- the CRRA solve ----------------------------------------------------------

def test_objective_grad_matches_finite_differences():
    matrix = _draws().to_numpy().T
    w = np.full(matrix.shape[1], 1/matrix.shape[1])
    numeric = approx_fprime(w, objective_func, 1e-7, matrix)
    assert objective_grad(w, matrix) == pytest.approx(numeric, rel=1e-4,
                                                      abs=1e-8)


def test_port_weight_holds_the_effective_n_floor():
    draws = _draws()
    free = port_weight(draws)
    held = port_weight(draws, n_eff=10.0)
    assert _effective_n(free) < 3            # the concentration it is for
    assert _effective_n(held) >= 10.0-1e-4
    assert held.sum() == pytest.approx(1.0)
    assert (held >= -1e-10).all()
    assert held.index.tolist() == draws.index.tolist()


def test_port_weight_floor_above_the_universe_is_equal_weight():
    held = port_weight(_draws(n_ticker=8), n_eff=50.0)
    assert held.to_numpy() == pytest.approx(np.full(8, 1/8), abs=1e-6)


def test_port_weight_floor_needs_a_long_only_book():
    with pytest.raises(ValueError, match='long-only'):
        port_weight(_draws(), long_only=False, n_eff=5.0)


# --- the maximum-Sharpe solve ------------------------------------------------

def test_max_sharpe_weight_holds_the_effective_n_floor():
    rng = np.random.default_rng(1)
    n = 40
    mu = rng.uniform(0.0, 0.1, n)
    a = rng.standard_normal((n, n))
    sigma = a@a.T/n+0.01*np.eye(n)
    free = max_sharpe_weight(mu, sigma, cap=1.0, min_weight=0.0)
    target = _effective_n(free)+5.0          # binding by construction
    held = max_sharpe_weight(mu, sigma, cap=1.0, min_weight=0.0, n_eff=target)
    assert _effective_n(held) >= target-1e-4
    assert held.sum() == pytest.approx(1.0)

    def sharpe (w):
        return (mu@w)/np.sqrt(w@sigma@w)
    # Binding, so it can only give up Sharpe ratio.
    assert sharpe(held) <= sharpe(free)+1e-9


def test_max_sharpe_weight_floor_and_cap_together():
    rng = np.random.default_rng(2)
    n = 60
    mu = rng.uniform(0.0, 0.1, n)
    sigma = np.diag(rng.uniform(0.01, 0.05, n))
    held = max_sharpe_weight(mu, sigma, cap=0.05, min_weight=0.0, n_eff=30.0)
    assert held.max() <= 0.05+1e-9
    assert _effective_n(held) >= 30.0-1e-4


def test_moment_port_weight_is_max_sharpe_on_the_simulated_moments():
    draws = _draws()
    matrix = draws.to_numpy().T
    got = moment_port_weight(draws, rf=0.01, n_eff=5.0)
    expected = max_sharpe_weight(matrix.mean(axis=0)-0.01,
                                 np.cov(matrix, rowvar=False), cap=1.0,
                                 min_weight=0.0, n_eff=5.0)
    assert got.to_numpy() == pytest.approx(expected)
    assert _effective_n(got) >= 5.0-1e-4


# --- the target a matched row is held to -------------------------------------

def test_effective_n_target_reads_the_book_in_force_at_each_date():
    books = pd.DataFrame({'A': [0.5, 0.25, np.nan],
                          'B': [0.5, 0.25, 1.0],
                          'C': [np.nan, 0.5, np.nan]},
                         index=pd.to_datetime(['2020-01-31', '2020-03-31',
                                               '2020-06-30']))
    got = effective_n_target(books, pd.to_datetime(
        ['2019-12-31', '2020-01-31', '2020-02-29', '2020-03-31', '2020-06-30']))
    assert np.isnan(got.iloc[0])              # before the comparator's first book
    assert got.iloc[1:].tolist() == pytest.approx([2.0, 2.0, 1/0.375, 1.0])


def test_a_row_with_no_comparator_is_solved_without_a_floor():
    from src.Empirical_Analysis.run import _breadth_targets
    assert (_breadth_targets(['proposed_ex_post_msr'], {}, None, [])
            == {'proposed_ex_post_msr': None})


# --- the one-sided Sharpe test -----------------------------------------------

def test_sharpe_one_sided_hac_pvalues_split_the_two_sided_one():
    x, b = _pair()
    two = sharpe_difference_test(x, b, n_boot=0)['pval_hac']
    hi = sharpe_difference_test(x, b, n_boot=0, alternative='greater')['pval_hac']
    lo = sharpe_difference_test(x, b, n_boot=0, alternative='less')['pval_hac']
    assert hi+lo == pytest.approx(1.0)
    assert min(hi, lo) == pytest.approx(two/2)


def test_sharpe_one_sided_bootstrap_follows_the_sign():
    x, b = _pair()
    hi = sharpe_difference_test(x, b, n_boot=999, alternative='greater')
    lo = sharpe_difference_test(x, b, n_boot=999, alternative='less')
    assert hi['sr_diff'] > 0
    assert hi['pval_boot'] < 0.05 < lo['pval_boot']


def test_sharpe_unknown_alternative_raises():
    x, b = _pair()
    with pytest.raises(ValueError, match='alternative must be'):
        sharpe_difference_test(x, b, alternative='bigger')


# --- the difference panels ---------------------------------------------------

def test_difference_block_runs_statistic_by_statistic_and_renders():
    x, b = _pair()
    returns = pd.DataFrame({'s_net': x, 'b_net': b, 'rf': 0.001})
    frame, kinds, stars = _difference_block(
        returns, [('S', 's', 'b')], 'Panel B: S minus B', n_boot=99)
    assert frame.shape == (9, 1)
    assert (frame.index.get_level_values(1).unique().tolist()
            == ['mean_return', 'sharpe', 'crra_ce'])
    assert kinds.tolist() == ['pct', 'num', 'pval', 'num', 'num', 'pval',
                              'pct', 'num', 'pval']
    # A one-sided test of a strategy built to win: every difference positive.
    assert (frame.iloc[::3, 0] > 0).all()
    tex = render_latex([(frame, kinds, stars)], 'caption', 'tab:x', ['note'])
    assert 'Panel B: S minus B' in tex and '\\quad $t$-statistic' in tex
