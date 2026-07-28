"""Tests for `src.Proposed_Model.utils`.

The `test_regression_*` cases cover four defects the first version of this
suite found and that have since been fixed: an unfilled upper triangle, a NaN
slipping past the `isinstance` guard, an `ideal_rank` that assumed the diagonal
always topped its column, and a rank axis mismatch. They are named so that a
future refactor reintroducing any of them fails loudly.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from src.Proposed_Model.utils.dependence_structure import cross_section_order, kendall_matrix


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four columns with a known dependence structure: b tracks a, c opposes a,
    d is independent."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    return pd.DataFrame({
        'a': x,
        'b': x * 2 + rng.normal(size=60) * 0.1,
        'c': -x + rng.normal(size=60) * 0.5,
        'd': rng.normal(size=60),
    })


# --------------------------------------------------------------------------
# kendall_matrix
# --------------------------------------------------------------------------

def test_shape_index_and_dtype(frame):
    m = kendall_matrix(frame)
    assert list(m.columns) == list(frame.columns)
    assert list(m.index) == list(frame.columns)
    assert m.shape == (4, 4)
    assert m.to_numpy().dtype == np.float64


def test_diagonal_is_one(frame):
    m = kendall_matrix(frame)
    assert (np.diag(m.to_numpy()) == 1.0).all()


def test_every_pair_matches_scipy(frame):
    m = kendall_matrix(frame)
    cols = list(frame.columns)
    for i, ci in enumerate(cols):
        for cj in cols[i + 1:]:
            expected = kendalltau(frame[ci], frame[cj]).statistic
            assert m.loc[cj, ci] == pytest.approx(expected)
            assert m.loc[ci, cj] == pytest.approx(expected)


def test_pairwise_dropna_not_listwise(frame):
    """A NaN in `d` must not shrink the sample used for the (a, b) pair."""
    holed = frame.copy()
    holed.loc[holed.index[:20], 'd'] = np.nan
    m = kendall_matrix(holed)
    assert m.loc['b', 'a'] == pytest.approx(
        kendalltau(frame['a'], frame['b']).statistic)
    # ...while the pairs involving `d` do use the reduced sample.
    sub = holed[['a', 'd']].dropna()
    assert m.loc['d', 'a'] == pytest.approx(
        kendalltau(sub['a'], sub['d']).statistic)


def test_perfect_monotone_pairs():
    x = np.arange(30.0)
    m = kendall_matrix(pd.DataFrame({'a': x, 'b': x, 'c': -x}))
    assert m.loc['b', 'a'] == pytest.approx(1.0)
    assert m.loc['c', 'a'] == pytest.approx(-1.0)


def test_single_column():
    m = kendall_matrix(pd.DataFrame({'a': [1.0, 2.0, 3.0]}))
    assert m.equals(pd.DataFrame([[1.0]], index=['a'], columns=['a']))


def test_input_frame_not_mutated(frame):
    before = frame.copy()
    kendall_matrix(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_regression_matrix_is_symmetric(frame):
    """The upper triangle used to be left at the fill value, and
    `cross_section_order` reads *rows*, so every traversal ran on fabricated
    numbers above the diagonal.
    """
    m = kendall_matrix(frame).to_numpy()
    assert np.allclose(m, m.T)


def test_regression_no_cell_is_left_at_a_fill_value(frame):
    """Nothing is silently 0.0: every off-diagonal cell is a computed tau."""
    m = kendall_matrix(frame)
    assert not m.isna().to_numpy().any()
    assert (m.loc['a', 'b'] != 0.0) and (m.loc['b', 'a'] != 0.0)


def test_regression_degenerate_pair_is_nan_not_zero():
    """A constant column makes tau undefined. NaN says 'not measured'; 0.0
    would claim a measured absence of association.
    """
    m = kendall_matrix(pd.DataFrame({'a': np.arange(20.0), 'b': np.ones(20)}))
    assert np.isnan(m.loc['b', 'a'])
    assert np.isnan(m.loc['a', 'b'])
    assert m.loc['a', 'a'] == 1.0


def test_duplicate_column_names_raise_clearly():
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 3)),
                      columns=['a', 'a', 'b'])
    with pytest.raises(ValueError, match='duplicate column names'):
        kendall_matrix(df)


# --------------------------------------------------------------------------
# cross_section_order
# --------------------------------------------------------------------------

def _keys(order: list[dict]) -> list[str]:
    return [next(iter(d)) for d in order]


def _children(order: list[dict]) -> list[str]:
    return [c for d in order for c in next(iter(d.values()))]


@pytest.fixture
def rank_matrix(frame) -> pd.DataFrame:
    """The intended input: column-wise `.rank()` of the association matrix."""
    return kendall_matrix(frame).abs().rank()


def test_every_item_appears_exactly_once_as_a_key(rank_matrix):
    order = cross_section_order(rank_matrix)
    assert sorted(_keys(order)) == sorted(rank_matrix.columns)


def test_every_non_root_appears_exactly_once_as_a_child(rank_matrix):
    order = cross_section_order(rank_matrix)
    root = _keys(order)[0]
    assert sorted(_children(order)) == sorted(
        c for c in rank_matrix.columns if c != root)


def test_default_init_is_alphabetically_first(rank_matrix):
    order = cross_section_order(rank_matrix)
    assert _keys(order)[0] == min(rank_matrix.columns)


def test_explicit_init_is_the_root(rank_matrix):
    order = cross_section_order(rank_matrix, init='c')
    assert _keys(order)[0] == 'c'
    assert sorted(_keys(order)) == sorted(rank_matrix.columns)


def test_unknown_init_raises(rank_matrix):
    with pytest.raises(ValueError, match='init should be one of'):
        cross_section_order(rank_matrix, init='zzz')


def _sym(rows: list[list[float]], labels: str = 'abcd') -> pd.DataFrame:
    return pd.DataFrame(rows, index=list(labels), columns=list(labels))


def test_each_step_follows_the_strongest_tie_still_available():
    """One unambiguous greedy walk: a -> c -> d -> b. From c the strongest
    partner overall is a, but a is visited, so the step falls to d."""
    rm = _sym([[1.0, 0.2, 0.9, 0.1],
               [0.2, 1.0, 0.3, 0.8],
               [0.9, 0.3, 1.0, 0.4],
               [0.1, 0.8, 0.4, 1.0]])
    assert cross_section_order(rm) == [
        {'a': ['c']}, {'c': ['d']}, {'d': ['b']}, {'b': []}]


def test_tie_at_the_strongest_overall_branches_to_multiple_children():
    """b and c are equally and maximally tied to a, so both attach to a; the
    walk then continues from b, the first of the two, and picks up d."""
    rm = _sym([[1.0, 0.7, 0.7, 0.2],
               [0.7, 1.0, 0.3, 0.4],
               [0.7, 0.3, 1.0, 0.5],
               [0.2, 0.4, 0.5, 1.0]])
    assert cross_section_order(rm) == [
        {'a': ['b', 'c']}, {'b': ['d']}, {'c': []}, {'d': []}]


def test_tie_below_the_strongest_overall_takes_only_the_first():
    """From b, c and d tie at 0.3, but b's strongest partner is a (already
    visited), so this is not a branch point and only the first is taken --
    decided by sorted label order, not by the data.
    """
    rm = _sym([[1.0, 0.9, 0.5, 0.4],
               [0.9, 1.0, 0.3, 0.3],
               [0.5, 0.3, 1.0, 0.6],
               [0.4, 0.3, 0.6, 1.0]])
    assert cross_section_order(rm) == [
        {'a': ['b']}, {'b': ['c']}, {'c': ['d']}, {'d': []}]


def test_two_column_matrix():
    rm = pd.DataFrame([[2.0, 1.0], [1.0, 2.0]],
                      index=list('ab'), columns=list('ab'))
    assert cross_section_order(rm) == [{'a': ['b']}, {'b': []}]


def test_single_column_matrix():
    rm = pd.DataFrame([[1.0]], index=['a'], columns=['a'])
    assert cross_section_order(rm) == [{'a': []}]


def test_rank_matrix_not_mutated(rank_matrix):
    before = rank_matrix.copy()
    cross_section_order(rank_matrix)
    pd.testing.assert_frame_equal(rank_matrix, before)


def test_empty_matrix_raises_clearly():
    with pytest.raises(ValueError, match='at least one column'):
        cross_section_order(pd.DataFrame())


def test_non_square_matrix_raises_clearly():
    rm = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]],
                      index=['a', 'b'], columns=['a', 'c'])
    with pytest.raises(ValueError, match='must be square'):
        cross_section_order(rm)


def test_all_nan_column_raises_clearly():
    """A column with no measurable association over the remaining items used
    to die on `max_list[0]` with a bare IndexError.
    """
    rm = pd.DataFrame([[1.0, np.nan, np.nan],
                       [np.nan, 1.0, 0.5],
                       [np.nan, 0.5, 1.0]],
                      index=list('abc'), columns=list('abc'))
    with pytest.raises(ValueError, match="no association available from 'a'"):
        cross_section_order(rm)


def test_regression_branch_fires_on_a_ranked_top_tie():
    """The case the multi-child branch exists for, fed through `.rank()`.

    `ideal_rank = n - 1` could not detect it: two tied top partners occupy
    ranks n-2 and n-1, and `.rank()`'s default `method='average'` collapses
    them to n-1.5, so the equality against n-1 was never true. Comparing
    against the column's own best non-self value has no such blind spot.
    """
    m = _sym([[1.0, 0.7, 0.7, 0.2],
              [0.7, 1.0, 0.3, 0.4],
              [0.7, 0.3, 1.0, 0.5],
              [0.2, 0.4, 0.5, 1.0]])
    n = m.shape[1]
    ranked = m.rank()
    # The blind spot itself, so this test explains its own reason for existing.
    assert ranked['a'][['b', 'c']].max() == n - 1.5 != n - 1
    for rm in (m, ranked):
        assert cross_section_order(rm)[0] == {'a': ['b', 'c']}


def test_regression_perfect_correlation_still_branches():
    """b and c are both perfect matches for a, so |tau| == 1 ties with the
    diagonal and drops the whole column a rank."""
    x = np.arange(30.0)
    m = kendall_matrix(
        pd.DataFrame({'a': x, 'b': x, 'c': x, 'd': -x % 7})).abs()
    for rm in (m, m.rank()):
        order = cross_section_order(rm)
        assert order[0] == {'a': ['b', 'c']}
        assert sorted(_keys(order)) == list('abcd')


def test_regression_column_wise_rank_is_the_supported_input(frame):
    """The pipeline as designed -- `.abs().rank()`, pandas' default axis=0 --
    must give the same walk as the unranked association matrix, because a
    column-wise rank is a monotone transform within each column and the
    traversal only ever compares within one.

    The old code ranked down columns but read across rows, so each row mixed
    numbers from four unrelated rankings.
    """
    m = kendall_matrix(frame).abs()
    assert cross_section_order(m) == cross_section_order(m.rank())


def test_regression_axis0_rank_equals_transposed_axis1_rank(frame):
    """Why axis=0 is the right design: the matrix is symmetric, so column `j`
    of `.rank()` is exactly row `j` of `.rank(axis=1)`. Ranking down columns
    and reading down columns is equivalent to ranking and reading across rows.
    """
    m = kendall_matrix(frame).abs()
    np.testing.assert_allclose(m.rank().to_numpy(),
                               m.rank(axis=1).to_numpy().T)
    assert (cross_section_order(m.rank())
            == cross_section_order(m.rank(axis=1).T))


def test_traversal_is_stable_under_any_column_monotone_transform(rank_matrix):
    baseline = cross_section_order(rank_matrix)
    assert cross_section_order(rank_matrix * 3.0 + 1.0) == baseline
    assert cross_section_order(np.exp(rank_matrix)) == baseline


@pytest.mark.parametrize('seed', range(25))
def test_structure_holds_on_random_matrices(seed):
    """Every item shows up exactly once as a key, every non-root exactly once
    as a child, and the walk always terminates.
    """
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 8))
    cols = [f'x{i}' for i in range(k)]
    df = pd.DataFrame(rng.normal(size=(40, k)), columns=cols)
    order = cross_section_order(kendall_matrix(df).abs())
    assert sorted(_keys(order)) == sorted(cols)
    root = _keys(order)[0]
    assert sorted(_children(order)) == sorted(c for c in cols if c != root)
