import pandas as pd
import numpy as np
from typing import cast
from scipy.stats import kendalltau
from pyvinecopulib import Bicop, FitControlsBicop

def kendall_matrix (df: pd.DataFrame)->pd.DataFrame:
    """Kendall tau rank correlation between every pair of columns.

    Each pair is scored on its own overlapping observations -- the two columns
    are dropped to their common non-null rows, not the whole frame -- so a
    column that is mostly missing shrinks only the pairs it takes part in.

    Args:
        df (pd.DataFrame): Numeric panel, one variable per column. Column names
            must be unique.

    Returns:
        pd.DataFrame: Square, symmetric, indexed and columned by `df.columns`,
            with 1.0 on the diagonal.

    Raises:
        ValueError: If `df` has duplicate column names, which would make the
            output's labels ambiguous to index.

    Note:
        A pair whose tau is undefined -- a constant column, or fewer than two
        overlapping observations -- is NaN, not 0.0. The distinction matters:
        0.0 is a measured absence of monotone association, NaN is the absence
        of a measurement. `cross_section_order` rejects NaN rather than
        traversing them.

    Example:
        >>> x = np.arange(30.0)
        >>> kendall_matrix(pd.DataFrame({'a': x, 'b': x, 'c': -x}))
             a    b    c
        a  1.0  1.0 -1.0
        b  1.0  1.0 -1.0
        c -1.0 -1.0  1.0
    """
    col: list[str] = list(df.columns)
    if len(set(col)) != len(col):
        raise ValueError('df must not have duplicate column names')

    # Every pair at once, as three matrix products, rather than a Python loop
    # calling scipy once per pair. The outer tree is a 20x260 frame -- 33,670
    # pairs, 96% of every kendalltau call the model makes -- and at that width
    # the per-pair `df[[a,b]].dropna()` costs more than the statistic does.
    #
    # Writing `s[i,j,c] = sign(x[j,c] - x[i,c])`, zeroed wherever either
    # observation is missing, gives `sum_ij s[:,:,a]*s[:,:,b] = 2(C - D)` over
    # exactly the rows both columns are observed on -- which is `S.T @ S`. The
    # tie correction needs the count of pairs untied in one column and observed
    # in the other, which is `|S|.T @ V`. Summing ordered pairs rather than
    # i < j doubles every term, and tau is their ratio, so the factor cancels.
    values: np.ndarray = df.to_numpy(dtype=np.float64)
    observed: np.ndarray = ~np.isnan(values)
    filled: np.ndarray = np.where(observed, values, 0.0)

    n_obs, n_item = values.shape
    pairwise_valid: np.ndarray = observed[:,None,:] & observed[None,:,:]
    sign: np.ndarray = np.sign(filled[None,:,:]-filled[:,None,:])*pairwise_valid

    flat_sign: np.ndarray = sign.reshape(n_obs*n_obs, n_item)
    flat_valid: np.ndarray = pairwise_valid.reshape(
        n_obs*n_obs, n_item).astype(np.float64)
    untied: np.ndarray = np.abs(flat_sign)

    concordance: np.ndarray = flat_sign.T@flat_sign
    # `untied.T @ flat_valid` counts, for each (a, b), the pairs untied in a and
    # observed in b; its transpose is that count with the roles swapped. tau-b
    # divides by the geometric mean of the two.
    untied_pairs: np.ndarray = untied.T@flat_valid
    denominator: np.ndarray = np.sqrt(untied_pairs*untied_pairs.T)

    # A constant column, or fewer than two overlapping observations, leaves the
    # denominator at zero. That is the undefined case the Note describes, and it
    # has to stay NaN rather than become 0.0 -- `cross_section_order` rejects
    # NaN but would happily traverse a fabricated zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        tau: np.ndarray = np.where(denominator > 0,
                                   concordance/denominator, np.nan)
    # Written last because a wholly constant column scores NaN against itself
    # above, and the diagonal is documented as 1.0 unconditionally.
    np.fill_diagonal(tau, 1.0)

    return pd.DataFrame(tau, index=col, columns=col, dtype=np.float64)

def cross_section_order (rank_matrix:pd.DataFrame, init:str|None=None
                         )->list[dict[str,list]]:
    """Chain the items into a traversal ordered by pairwise association.

    Starting from `init`, repeatedly step to the strongest-associated item not
    yet visited, then repeat from there. The result is a walk in which each
    item is handed the successor it is most closely tied to among whatever is
    still unvisited.

    A step branches instead of chaining when two or more candidates tie for the
    strongest association *and* that association is also the strongest in the
    item's whole row -- i.e. nothing already visited was a better partner. Then
    all of the tied candidates are attached to the current item at once.

    Args:
        rank_matrix (pd.DataFrame): Square matrix of pairwise association,
            indexed and columned identically, where a larger value means a
            stronger tie. Values are only ever compared *within a column*, so
            `kendall_matrix(df).abs().rank()` -- pandas' default column-wise
            rank -- is the intended input, and the unranked
            `kendall_matrix(df).abs()` works just as well. Since the matrix is
            symmetric, column `j` holds the same values as row `j`, so a
            column-wise rank *is* the row-wise ranking of each item, stored
            down the column. Reading it across rows instead would mix numbers
            from unrelated rankings.
        init (str | None): Item to start from. Defaults to the alphabetically
            first.

    Returns:
        list[dict[str, list]]: One single-key dict per item, in visit order.
            The key is the item, the value the list of items attached to it --
            one for a chained step, several for a branching one, empty for a
            leaf. Every item appears exactly once as a key, and every item
            other than `init` exactly once inside a value.

    Raises:
        ValueError: If `rank_matrix` is empty, not square, has duplicate
            labels, `init` is not one of the items, or the row reached is all
            NaN over the unvisited items (no association to step along).

    Example:
        >>> rm = pd.DataFrame(
        ...     [[1.0, 0.2, 0.9, 0.1],
        ...      [0.2, 1.0, 0.3, 0.8],
        ...      [0.9, 0.3, 1.0, 0.4],
        ...      [0.1, 0.8, 0.4, 1.0]],
        ...     index=list('abcd'), columns=list('abcd'))
        >>> cross_section_order(rm)
        [{'a': ['c']}, {'c': ['d']}, {'d': ['b']}, {'b': []}]
    """
    item_list: list[str] = list(rank_matrix.columns)
    item_list.sort()
    if len(item_list)==0:
        raise ValueError('rank_matrix must have at least one column')
    if len(set(item_list))!=len(item_list):
        raise ValueError('rank_matrix must not have duplicate labels')
    if list(rank_matrix.index)!=list(rank_matrix.columns):
        raise ValueError('rank_matrix must be square, with index and columns '
                         'holding the same labels in the same order')

    order: list[dict[str,list]] = []
    pointer: int = 0
    if init is  None:
        init = item_list[0]

    try:
        item_list.remove(init)
    except ValueError :
        raise ValueError(f'init should be one of {item_list}')

    order.append({init: []})
    while len(item_list)>0:
        item: str = next(iter(order[pointer]))
        # Down the column, not across the row: a column-wise `.rank()` stores
        # item `i`'s ranking in column `i`.
        column = rank_matrix[item]
        if not isinstance(column,pd.Series):
            raise ValueError('temp_series should be pandas Series type')
        temp_series = column[item_list]

        max_value = temp_series.max()
        if pd.isna(max_value):
            raise ValueError(f'no association available from {item!r}: every '
                             f'remaining item is NaN in its column')
        max_list: list[str] = list(temp_series[temp_series==max_value].index)

        # "Is this the item's best partner overall?", asked of the values
        # themselves rather than of a rank arithmetic that assumes the diagonal
        # outranks everything -- it does not when an off-diagonal |tau| hits 1
        # and ties with it, which drops the whole column a rank.
        best_overall = column.drop(index=item).max()

        if ((max_value==best_overall)&(len(max_list)>=2)):
            order[pointer][item].extend(max_list)
            for i in max_list:
                order.append({i:[]})
                item_list.remove(i)
        else:
            order[pointer][item].append(max_list[0])
            order.append({max_list[0]:[]})
            item_list.remove(max_list[0])
        pointer+=1
    return order

def copula_structure (df: pd.DataFrame, controls: FitControlsBicop|None=None
                      )->list[dict[str,list[tuple[str,Bicop]]]]:
    """Order a panel by pairwise association and fit a copula to every edge.

    The whole pipeline from raw columns to fitted copulas: score every pair
    with `kendall_matrix`, rank the absolute associations, chain them into a
    walk with `cross_section_order`, then fit each edge of that walk.

    Each item is paired with each of the items attached to it, and that pair's
    family is selected on the pair's own overlapping observations -- the two
    columns are dropped to their common non-null rows, matching how
    `kendall_matrix` scored the association that put the edge there in the
    first place.

    The pair is handed to the fit as pseudo-observations: each column is ranked
    and divided by `len + 1`, mapping it into the open unit square without ever
    reaching 0 or 1, which is the marginal-free input a copula is defined on.

    Args:
        df (pd.DataFrame): Numeric panel, one item per column. Column names
            must be unique.
        controls (FitControlsBicop | None): Selection controls passed to
            `Bicop.select`. Defaults to `FitControlsBicop()`.

    Returns:
        list[dict[str, list[tuple[str, Bicop]]]]: One single-key dict per item,
            in visit order, the value pairing each attached item with the
            `Bicop` fitted to that edge. The keys and the names inside the
            values reproduce the walk, so the structure is readable off the
            result without returning it separately.

    Raises:
        ValueError: Propagated from `kendall_matrix` (duplicate column names)
            or `cross_section_order` (no association to step along, i.e. some
            item is NaN against every item still unvisited).

    Note:
        Traversal stops at the first leaf rather than skipping it. That is safe
        for `cross_section_order` output -- it fills each dict as its pointer
        reaches it, so the dicts left empty are exactly the tail it never
        reached, and they are contiguous at the end.

    Example:
        >>> x = np.random.default_rng(0).normal(size=(200, 2))
        >>> df = pd.DataFrame({'a': x[:, 0], 'b': x[:, 0] + 0.1 * x[:, 1]})
        >>> out = copula_structure(df)
        >>> next(iter(out[0]))
        'a'
    """
    if controls is None:
        controls = FitControlsBicop()

    rank_matrix: pd.DataFrame = kendall_matrix(df).abs().rank()
    struct: list[dict[str,list]] = cross_section_order(rank_matrix)

    copula: list[dict[str,list[tuple[str,Bicop]]]] = []
    for d in struct:
        item: str = next(iter(d))
        child_list: list[str] = d[item]
        copula_list: list[Bicop] = []
        if not child_list:
            break
        for child in child_list:
            cop_obj: Bicop = Bicop()
            # A fresh frame, not a view of `df`: the fit needs the pair's own
            # complete rows, and dropping them in place would reach back into
            # the caller's panel.
            train_data: pd.DataFrame = df[[item,child]].dropna()
            train_arr: np.ndarray = train_data.apply(lambda x: x.rank(
                                    )/(len(x)+1)).to_numpy()
            train_arr = train_arr.reshape(-1,2)
            cop_obj.select(train_arr,controls=controls)
            copula_list.append(cop_obj)
        copula.append({item:list(zip(child_list,copula_list))})
    return copula
