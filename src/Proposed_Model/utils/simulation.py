import pandas as pd
import numpy as np
from collections.abc import Sequence
from pyvinecopulib import Bicop

# hinv can land exactly on a bound; the conditional sampler downstream feeds
# these through an inverse CDF, where 0 and 1 become infinities.
_EPS: float = 1e-10

# The four ratios the model simulates, and the order of the `d` axis. 'g' leads
# because it is the variable the outer tree is fitted on and the inner trees are
# rooted at, so it is the one every other column is drawn conditionally on.
# This is also what `RIM_PortOp.depedence_structure` selects before fitting, so
# the trees hold these four and nothing else. Price to book is not among them:
# `data_generator.generator` returns it separately, as a RIM input rather than
# as a characteristic, and it never reaches a copula.
CHARACTERISTICS: tuple[str,...] = ('g','ate','ato','ros')

Structure = list[dict[str, list[tuple[str, Bicop]]]]


def tree_edges (structure:Structure)->list[tuple[str,str,Bicop]]:
    """Flatten `copula_structure` output into its edges.

    Each edge is returned in the order the pair was handed to `Bicop.select`:
    the dict key was the first argument and the name inside the tuple the
    second. That order is what decides `hinv1` against `hinv2` later, so it is
    preserved rather than normalised.

    Args:
        structure (Structure): Output of `dependence_structure.copula_structure`.

    Returns:
        list[tuple[str, str, Bicop]]: One `(arg1, arg2, bicop)` per edge.
    """
    return [(item,child,bicop)
            for d in structure for item, pairs in d.items()
            for child, bicop in pairs]

def tree_nodes (edges:list[tuple[str,str,Bicop]])->list[str]:
    """Every name appearing in `edges`, in first-seen order."""
    nodes: list[str] = []
    for arg1, arg2, _ in edges:
        for name in (arg1,arg2):
            if name not in nodes:
                nodes.append(name)
    return nodes

def tree_adjacency (edges:list[tuple[str,str,Bicop]]
                    )->dict[str,list[tuple[str,Bicop,bool]]]:
    """Undirected adjacency that remembers which side of each fit a node sat on.

    `copula_structure` roots its walk at one item, but sampling may need to
    start somewhere else -- the inner trees are rooted at 'g', which is rarely
    the item the walk began from. Traversing an edge against the direction it
    was fitted in means conditioning on the fit's *second* argument instead of
    its first, so the flag below records which one a node was.

    Args:
        edges (list[tuple[str, str, Bicop]]): Output of `tree_edges`.

    Returns:
        dict[str, list[tuple[str, Bicop, bool]]]: Maps a node to its
            neighbours, each as `(neighbour, bicop, node_is_arg1)`.
    """
    adj: dict[str,list[tuple[str,Bicop,bool]]] = {}
    for arg1, arg2, bicop in edges:
        adj.setdefault(arg1,[]).append((arg2,bicop,True))
        adj.setdefault(arg2,[]).append((arg1,bicop,False))
    return adj

def _step (bicop:Bicop, u_parent:np.ndarray, w:np.ndarray,
           parent_is_arg1:bool)->np.ndarray:
    """One conditional draw across a fitted edge.

    `hinv1(u1, w)` inverts `P(U2 <= u2 | U1 = u1)`, so it produces the second
    argument from the first; `hinv2(w, u2)` goes the other way. Passing the
    pair in the wrong slot is silent -- it samples a valid uniform from the
    wrong conditional -- and for a rotated Clayton, which these fits do allow,
    the two disagree.
    """
    if parent_is_arg1:
        return np.asarray(bicop.hinv1(np.column_stack([u_parent,w])),
                          dtype=np.float64)
    return np.asarray(bicop.hinv2(np.column_stack([w,u_parent])),
                      dtype=np.float64)

def sample_tree (structure:Structure, n_sample:int,
                 rng:np.random.Generator, init:str|None=None,
                 init_u:np.ndarray|None=None)->pd.DataFrame:
    """Draw from the copula of a fitted dependence tree.

    Walks out from `init`, drawing each node conditionally on the neighbour it
    was reached from. Because the tree carries no copulas beyond its own edges,
    two nodes are conditionally independent given the path between them, which
    is the level-1 truncation `copula_structure` fits.

    Args:
        structure (Structure): Output of `copula_structure`.
        n_sample (int): Number of draws.
        rng (np.random.Generator): Source of the uniforms driving each step.
        init (str | None): Node to start from. Defaults to the item
            `copula_structure` began its own walk at, i.e. the one that never
            appears as a child.
        init_u (np.ndarray | None): Values to hold `init` at, shape
            `(n_sample,)` on the open interval (0, 1). Defaults to a fresh
            uniform draw. This is what lets an inner tree hang off a node that
            an outer tree already sampled.

    Returns:
        pd.DataFrame: `n_sample` rows, one column per node, on (0, 1).

    Raises:
        ValueError: If `structure` has no edges, if `init` is not one of its
            nodes, if `init_u` is the wrong shape or outside (0, 1), or if the
            structure is disconnected and so leaves nodes unreachable.
    """
    edges: list[tuple[str,str,Bicop]] = tree_edges(structure)
    if not edges:
        raise ValueError('structure has no edges to sample along')
    nodes: list[str] = tree_nodes(edges)
    adj = tree_adjacency(edges)

    if init is None:
        # The walk's own starting item: the only node never reached as a child.
        children: set[str] = {arg2 for _, arg2, _ in edges}
        roots: list[str] = [name for name in nodes if name not in children]
        if len(roots)!=1:
            raise ValueError(f'expected exactly one root, found {roots}; pass '
                             f'init explicitly')
        init = roots[0]
    elif init not in adj:
        raise ValueError(f'init {init!r} is not a node of this structure, '
                         f'which holds {nodes}')

    if init_u is None:
        u_init: np.ndarray = rng.uniform(size=n_sample)
    else:
        u_init = np.asarray(init_u,dtype=np.float64).reshape(-1)
        if u_init.shape[0]!=n_sample:
            raise ValueError(f'init_u must hold {n_sample} values, got '
                             f'{u_init.shape[0]}')
        if np.any((u_init<=0)|(u_init>=1)) or not np.all(np.isfinite(u_init)):
            raise ValueError('init_u must lie strictly inside (0, 1)')

    drawn: dict[str,np.ndarray] = {init: u_init}
    queue: list[str] = [init]
    while queue:
        parent: str = queue.pop(0)
        for child, bicop, parent_is_arg1 in adj[parent]:
            if child in drawn:
                continue
            w: np.ndarray = rng.uniform(size=n_sample)
            drawn[child] = np.clip(
                _step(bicop,drawn[parent],w,parent_is_arg1),_EPS,1-_EPS)
            queue.append(child)

    missed: list[str] = [name for name in nodes if name not in drawn]
    if missed:
        raise ValueError(f'{missed} are not reachable from {init!r}; the '
                         f'structure is not a connected tree')
    return pd.DataFrame({name: drawn[name] for name in nodes})

def characteristic_order (inner_copula:dict[str,Structure])->list[str]:
    """Every characteristic the fitted trees actually carry, sorted.

    This is what the trees *hold*, which need not be what `sample_joint`
    returns: that is fixed by its `characteristics` argument, and a name the
    trees carry but that sequence omits is still drawn and still conditions
    whatever is drawn through it. Use this to see what a fit contains before
    deciding what to pull out of it.

    Args:
        inner_copula (dict[str, Structure]): One tree of characteristics per
            ticker.

    Returns:
        list[str]: The characteristics, sorted, shared by every ticker.

    Raises:
        ValueError: If `inner_copula` is empty, or if the tickers do not all
            carry the same set of characteristics.
    """
    if not inner_copula:
        raise ValueError('inner_copula is empty')
    per_ticker: dict[str,set[str]] = {
        ticker: set(tree_nodes(tree_edges(structure)))
        for ticker, structure in inner_copula.items()}
    first: str = next(iter(per_ticker))
    shared: set[str] = per_ticker[first]
    odd: dict[str,set[str]] = {t: s for t, s in per_ticker.items()
                               if s!=shared}
    if odd:
        raise ValueError(
            f'tickers do not share one set of characteristics: {first!r} has '
            f'{sorted(shared)} while '
            + '; '.join(f'{t!r} has {sorted(s)}' for t, s in odd.items()))
    return sorted(shared)

def sample_joint (outer_copula:Structure, inner_copula:dict[str,Structure],
                  n_period:int, n_simulation:int, seed:int|None=None,
                  inner_init:str='g',
                  characteristics:Sequence[str]=CHARACTERISTICS
                  )->dict[str,np.ndarray]:
    """Draw the whole panel: the outer tree across tickers, then each inner tree.

    The outer tree ties the tickers together on `inner_init` alone -- it is
    fitted on that one characteristic unstacked across the universe -- so it is
    sampled first and fixes every ticker's `inner_init`. Each ticker's inner
    tree then hangs off that value, spreading it to the ticker's remaining
    characteristics. `inner_init` is therefore the only variable both trees
    touch, and the one the whole cross-sectional dependence travels through.

    Args:
        outer_copula (Structure): First element of
            `RIM_PortOp.depedence_structure`, a tree whose nodes are tickers.
        inner_copula (dict[str, Structure]): Second element, one tree of
            characteristics per ticker.
        n_period (int): Horizon to simulate, in quarters. Axis 0 of the result.
        n_simulation (int): Simulations per period. Axis 1 of the result.
        seed (int | None): Seed for the draw. Defaults to fresh entropy.
        inner_init (str): The characteristic the outer tree is fitted on and
            the inner trees are rooted at. Defaults to 'g'.
        characteristics (Sequence[str]): Order of the `d` axis. Defaults to
            `CHARACTERISTICS`, i.e. `('g', 'ate', 'ato', 'ros')` -- so axis 2
            is `0: g, 1: ate, 2: ato, 3: ros` and `d` is 4, which is every name
            the trees carry. A sequence that omits one of them still draws it,
            and it still conditions whatever is drawn through it; it is only
            left out of the result. The array carries no labels, so this
            sequence is the only record of what axis 2 means.

    Returns:
        dict[str, np.ndarray]: One entry per ticker, each of shape
            `(n_period, n_simulation, len(characteristics))` on (0, 1). With
            the default `characteristics` that is
            `(n_period, n_simulation, 4)`, axis 2 ordered

                index 0 -> 'g'      revenue growth, the tree root
                index 1 -> 'ate'    assets to equity
                index 2 -> 'ato'    asset turnover
                index 3 -> 'ros'    return on sales

            These are copula draws, so they are still uniforms --
            `sampling_distribution.sample_conditional` is what carries them
            onto the scale of the characteristic.

    Raises:
        ValueError: If `n_period` or `n_simulation` is below 1, if a ticker in
            the outer tree has no inner tree, if a ticker's inner tree does not
            contain `inner_init`, or if a ticker's tree does not carry every
            name in `characteristics`.

    Note:
        The `n_period` draws are independent across time. The copula fits hold
        no lag structure -- `copula_structure` scores contemporaneous
        association only -- so there is nothing in them to carry one quarter
        into the next. In this model the serial dependence lives in eq (10)'s
        signal, which is a function of *realised* values rather than of
        uniforms, so it can only enter once these draws have been through
        `sample_conditional`. Feeding a period's output back into the next
        period's signal is what makes the horizon a path rather than
        `n_period` unrelated cross-sections; that recursion is not done here.

    Example:
        >>> model = RIM_PortOp('2020-03-31', ['A', 'AAPL', 'ABC'])
        >>> outer, inner = model.depedence_structure()
        >>> u = sample_joint(outer, inner, n_period=12, n_simulation=1000, seed=0)
        >>> u['AAPL'].shape
        (12, 1000, 4)
        >>> u['AAPL'][:, :, 0]          # 'g' for every period and simulation
        array(...)
    """
    if n_period<1 or n_simulation<1:
        raise ValueError(f'n_period and n_simulation must both be at least 1, '
                         f'got {n_period} and {n_simulation}')
    column: list[str] = list(characteristics)

    rng: np.random.Generator = np.random.default_rng(seed)
    # One flat block of n_period*n_simulation draws, split across time
    # afterwards: the draws are exchangeable along that axis, so where the
    # split falls makes no difference to the law of the result.
    total: int = n_period*n_simulation
    outer_u: pd.DataFrame = sample_tree(outer_copula,total,rng)

    missing: list[str] = [t for t in outer_u.columns if t not in inner_copula]
    if missing:
        raise ValueError(f'no inner tree for {missing}')

    drawn: dict[str,np.ndarray] = {}
    for ticker in outer_u.columns:
        structure: Structure = inner_copula[ticker]
        nodes: list[str] = tree_nodes(tree_edges(structure))
        if inner_init not in nodes:
            raise ValueError(f'{ticker!r} inner tree holds {nodes}, which does '
                             f'not include {inner_init!r}; it cannot be rooted '
                             f'at the variable the outer tree sampled')
        absent: list[str] = [c for c in column if c not in nodes]
        if absent:
            raise ValueError(f'{ticker!r} inner tree holds {nodes} and cannot '
                             f'supply {absent}')
        frame: pd.DataFrame = sample_tree(structure,total,rng,init=inner_init,
                                          init_u=outer_u[ticker].to_numpy())
        # Reindexed to `column`, never to the tree's own walk order: the walk
        # orders differ between tickers, so inheriting it would put different
        # characteristics on the same index of axis 2.
        drawn[ticker] = frame[column].to_numpy().reshape(
            n_period,n_simulation,len(column))
    return drawn
