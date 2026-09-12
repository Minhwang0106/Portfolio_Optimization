"""The three tables the paper reports, as CSV and as LaTeX.

`run.py` writes everything a backtest produces to `RAW_BACKTEST_DIR`, most of
which no table ever shows -- per-date weights, per-month diagnostics, the
wealth path. This module is the other half: it selects the numbers that
actually appear in the paper and writes them twice, once as CSV in
`PROCESSED_BACKTEST_DIR` for anything downstream that wants to read them and
once as `.tex` in `LATEX_DIR` to be `\\input{}` straight into Overleaf. Same
content, two forms. Nothing here recomputes a backtest; it reads
`raw_backtest/` and the two inference modules.

The point of generating the `.tex` rather than typing it is that the backtest
gets re-run. A hand-typed table silently goes stale, and a stale number in a
published table is the kind of error that survives review.

The three tables:

* **Table 1** -- ex post. Bielstein and Hanauer's portfolio against the proposed
  model four ways, a 2x2: the objective, maximum Sharpe on its simulated
  moments or CRRA over the whole simulation, crossed with the breadth, held to
  B&H's or left free. B&H against max Sharpe at its breadth isolates the
  inputs; each pair of columns isolates one of the other two. Panel A the
  levels, Panel B each proposed row minus B&H.
* **Table 2** -- historical, i.e. implementable. Equal weight, EPO, PPP, and the
  proposed model at EPO's breadth, at PPP's, and unconstrained. Panel A the
  levels, Panel B every row minus equal weight, Panel C each matched row minus
  the comparator whose breadth it holds.
* **Table 3** -- the thought experiment of `experiment_thought`, unchanged.

Every difference test in Tables 1 and 2 is one-sided, `H1`: the row beats its
benchmark -- the question the paper puts to each comparison, fixed before the
numbers. Within a panel the statistics run mean return, then Sharpe ratio, then
certainty equivalent: from the one that ignores risk entirely to the one that
penalises it hardest, so reading down a panel is reading the significance decay
a volatility gap produces.

Each table is built from whichever of its strategies `summary.csv` holds, and
Table 1 is skipped, with a warning, until `icc_mvo_ex_post` has been run.

LaTeX conventions. Output uses `booktabs` and `threeparttable` and nothing
else, so it compiles in any Overleaf project without a preamble hunt. Columns
are plain `r` with a fixed number of decimals per row block, which aligns as
well as `siunitx` for these widths and cannot break on a version difference.
Each file carries its required packages in a leading comment.

Run:
    python -m src.Empirical_Analysis.tables
    python -m src.Empirical_Analysis.tables --n-sim 1000   # faster Table 3
"""
import argparse
import warnings
from pathlib import Path
import pandas as pd

from constant import (RAW_BACKTEST_DIR, PROCESSED_BACKTEST_DIR, LATEX_DIR,
                      risk_aversion, backtest_cost_bps)
from .sharpe_inference import sharpe_difference_test
from .ce_inference import ce_difference_test
from .mean_inference import mean_difference_test
from . import experiment_thought as thought

# Table 1, ex post: B&H first, then the proposed model as a 2x2 -- the two
# objectives at B&H's breadth, then the two unconstrained. Every column after
# B&H is the proposed model, which the notes say rather than each header.
EX_POST_BENCHMARK: str = 'icc_mvo_ex_post'
EX_POST_ORDER: tuple[str, ...] = ('icc_mvo_ex_post',
                                  'proposed_ex_post_msr_icc_n',
                                  'proposed_ex_post_icc_n',
                                  'proposed_ex_post_msr', 'proposed_ex_post')
EX_POST_LABEL: dict[str, str] = {
    'icc_mvo_ex_post': 'B&H',
    'proposed_ex_post_msr_icc_n': 'Max Sharpe, B&H $N$',
    'proposed_ex_post_icc_n': 'CRRA, B&H $N$',
    'proposed_ex_post_msr': 'Max Sharpe, unconstrained',
    'proposed_ex_post': 'CRRA, unconstrained'}

# Table 2, historical: the benchmark, the two models from the literature, then
# the proposed model at each one's breadth and unconstrained.
HISTORICAL_BENCHMARK: str = 'equal_weight'
HISTORICAL_ORDER: tuple[str, ...] = ('equal_weight', 'epo', 'ppp',
                                     'proposed_historical_epo_n',
                                     'proposed_historical_ppp_n',
                                     'proposed_historical')
HISTORICAL_LABEL: dict[str, str] = {
    'equal_weight': 'Equal-weight', 'epo': 'EPO', 'ppp': 'PPP',
    'proposed_historical_epo_n': 'Proposed, EPO $N$',
    'proposed_historical_ppp_n': 'Proposed, PPP $N$',
    'proposed_historical': 'Proposed, unconstrained'}
# Table 2's Panel C: each matched row against the comparator whose breadth it
# holds -- the comparison the matching exists to make.
MATCHED: tuple[tuple[str, str], ...] = (('proposed_historical_epo_n', 'epo'),
                                        ('proposed_historical_ppp_n', 'ppp'))

# H1 of every difference test in Tables 1 and 2: the row beats its benchmark.
ALTERNATIVE: str = 'greater'

# Everything from `metrics.SUMMARY_ROWS` except `sharpe_pval`, a two-sided test
# against equal weight; the tests the tables report are their own panels.
PERFORMANCE_ROWS: dict[str, str] = {
    'ann_return': 'Annualised return (%)',
    'ann_vol': 'Annualised volatility (%)',
    'max_drawdown': 'Maximum drawdown (%)',
    'sharpe': 'Sharpe ratio',
    'ann_crra_ce': 'CRRA certainty equivalent (%)',
    'ann_turnover': 'Annualised turnover',
    'avg_effective_n': 'Average effective N'}
PERFORMANCE_KIND: dict[str, str] = {
    'Annualised return (%)': 'pct', 'Annualised volatility (%)': 'pct',
    'Maximum drawdown (%)': 'pct', 'Sharpe ratio': 'num',
    'CRRA certainty equivalent (%)': 'pct', 'Annualised turnover': 'num',
    'Average effective N': 'num'}

PANEL_LEVELS: str = 'Panel A: Performance'

# The difference panels of Tables 1 and 2: per statistic, the difference, its
# t-statistic and its bootstrap p-value. Mean first, certainty equivalent last;
# see the module docstring.
DIFFERENCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ('mean_return', 'Mean return (pp)', 'pct'),
    ('sharpe', 'Sharpe ratio', 'num'),
    ('crra_ce', 'CRRA certainty equivalent (pp)', 'pct'))
T_ROW: str = '\\quad $t$-statistic'
P_ROW: str = '\\quad Bootstrap $p$-value'

# Table 3's test panels, in the same order.
TEST_PANEL: tuple[tuple[str, str, str], ...] = (
    ('mean_return', 'Panel A: Mean return', 'pct'),
    ('sharpe', 'Panel B: Sharpe ratio', 'num'),
    ('crra_ce', 'Panel C: CRRA certainty equivalent', 'pct'))
DIFFERENCE: str = 'Difference'
T_STAT: str = '$t$-statistic'
P_VALUE: str = 'Bootstrap $p$-value'

# Table 3's panels carry different columns from each other, so they are three
# tabulars in one float rather than one frame; the panel name has to travel
# with each of them.
PANEL_ASSETS: str = 'Panel A: Asset characteristics'
PANEL_TESTS: str = 'Panel B: Difference tests, A minus B'
PANEL_MONTE: str = 'Panel C: Distribution of $t$-statistics across realisations'

STAR_LEVEL: tuple[tuple[float, str], ...] = ((0.01, '***'), (0.05, '**'),
                                             (0.10, '*'))


def _stars (pval: float)-> str:
    """Significance markers at the 1%, 5% and 10% levels.

    Emitted as `$^{***}$` rather than a bare `***`, which text mode would set
    on the baseline and push the column out of alignment.
    """
    if pd.isna(pval):
        return ''
    for level, mark in STAR_LEVEL:
        if pval < level:
            return f'$^{{{mark}}}$'
    return ''


def _escape (text: str)-> str:
    """Escape the LaTeX specials that appear in these labels.

    Deliberately narrow. `$` and `\\` are left alone because the row labels
    carry maths (`$t$-statistic`, `$\\gamma$`) on purpose.
    """
    for char in ('%', '&', '#', '_'):
        text = text.replace(char, '\\'+char)
    return text


def _cell (value: float, kind: str, star: str = '')-> str:
    """One formatted table entry.

    Args:
        value (float): The number.
        kind (str): `'pct'` and `'share'` scale by 100, `'num'` does not, and
            all three show two decimals; `'pval'` shows three in square
            brackets.
        star (str): Significance markers appended after the number.

    Returns:
        str: The cell, or `--` for a missing value.

    Note:
        Every kind but `'pval'` gives two decimals, and that is what makes the
        columns line up. These are plain `r` columns, so cells align on their
        last character rather than on the decimal point -- mixing a one-decimal
        share with a two-decimal statistic in the same column, as an earlier
        version did in Table 3's Panel C, leaves the decimal points visibly
        staggered. Uniform decimals make right-alignment and decimal alignment
        the same thing, without needing `siunitx` or `dcolumn`. `'pval'` is the
        deliberate exception: the brackets already mark it as a different kind
        of quantity, and bracketed p-values are conventional.
    """
    if pd.isna(value):
        return '--'
    if kind == 'pval':
        body: str = f'[{value:.3f}]'
    elif kind in ('pct', 'share'):
        body = f'{100*value:.2f}'
    else:
        body = f'{value:.2f}'
    return body+star


def _kind_frame (kinds: pd.Series|pd.DataFrame,
                 frame: pd.DataFrame)-> pd.DataFrame:
    """Broadcast a per-row format spec to a per-cell one.

    Most rows are one unit across, so a `Series` keyed by row label is the
    natural way to write them. A few are not -- Table 3's `Difference` row runs
    a percentage, a ratio and another percentage across its three columns --
    and those pass a `DataFrame` instead.
    """
    if isinstance(kinds, pd.DataFrame):
        return kinds.reindex(index=frame.index, columns=frame.columns)
    return pd.DataFrame({c: kinds.reindex(frame.index)
                         for c in frame.columns})


def _pad (cells: list[str], n_col: int)-> str:
    """Right-align a block's cells inside a tabular that may be wider.

    A block with fewer columns than the table is flushed to the right, so every
    panel's numbers land in the same place regardless of how many it has. See
    `render_latex` for why the tabular is shared in the first place.
    """
    return ' & '.join(['']*(n_col-len(cells))+cells)


def _block_lines (frame: pd.DataFrame, kinds: pd.Series|pd.DataFrame,
                  stars: pd.DataFrame|None, n_col: int)-> list[str]:
    """One block's column header, panel titles and data rows.

    Args:
        frame (pd.DataFrame): Numeric values. A 2-level index is read as
            `(panel, row)`; a flat index emits no panel titles.
        kinds (pd.Series | pd.DataFrame): Format kind per row, or per cell.
        stars (pd.DataFrame | None): Significance markers aligned with `frame`.
        n_col (int): Numeric columns in the *shared* tabular, which may exceed
            this block's own.

    Returns:
        list[str]: Rows only -- the caller owns `tabular` and the rules.
    """
    kind_of: pd.DataFrame = _kind_frame(kinds, frame)
    header: list[str] = [f'{{{_escape(str(c))}}}' for c in frame.columns]
    lines: list[str] = [' & '+_pad(header, n_col)+r' \\', '\\midrule']

    panelled: bool = isinstance(frame.index, pd.MultiIndex)
    seen: str|None = None
    for key, row in frame.iterrows():
        label: str = str(key[-1]) if panelled else str(key)
        if panelled and key[0] != seen:
            if seen is not None:
                lines.append('\\addlinespace')
            lines.append(f'\\multicolumn{{{n_col+1}}}{{l}}'
                         f'{{\\textit{{{_escape(str(key[0]))}}}}} \\\\')
            seen = key[0]
        marks: pd.Series = (stars.loc[key] if stars is not None
                            else pd.Series('', index=frame.columns))
        cells: list[str] = [_cell(row[c], kind_of.at[key, c], str(marks[c]))
                            for c in frame.columns]
        lines.append(f'{_escape(label)} & '+_pad(cells, n_col)+r' \\')
    return lines


def render_latex (blocks: list[tuple[pd.DataFrame, pd.Series,
                                     pd.DataFrame|None]],
                  caption: str, label: str, notes: list[str])-> str:
    """A complete `table` float: one `tabular`, however many blocks.

    Several blocks are how a table whose panels have *different columns* is
    written -- Table 3's asset panel has two and its test panels have three.

    They share one tabular, widened to the largest block and with the narrower
    ones flushed right. Giving each block its own tabular is the obvious
    alternative and it does not work: `\\centering` then centres each box at its
    own natural width, so the panels sit at different left edges with rules of
    different lengths and read as three tables that happen to be adjacent
    rather than as three panels of one. Sharing the tabular is what makes the
    column positions and the rules line up.
    """
    n_col: int = max(len(frame.columns) for frame, _, _ in blocks)
    lines: list[str] = [
        '% Requires: \\usepackage{booktabs} \\usepackage{threeparttable}',
        '\\begin{table}[htbp]', '\\centering', '\\footnotesize',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        '\\begin{threeparttable}',
        f'\\begin{{tabular}}{{l{"r"*n_col}}}', '\\toprule']
    for i, (frame, kinds, stars) in enumerate(blocks):
        if i:
            lines += ['\\addlinespace', '\\midrule']
        lines += _block_lines(frame, kinds, stars, n_col)
    lines += ['\\bottomrule', '\\end{tabular}']
    lines.append('\\begin{tablenotes}[flushleft]\\footnotesize')
    lines += [f'\\item {note}' for note in notes]
    lines += ['\\end{tablenotes}', '\\end{threeparttable}', '\\end{table}']
    return '\n'.join(lines)+'\n'


def _levels_block (summary: pd.DataFrame, order: tuple[str, ...],
                   labels: dict[str, str]
                   )-> tuple[pd.DataFrame, pd.Series, None]:
    """Panel A of Tables 1 and 2: what each strategy did, with no inference in it.

    Returns:
        tuple: the frame indexed `(panel, row)`, the format kind per row, and
            no significance markers.
    """
    columns: list[str] = [s for s in order if s in summary.columns]
    rows: list[tuple[str, str]] = [(PANEL_LEVELS, PERFORMANCE_ROWS[r])
                                   for r in PERFORMANCE_ROWS]
    frame: pd.DataFrame = summary.loc[list(PERFORMANCE_ROWS), columns]
    frame.index = pd.MultiIndex.from_tuples(rows)
    frame.columns = [labels[c] for c in columns]
    kinds: pd.Series = pd.Series({row: PERFORMANCE_KIND[row[1]] for row in rows})
    return frame, kinds.reindex(frame.index), None


def _difference_block (returns: pd.DataFrame,
                       pairs: list[tuple[str, str, str]], panel: str,
                       gamma: float = risk_aversion, n_boot: int = 4999,
                       seed: int = 0, alternative: str = ALTERNATIVE
                       )-> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """One panel of difference tests: three statistics for each pair.

    Args:
        returns (pd.DataFrame): `monthly_returns.csv` -- `<strategy>_net` and
            `rf`. Net returns throughout; the Sharpe test is handed excess
            returns and the other two total returns, which is what each of
            them requires.
        pairs (list[tuple[str, str, str]]): `(column, strategy, benchmark)`,
            one column of the panel each, reporting `strategy` minus
            `benchmark`.
        panel (str): The panel's title.
        gamma (float): Relative risk aversion for the CE test.
        n_boot (int): Bootstrap resamples per test.
        seed (int): Seeds every bootstrap.
        alternative (str): Passed to all three tests. Defaults to
            `ALTERNATIVE`.

    Returns:
        tuple: the numeric frame, indexed `(panel, statistic, row)` so that the
            three t-statistic rows stay distinct; the format kind per row; and
            the significance markers, on the difference rows only.
    """
    rf: pd.Series = returns['rf']
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    marks: dict[tuple[str, str, str], dict[str, str]] = {}
    kinds: dict[tuple[str, str, str], str] = {}
    for key, name, kind in DIFFERENCE_ROWS:
        head: tuple[str, str, str] = (panel, key, name)
        t_row: tuple[str, str, str] = (panel, key, T_ROW)
        p_row: tuple[str, str, str] = (panel, key, P_ROW)
        kinds.update({head: kind, t_row: 'num', p_row: 'pval'})
        for row in (head, t_row, p_row):
            values[row] = {}
            marks[row] = {}

        for column, strategy, benchmark in pairs:
            x: pd.Series = returns[f'{strategy}_net']
            b: pd.Series = returns[f'{benchmark}_net']
            if key == 'sharpe':
                out = sharpe_difference_test(x-rf, b-rf, n_boot=n_boot,
                                             seed=seed, alternative=alternative)
                diff, err = 'sr_diff', 'sr_diff_se'
            elif key == 'crra_ce':
                out = ce_difference_test(x, b, gamma=gamma, n_boot=n_boot,
                                         seed=seed, alternative=alternative)
                diff, err = 'ce_diff', 'ce_diff_se'
            else:
                out = mean_difference_test(x, b, n_boot=n_boot, seed=seed,
                                           alternative=alternative)
                diff, err = 'mean_diff', 'mean_diff_se'
            se: float = float(out[err])
            pval: float = float(out['pval_boot'])
            values[head][column] = float(out[diff])
            values[t_row][column] = (float(out[diff])/se if se > 0
                                     else float('nan'))
            values[p_row][column] = pval
            marks[head][column] = _stars(pval)

    columns: list[str] = [column for column, _, _ in pairs]
    frame: pd.DataFrame = pd.DataFrame(values).T.reindex(columns=columns)
    frame.index = pd.MultiIndex.from_tuples(frame.index)
    star_frame: pd.DataFrame = (pd.DataFrame(marks).T
                                .reindex(index=frame.index, columns=columns)
                                .fillna(''))
    return frame, pd.Series(kinds).reindex(frame.index), star_frame


def table_ex_post (summary: pd.DataFrame, returns: pd.DataFrame,
                   gamma: float = risk_aversion, n_boot: int = 4999,
                   seed: int = 0
                   )-> dict[str, tuple[pd.DataFrame, pd.Series,
                                       pd.DataFrame|None]]:
    """Table 1: B&H and the proposed model's ex post rows.

    Returns:
        dict: `'performance'` (Panel A) and `'tests'` (Panel B, each proposed
            row minus B&H), each a `(frame, kinds, stars)` triple ready for
            `render_latex`.
    """
    order: list[str] = [s for s in EX_POST_ORDER if s in summary.columns]
    pairs: list[tuple[str, str, str]] = [
        (EX_POST_LABEL[s], s, EX_POST_BENCHMARK)
        for s in order if s != EX_POST_BENCHMARK]
    return {'performance': _levels_block(summary, EX_POST_ORDER, EX_POST_LABEL),
            'tests': _difference_block(returns, pairs,
                                       'Panel B: Difference from B&H',
                                       gamma=gamma, n_boot=n_boot, seed=seed)}


def table_historical (summary: pd.DataFrame, returns: pd.DataFrame,
                      gamma: float = risk_aversion, n_boot: int = 4999,
                      seed: int = 0
                      )-> dict[str, tuple[pd.DataFrame, pd.Series,
                                          pd.DataFrame|None]]:
    """Table 2: equal weight, EPO, PPP and the proposed model's historical rows.

    Returns:
        dict: `'performance'` (Panel A), `'vs_equal_weight'` (Panel B, every
            row minus equal weight) and, when a matched row and its comparator
            are both there, `'vs_matched'` (Panel C, each matched row minus
            that comparator). Each a `(frame, kinds, stars)` triple.
    """
    order: list[str] = [s for s in HISTORICAL_ORDER if s in summary.columns]
    vs_ew: list[tuple[str, str, str]] = [
        (HISTORICAL_LABEL[s], s, HISTORICAL_BENCHMARK)
        for s in order if s != HISTORICAL_BENCHMARK]
    vs_matched: list[tuple[str, str, str]] = [
        (f'{HISTORICAL_LABEL[row]} $-$ {HISTORICAL_LABEL[comparator]}',
         row, comparator)
        for row, comparator in MATCHED
        if row in summary.columns and comparator in summary.columns]
    blocks: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame|None]] = {
        'performance': _levels_block(summary, HISTORICAL_ORDER,
                                     HISTORICAL_LABEL),
        'vs_equal_weight': _difference_block(
            returns, vs_ew, 'Panel B: Difference from equal weight',
            gamma=gamma, n_boot=n_boot, seed=seed)}
    if vs_matched:
        blocks['vs_matched'] = _difference_block(
            returns, vs_matched,
            'Panel C: Difference from the matched comparator',
            gamma=gamma, n_boot=n_boot, seed=seed)
    return blocks


def table_experiment (seed: int = 0, n_sim: int = 10000,
                      gamma: float = thought.GAMMA, n_boot: int = 4999,
                      distribution: str = 'gaussian'
                      )-> dict[str, tuple[pd.DataFrame, pd.Series,
                                          pd.DataFrame|None]]:
    """Table 3: the thought experiment, as three panels of different shapes.

    Panel A has two columns (the assets) and Panels B and C have three (the
    tests), which is why this returns a dict of blocks rather than one frame.

    Returns:
        dict: `'assets'`, `'tests'` and `'montecarlo'`, each a
            `(frame, kinds, stars)` triple ready for `render_latex`.
    """
    table, tests, _ = thought.run_single(seed=seed, gamma=gamma,
                                         n_boot=n_boot,
                                         distribution=distribution)
    asset_rows: dict[str, tuple[str, str]] = {
        # The multiplication sign lives in the label, not the cell: a trailing
        # `$\times$` would push the number left of the column's other entries.
        'terminal_wealth': ('Terminal wealth ($\\times$)', 'num'),
        'ann_return_geom': ('Annualised return, geometric (%)', 'pct'),
        'ann_vol': ('Annualised volatility (%)', 'pct'),
        'max_drawdown': ('Maximum drawdown (%)', 'pct'),
        'worst_month': ('Worst month (%)', 'pct'),
        'sharpe': ('Sharpe ratio', 'num'),
        'ann_crra_ce': ('CRRA certainty equivalent (%)', 'pct')}
    assets: pd.DataFrame = table.loc[list(asset_rows)]
    assets.index = pd.MultiIndex.from_tuples(
        [(PANEL_ASSETS, asset_rows[r][0]) for r in assets.index])
    assets.columns = ['Asset A', 'Asset B']
    asset_kind: pd.Series = pd.Series(
        {(PANEL_ASSETS, asset_rows[r][0]): asset_rows[r][1]
         for r in asset_rows})

    test_label: dict[str, str] = {'mean_return': 'Mean return',
                                  'sharpe': 'Sharpe ratio',
                                  'crra_ce': 'CRRA CE'}
    order: list[str] = [k for k, _, _ in TEST_PANEL]
    single: pd.DataFrame = pd.DataFrame({
        DIFFERENCE: tests.loc['diff', order],
        T_STAT: tests.loc['t_stat', order],
        P_VALUE: tests.loc['pval_boot', order]}).T
    single.columns = [test_label[c] for c in single.columns]
    single.index = pd.MultiIndex.from_tuples(
        [(PANEL_TESTS, r) for r in single.index])
    single_star: pd.DataFrame = pd.DataFrame(
        '', index=single.index, columns=single.columns)
    single_star.loc[(PANEL_TESTS, DIFFERENCE)] = [
        _stars(tests.at['pval_boot', k]) for k in order]
    # The `Difference` row is the one place a single row carries three
    # different units: percentage points, ratio, percentage points. Every other
    # row here is uniform, so only this one needs the per-cell form.
    single_kind: pd.DataFrame = pd.DataFrame(
        {label: {(PANEL_TESTS, DIFFERENCE): kind,
                 (PANEL_TESTS, T_STAT): 'num',
                 (PANEL_TESTS, P_VALUE): 'pval'}
         for (key, _, kind), label in zip(TEST_PANEL, single.columns)})

    draws: pd.DataFrame = thought.run_monte_carlo(
        n_sim=n_sim, gamma=gamma, distribution=distribution)
    stats: pd.DataFrame = thought.summarise_monte_carlo(draws)
    mc_rows: dict[str, tuple[str, str]] = {
        'mean': ('Mean', 'num'), 'std': ('Standard deviation', 'num'),
        'p5': ('5th percentile', 'num'), 'median': ('Median', 'num'),
        'p95': ('95th percentile', 'num'),
        'p99': ('99th percentile', 'num'),
        # Bare `%` here -- `_escape` adds the backslash, and pre-escaping would
        # emit `\\%`, which is a line break followed by a comment.
        'share_t_above_0': ('Share with $t>0$ (%)', 'share'),
        'reject_at_5%': ('Reject at the 5% level (%)', 'share'),
        'reject_for_A_at_5%': ('\\quad in favour of A (%)', 'share'),
        'reject_for_B_at_5%': ('\\quad in favour of B (%)', 'share')}
    monte: pd.DataFrame = stats.loc[list(mc_rows), order]
    monte.index = pd.MultiIndex.from_tuples(
        [(PANEL_MONTE, mc_rows[r][0]) for r in monte.index])
    monte.columns = [test_label[c] for c in monte.columns]
    monte_kind: pd.Series = pd.Series(
        {(PANEL_MONTE, mc_rows[r][0]): mc_rows[r][1] for r in mc_rows})

    return {'assets': (assets, asset_kind, None),
            'tests': (single, single_kind, single_star),
            'montecarlo': (monte, monte_kind, None)}


def _period (returns: pd.DataFrame)-> str:
    """`2015-01 to 2025-12`, for the table notes."""
    idx: pd.DatetimeIndex = pd.to_datetime(returns.index)
    return f'{idx.min():%Y-%m} to {idx.max():%Y-%m}'


def build_all (raw_dir: Path = RAW_BACKTEST_DIR,
               processed_dir: Path = PROCESSED_BACKTEST_DIR,
               latex_dir: Path = LATEX_DIR, n_sim: int = 10000,
               n_boot: int = 4999, gamma: float = risk_aversion,
               seed: int = 0, skip_experiment: bool = False)-> list[Path]:
    """Build all three tables and write both forms of each.

    Args:
        raw_dir (Path): A directory `run.save` has written.
        processed_dir (Path): Where the CSV form goes.
        latex_dir (Path): Where the `.tex` form goes.
        n_sim (int): Monte Carlo realisations for Table 3's Panel C.
        n_boot (int): Bootstrap resamples for every test.
        gamma (float): Relative risk aversion for the CE test.
        seed (int): Seeds every bootstrap and Table 3's shown realisation.
        skip_experiment (bool): Build Tables 1 and 2 only. Table 3 is the slow
            one -- `n_sim` realisations of two HAC estimates each.

    Returns:
        list[Path]: Every file written, in order.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write (frame: pd.DataFrame, name: str)-> None:
        path: Path = processed_dir/f'{name}.csv'
        frame.to_csv(path)
        written.append(path)

    def _write_tex (text: str, name: str)-> None:
        path: Path = latex_dir/f'{name}.tex'
        path.write_text(text, encoding='utf-8')
        written.append(path)

    summary: pd.DataFrame = pd.read_csv(raw_dir/'summary.csv', index_col=0)
    returns: pd.DataFrame = pd.read_csv(raw_dir/'monthly_returns.csv',
                                        index_col=0, parse_dates=True)
    period: str = _period(returns)
    n_month: int = len(returns)
    common: str = (f'Sample {period} ({n_month} monthly observations). '
                   f'Returns are net of {backtest_cost_bps:.0f}~bps of '
                   f'one-way transaction cost.')
    tests_note: str = (
        'All tests are one-sided and paired, $H_1$: the row exceeds its '
        'benchmark. Each is a delta method over sample moments with a '
        'prewhitened quadratic-spectral HAC long-run covariance (Andrews, '
        '1991; Andrews and Monahan, 1992); the reported $p$-value is from a '
        'studentized circular block bootstrap with block length 5 and '
        f'{n_boot} resamples (Ledoit and Wolf, 2008).')
    units_note: str = (
        'Mean return and certainty equivalent differences are annualised, in '
        'percentage points; the Sharpe ratio difference is annualised, in '
        'ratio units. The certainty equivalent is the annualised certain '
        f'return an investor with CRRA utility at $\\gamma={gamma:g}$ would '
        'accept in place of the realised series. Turnover is the annualised '
        'sum of absolute weight changes. Effective N is $1/\\sum_i w_i^2$ '
        'averaged over months: the number of equally weighted holdings that '
        'would be as concentrated as the book.')
    stars_note: str = ('$^{***}$, $^{**}$ and $^{*}$ denote significance at '
                       'the 1\\%, 5\\% and 10\\% levels.')

    if (EX_POST_BENCHMARK in summary.columns
            and any(s in summary.columns for s in EX_POST_ORDER[1:])):
        ex_post = table_ex_post(summary, returns, gamma=gamma, n_boot=n_boot,
                                seed=seed)
        for name, (frame, _, _) in ex_post.items():
            _write(frame, f'table1_ex_post_{name}')
        n_test: int = 3*len(ex_post['tests'][0].columns)
        _write_tex(render_latex(
            [ex_post['performance'], ex_post['tests']],
            caption='Ex post: Bielstein and Hanauer against the proposed '
                    'model.',
            label='tab:ex_post',
            notes=[
                common+' Every row is ex post: it is formed with accounting '
                'data realised after the formation date, so it is a benchmark '
                'under perfect foresight and not an implementable strategy.',
                'B\\&H is the portfolio of Bielstein and Hanauer (2019): '
                'maximum Sharpe ratio on the implied cost of capital of '
                'Gebhardt, Lee and Swaminathan (2001) plus rescaled momentum, '
                'with a Ledoit--Wolf covariance and a 5\\% cap per name. Its '
                'explicit earnings years are the realised ones over the same '
                "future window the proposed model's moments are drawn from.",
                'Every column after B\\&H is the proposed model, solved on one '
                'set of simulated implied returns. Max Sharpe uses only their '
                'mean and covariance, through B\\&H\'s own rule; CRRA uses the '
                "whole distribution. The B\\&H $N$ columns are held, at each "
                "rebalance date, to at least B\\&H's effective number of names "
                'at that date; the unconstrained columns are not, and CRRA '
                'unconstrained is the model as specified.',
                'Panel B reports each proposed row minus B\\&H. '+tests_note,
                units_note,
                f'Panel B carries {n_test} tests; the notes on multiple '
                'testing in the text apply.',
                stars_note]),
            'table1_ex_post')
    else:
        warnings.warn('Table 1 skipped: summary.csv has no icc_mvo_ex_post '
                      'column, or no proposed ex post row to set against it. '
                      'Run them with `run --only`, then `run --rebuild`.',
                      RuntimeWarning)

    if HISTORICAL_BENCHMARK in summary.columns:
        historical = table_historical(summary, returns, gamma=gamma,
                                      n_boot=n_boot, seed=seed)
        for name, (frame, _, _) in historical.items():
            _write(frame, f'table2_historical_{name}')
        n_test = 3*sum(len(block[0].columns)
                       for name, block in historical.items()
                       if name != 'performance')
        _write_tex(render_latex(
            list(historical.values()),
            caption='Historical: the proposed model against equal weight, EPO '
                    'and PPP.',
            label='tab:historical',
            notes=[
                common+' The proposed model elicits its moments from the '
                'training window only, so every row is implementable.',
                'The EPO $N$ and PPP $N$ rows hold the proposed model, at each '
                "rebalance date, to that comparator's effective number of "
                'names at that date; the unconstrained row is the model as '
                'specified.',
                'Panel B reports each strategy minus equal weight; Panel C '
                'each matched row minus the comparator whose breadth it '
                'holds. '+tests_note,
                units_note,
                f'Panels B and C carry {n_test} tests; the notes on multiple '
                'testing in the text apply.',
                stars_note]),
            'table2_historical')
    else:
        warnings.warn('Table 2 skipped: summary.csv has no equal_weight '
                      'column to test against.', RuntimeWarning)

    if not skip_experiment:
        blocks: dict = table_experiment(seed=seed, n_sim=n_sim, gamma=gamma,
                                        n_boot=n_boot)
        for name, (frame, _, _) in blocks.items():
            _write(frame, f'table3_experiment_{name}')
        _write_tex(render_latex(
            [blocks['assets'], blocks['tests'], blocks['montecarlo']],
            caption='A thought experiment: terminal wealth against '
                    'risk-adjusted performance.',
            label='tab:experiment',
            notes=[
                'Two synthetic assets over 132 months. Asset A is Gaussian '
                'with a 10\\% annual mean and 1\\% annual volatility. Asset B '
                'is Gaussian with a 40\\% annual mean and 40\\% annual '
                'volatility for 120 months; its final 12 months are '
                'deterministic and set so that terminal wealth is exactly '
                '$40\\times$ in every realisation.',
                f'Panels A and B show one realisation (seed {seed}); Panel C '
                f'establishes that it is typical over {n_sim:,} of them. All '
                'differences are A minus B, so a positive value favours the '
                'steadier asset.',
                'In Panel B the mean return and certainty equivalent '
                'differences are in percentage points and the Sharpe ratio '
                'difference is in ratio units.',
                'Panel C is the point of the table. On identical data the '
                'Sharpe ratio test favours A in almost every realisation, the '
                'mean return test favours B in almost every realisation, and '
                'the certainty equivalent test separates them in fewer than '
                'one realisation in three hundred. The three statistics do '
                'not merely disagree about significance: two of them reject '
                'in opposite directions and the third cannot tell the assets '
                'apart. Which one a study reports is therefore not a '
                'presentational choice.',
                'One caveat on interpretation and one on inference. Asset B '
                'is Gaussian in simple returns, for which '
                '$E[(1+R)^{1-\\gamma}]$ formally diverges at $R=-1$, so its '
                'population certainty equivalent does not exist. At this '
                'volatility a monthly return of $-100\\%$ is roughly nine '
                'standard deviations away and is never drawn, so the sample '
                'estimate is well behaved and the column can be read as '
                'printed; at higher volatilities it cannot, and '
                '`distribution=\'lognormal\'` is the specification to use '
                'then. Second, the final 12 observations are constant by '
                'construction, which violates the stationarity the HAC '
                'estimator and the block bootstrap assume; the effect sizes '
                'rather than the $p$-values are what this table is evidence '
                'for.',
                '$^{***}$, $^{**}$ and $^{*}$ denote significance at the 1\\%, '
                '5\\% and 10\\% levels.']),
            'table3_experiment')
    return written


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw', type=Path, default=RAW_BACKTEST_DIR)
    parser.add_argument('--processed', type=Path,
                        default=PROCESSED_BACKTEST_DIR)
    parser.add_argument('--latex', type=Path, default=LATEX_DIR)
    parser.add_argument('--n-sim', type=int, default=10000,
                        help="realisations behind Table 3's Panel C")
    parser.add_argument('--n-boot', type=int, default=4999)
    parser.add_argument('--gamma', type=float, default=risk_aversion)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-experiment', action='store_true',
                        help='build Tables 1 and 2 only')
    return parser.parse_args()


if __name__ == '__main__':
    args = _cli()
    paths = build_all(raw_dir=args.raw, processed_dir=args.processed,
                      latex_dir=args.latex, n_sim=args.n_sim,
                      n_boot=args.n_boot, gamma=args.gamma, seed=args.seed,
                      skip_experiment=args.skip_experiment)
    for written in paths:
        try:
            shown = written.relative_to(Path.cwd())
        except ValueError:
            shown = written
        print(f'  {shown}')
    print(f'\n{len(paths)} files written')
