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

* **Table 1** -- ex post. Bielstein and Hanauer's portfolio under maximum
  Sharpe or quadratic (mean-variance) utility on the same inputs, against the
  proposed model under three objectives on its simulated moments or whole
  distribution -- maximum Sharpe, quadratic utility, or CRRA -- every proposed
  column held to B&H's effective N. No unconstrained column: breadth is fixed
  throughout, so every comparison isolates the objective alone. Panel A the
  levels; Panels B and C each proposed row minus B&H, twice over -- against
  B&H's max-Sharpe column, then against its quadratic-utility column.
* **Table 2** -- historical, i.e. implementable. Equal weight, EPO, PPP, and the
  proposed model at EPO's breadth, at PPP's, and unconstrained. Panel A the
  levels, Panel B every row minus equal weight, Panel C each matched row minus
  the comparator whose breadth it holds.
* **Table 3** -- the thought experiment, two panels. Panel A
  (`experiment_thought_panel_a`): two synthetic assets calibrated to Table 1's
  `Max Sharpe, B&H N` and `CRRA, B&H N` columns, each given realistic path
  volatility, and how often each has the larger mean return, Sharpe ratio and
  CRRA CE -- a share, not a test. Panel B (`experiment_thought`): the original
  pair and the Monte Carlo distribution of the disagreement between the three
  tests, the one-realisation illustration having been dropped as clutter.

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
from .inference.sharpe_inference import sharpe_difference_test
from .inference.ce_inference import ce_difference_test
from .inference.mean_inference import mean_difference_test
from .experiment import experiment_thought as thought
from .experiment import experiment_thought_panel_a as thought_a

# Table 1, ex post: B&H's two objectives, then the proposed model's three, all
# at B&H's effective N -- no unconstrained column. Every proposed column's
# breadth is the same, which the notes say once rather than each header.
EX_POST_ORDER: tuple[str, ...] = ('icc_mvo_ex_post', 'icc_mvo_ex_post_qu',
                                  'proposed_ex_post_msr_icc_n',
                                  'proposed_ex_post_qu_icc_n',
                                  'proposed_ex_post_icc_n')
EX_POST_LABEL: dict[str, str] = {
    'icc_mvo_ex_post': 'B&H, Max Sharpe',
    'icc_mvo_ex_post_qu': 'B&H, Quadratic Utility',
    'proposed_ex_post_msr_icc_n': 'Max Sharpe',
    'proposed_ex_post_qu_icc_n': 'Quadratic Utility',
    'proposed_ex_post_icc_n': 'CRRA'}
# Each proposed column is tested against both of B&H's columns, one panel
# per benchmark: (benchmark strategy, this table's key for the block, panel
# title). Order matters -- it is the order the panels render in.
EX_POST_BENCHMARKS: tuple[tuple[str, str, str], ...] = (
    ('icc_mvo_ex_post', 'tests_vs_max_sharpe',
     'Panel B: Difference from B&H, Max Sharpe'),
    ('icc_mvo_ex_post_qu', 'tests_vs_qu',
     'Panel C: Difference from B&H, Quadratic Utility'))

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

# Table 3's three statistics, in the same order as `DIFFERENCE_ROWS`; `order`
# is what both of Table 3's panels read from this. The labels feed
# `TEST_LABEL`, the column headers both panels share.
TEST_PANEL: tuple[tuple[str, str, str], ...] = (
    ('mean_return', 'Mean return', 'pct'),
    ('sharpe', 'Sharpe ratio', 'num'),
    ('crra_ce', 'CRRA CE', 'pct'))
TEST_LABEL: dict[str, str] = {key: label for key, label, _ in TEST_PANEL}

# Table 3, Panel A: `experiment_thought_panel_a.win_shares`' two rows, kept in
# this order so Asset 1 (the fixed, zero-terminal-risk target) reads first.
PANEL_WIN: str = 'Panel A: Share of realisations with the larger value'
WIN_ROWS: dict[str, str] = {'asset_1_higher': 'Asset 1 higher (%)',
                            'asset_2_higher': 'Asset 2 higher (%)'}

# Table 3, Panel B: the Monte Carlo distribution is the point of this panel
# (see `experiment_thought`'s docstring), and the one-realisation illustration
# it used to sit beside was dropped as clutter.
PANEL_MONTE: str = 'Panel B: Distribution of $t$-statistics across realisations'

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


def _header (label: str)-> str:
    """A column header, stacked onto two lines at its first comma.

    A one-line header makes its column as wide as the whole label, and at five
    or six labels like 'Max Sharpe, unconstrained' the tabular runs off the
    page. Stacking halves that width. A nested `tabular` does the stacking so no
    package beyond the two the module promises is needed.
    """
    text: str = _escape(label)
    if ', ' not in text:
        return f'{{{text}}}'
    top, bottom = text.split(', ', 1)
    return f'\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{top}\\\\{bottom}\\end{{tabular}}'


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
    header: list[str] = [_header(str(c)) for c in frame.columns]
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

    Every proposed column is tested against both of B&H's columns
    (`EX_POST_BENCHMARKS`), each in its own panel -- the proposed-strategy
    list for both panels is read off the same `order` this builds Panel A
    from, less whichever of `EX_POST_ORDER` is itself a benchmark, so the two
    panels cannot drift out of sync with Panel A or with each other.

    Returns:
        dict: `'performance'` (Panel A), plus one entry per benchmark in
            `EX_POST_BENCHMARKS` that is present in `summary.columns` --
            keyed by that tuple's own key (today, `'tests_vs_max_sharpe'` and
            `'tests_vs_qu'`), each proposed row minus that benchmark. Each
            value a `(frame, kinds, stars)` triple ready for `render_latex`.
    """
    order: list[str] = [s for s in EX_POST_ORDER if s in summary.columns]
    benchmarks: set[str] = {b for b, _, _ in EX_POST_BENCHMARKS}
    proposed: list[str] = [s for s in order if s not in benchmarks]
    blocks: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame|None]] = {
        'performance': _levels_block(summary, EX_POST_ORDER, EX_POST_LABEL)}
    for benchmark, key, title in EX_POST_BENCHMARKS:
        if benchmark not in summary.columns:
            continue
        pairs: list[tuple[str, str, str]] = [
            (EX_POST_LABEL[s], s, benchmark) for s in proposed]
        blocks[key] = _difference_block(returns, pairs, title, gamma=gamma,
                                        n_boot=n_boot, seed=seed)
    return blocks


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


def table_experiment_panel_a (n_sim: int = 10000
                              )-> tuple[pd.DataFrame, pd.Series, None]:
    """Table 3, Panel A: how often each synthetic asset has the larger value.

    Two assets calibrated to Table 1's `Max Sharpe, B&H N` and `CRRA, B&H N`
    columns -- `experiment_thought_panel_a`'s docstring has the construction
    and the calibration -- each with realistic path volatility, unlike Panel
    B's Asset A. Deliberately not a hypothesis test: the paper does not yet
    test Table 1's Max-Sharpe column against its CRRA column either, so this
    only counts, out of `n_sim` realisations, which asset has the larger
    value of each statistic.

    Returns:
        tuple: the frame indexed `(PANEL_WIN, row)`, the format kind per row
            (`'share'` throughout), and no significance markers -- there is
            no test here to star.
    """
    order: list[str] = [k for k, _, _ in TEST_PANEL]
    draws: pd.DataFrame = thought_a.run_monte_carlo(n_sim=n_sim)
    shares: pd.DataFrame = thought_a.win_shares(draws, metrics=tuple(order))
    frame: pd.DataFrame = shares.loc[list(WIN_ROWS), order]
    frame.index = pd.MultiIndex.from_tuples(
        [(PANEL_WIN, WIN_ROWS[r]) for r in frame.index])
    frame.columns = [TEST_LABEL[c] for c in order]
    kind: pd.Series = pd.Series(
        {(PANEL_WIN, WIN_ROWS[r]): 'share' for r in WIN_ROWS})
    return frame, kind.reindex(frame.index), None


def table_experiment (n_sim: int = 10000, gamma: float = thought.GAMMA,
                      distribution: str = 'gaussian'
                      )-> dict[str, tuple[pd.DataFrame, pd.Series,
                                          pd.DataFrame|None]]:
    """Table 3: the thought experiment, two panels.

    Panel A (`table_experiment_panel_a`) calibrates two synthetic assets to
    Table 1's own Max-Sharpe-versus-CRRA gap and counts win shares. Panel B is
    `experiment_thought`'s Monte Carlo distribution the thought experiment
    turns on.

    Panel B's one-realisation illustration -- the two assets' own numbers, and
    the single difference test built from them -- used to sit in front of it
    as two more panels, and took a `seed` and an `n_boot` to build. Dropped:
    the Monte Carlo distribution is the point of the panel (see
    `experiment_thought`'s docstring), the pair's construction is stated in
    the table notes, and `experiment_thought`'s own CLI still prints one
    realisation for anyone who wants to see it directly. `run_monte_carlo`
    takes no seed of its own -- each of its `n_sim` realisations seeds
    itself off its own index -- and defaults its per-realisation bootstrap
    off, one HAC $t$-statistic being the point of each draw.

    Returns:
        dict: `'panel_a'` and `'montecarlo'`, each a `(frame, kinds, stars)`
            triple ready for `render_latex`.
    """
    order: list[str] = [k for k, _, _ in TEST_PANEL]

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
    monte.columns = [TEST_LABEL[c] for c in monte.columns]
    monte_kind: pd.Series = pd.Series(
        {(PANEL_MONTE, mc_rows[r][0]): mc_rows[r][1] for r in mc_rows})

    return {'panel_a': table_experiment_panel_a(n_sim=n_sim),
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
        n_sim (int): Monte Carlo realisations behind Table 3.
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
    # Notes carry three things only: what the table shows, what its columns
    # are, and the discretionary choices behind the numbers. Interpretation
    # belongs in the text.
    common: str = f'Sample {period} ({n_month} monthly observations).'
    inference: str = (
        'HAC standard errors (Andrews and Monahan, 1992); $p$-values from a '
        f'studentized circular block bootstrap, block length 5, {n_boot} '
        'resamples (Ledoit and Wolf, 2008).')
    choices: str = (
        f'Returns are net of {backtest_cost_bps:.0f}~bps one-way transaction '
        f'cost; CRRA utility uses $\\gamma={gamma:g}$. Tests are one-sided and '
        'paired, $H_1$: the strategy beats its benchmark. '+inference)
    # Table 1 only: Table 2 has no quadratic-utility column, so `choices`
    # above stays as it is for Table 2's own notes.
    choices_ex_post: str = (
        f'Returns are net of {backtest_cost_bps:.0f}~bps one-way transaction '
        f'cost; CRRA utility and quadratic utility both use '
        f'$\\gamma={gamma:g}$. Tests are one-sided and paired, $H_1$: the '
        'strategy beats its benchmark. '+inference)
    stars_note: str = ('$^{***}$, $^{**}$ and $^{*}$ denote significance at '
                       'the 1\\%, 5\\% and 10\\% levels.')

    ex_post_benchmark_cols: set[str] = {b for b, _, _ in EX_POST_BENCHMARKS}
    ex_post_proposed_cols: list[str] = [s for s in EX_POST_ORDER
                                        if s not in ex_post_benchmark_cols]
    if (any(b in summary.columns for b in ex_post_benchmark_cols)
            and any(s in summary.columns for s in ex_post_proposed_cols)):
        ex_post = table_ex_post(summary, returns, gamma=gamma, n_boot=n_boot,
                                seed=seed)
        for name, (frame, _, _) in ex_post.items():
            _write(frame, f'table1_ex_post_{name}')
        _write_tex(render_latex(
            list(ex_post.values()),
            caption='Ex post: Bielstein and Hanauer against the proposed '
                    'model.',
            label='tab:ex_post',
            notes=[
                common+' Bielstein and Hanauer (B\\&H) against the proposed '
                'model, both formed with accounting data realised after the '
                'formation date: a perfect-foresight benchmark, not an '
                'implementable strategy. Panel B is each proposed column '
                'minus B\\&H, Max Sharpe; Panel C is each proposed column '
                'minus B\\&H, Quadratic Utility.',
                'B\\&H: the implied cost of capital of Gebhardt, Lee and '
                'Swaminathan (2001) plus momentum, with a Ledoit--Wolf '
                'covariance and a 5\\% cap per name, under either a '
                'maximum-Sharpe-ratio or a maximum-quadratic-(mean-variance)'
                '-utility objective. The proposed model\'s three columns -- '
                'Max Sharpe, Quadratic Utility and CRRA, on its own '
                'simulated mean and covariance (the first two) or whole '
                'simulated distribution (CRRA) -- are each held to at least '
                'B\\&H\'s effective number of names, $N=1/\\sum_i w_i^2$, '
                'at each rebalance.',
                choices_ex_post,
                stars_note]),
            'table1_ex_post')
    else:
        warnings.warn('Table 1 skipped: summary.csv has no icc_mvo_ex_post '
                      'or icc_mvo_ex_post_qu column, or no proposed ex post '
                      'row to set against either. Run them with `run '
                      '--only`, then `run --rebuild`.', RuntimeWarning)

    if HISTORICAL_BENCHMARK in summary.columns:
        historical = table_historical(summary, returns, gamma=gamma,
                                      n_boot=n_boot, seed=seed)
        for name, (frame, _, _) in historical.items():
            _write(frame, f'table2_historical_{name}')
        _write_tex(render_latex(
            list(historical.values()),
            caption='Historical: the proposed model against equal weight, EPO '
                    'and PPP.',
            label='tab:historical',
            notes=[
                common+' The proposed model against equal weight, EPO and '
                'PPP, all using the training window only, so every column is '
                'implementable. Panel B is each strategy minus equal weight; '
                'Panel C each matched column minus its comparator.',
                'Proposed, EPO $N$ and PPP $N$: the proposed model held to at '
                "least that comparator's effective number of names, "
                '$N=1/\\sum_i w_i^2$, at each rebalance. Proposed, '
                'unconstrained: no such floor.',
                choices,
                stars_note]),
            'table2_historical')
    else:
        warnings.warn('Table 2 skipped: summary.csv has no equal_weight '
                      'column to test against.', RuntimeWarning)

    if not skip_experiment:
        blocks: dict = table_experiment(n_sim=n_sim, gamma=gamma)
        for name, (frame, _, _) in blocks.items():
            _write(frame, f'table3_experiment_{name}')
        _write_tex(render_latex(
            [blocks['panel_a'], blocks['montecarlo']],
            caption='A thought experiment: three tests, one dataset, three '
                    'answers.',
            label='tab:experiment',
            notes=[
                'Panel A: two synthetic assets calibrated to Table 1\'s '
                '`Max Sharpe, B\\&H $N$\' and `CRRA, B\\&H $N$\' columns. '
                'Asset 1\'s terminal compound return is a fixed 15\\% a '
                'year, every realisation; its first ten years draw at 15\\% '
                'mean, 16.6\\% volatility, and the last year is set so '
                'terminal wealth hits that target exactly. Asset 2\'s '
                'target is itself drawn once per realisation, from a 25\\% '
                'mean, 2\\% volatility Gaussian, so it keeps a little '
                'terminal-wealth risk; its first ten years draw at 25\\% '
                'mean, 19\\% volatility, built the same way. Both carry '
                'realistic path risk, unlike Panel B\'s Asset A. No test is '
                f'run: the share is a count over {n_sim:,} realisations of '
                'which asset has the larger value.',
                'Panel B: two synthetic assets over 132 months, A minus B: A '
                'is Gaussian with a 10\\% annual mean and 1\\% volatility, a '
                'steady compounder; B is Gaussian with a 40\\% mean and '
                '40\\% volatility for 120 months, then set so that terminal '
                'wealth is exactly $40\\times$ in every realisation. A '
                'positive value favours the steadier asset. The '
                f'distribution of $t$-statistics is over {n_sim:,} '
                'realisations of the pair; "reject" is against the null of '
                'no difference at the 5\\% level.',
                f'CRRA utility uses $\\gamma={gamma:g}$ throughout. Panel '
                'B\'s $t$-statistic is diff / HAC standard error (Andrews '
                'and Monahan, 1992); no bootstrap runs per realisation.']),
            'table3_experiment')
    return written


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw', type=Path, default=RAW_BACKTEST_DIR)
    parser.add_argument('--processed', type=Path,
                        default=PROCESSED_BACKTEST_DIR)
    parser.add_argument('--latex', type=Path, default=LATEX_DIR)
    parser.add_argument('--n-sim', type=int, default=10000,
                        help="realisations behind Table 3's Monte Carlo")
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
