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

* **Table 1** -- performance, no inference. What each strategy did.
* **Table 2** -- the three difference tests against equal weight, ordered mean
  return, then Sharpe, then certainty equivalent. That order is deliberate: it
  runs from the statistic that ignores risk entirely to the one that penalises
  it hardest, and reading down the panels is reading the significance decay
  that the volatility gap produces. The accompanying text has to say so.
* **Table 3** -- the thought experiment of `experiment_thought`, which is what
  makes Table 2's pattern interpretable rather than merely awkward.

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
from pathlib import Path
import pandas as pd

from constant import (RAW_BACKTEST_DIR, PROCESSED_BACKTEST_DIR, LATEX_DIR,
                      risk_aversion, backtest_cost_bps)
from .sharpe_inference import sharpe_difference_test
from .ce_inference import ce_difference_test
from .mean_inference import mean_difference_test
from . import experiment_thought as thought

BENCHMARK: str = 'equal_weight'
# Benchmark first, then the two models from the literature, then ours -- the bar,
# what clears it, and what this paper proposes.
STRATEGY_ORDER: tuple[str, ...] = ('equal_weight', 'epo', 'ppp',
                                   'proposed_historical', 'proposed_ex_post')
STRATEGY_LABEL: dict[str, str] = {
    'equal_weight': 'Equal-weight', 'epo': 'EPO', 'ppp': 'PPP',
    'proposed_historical': 'Proposed (hist.)',
    'proposed_ex_post': 'Proposed (ex post)'}

# Everything from `metrics.SUMMARY_ROWS` except `sharpe_pval`, which moves to
# Table 2 so that all the inference lives in one place.
PERFORMANCE_ROWS: dict[str, str] = {
    'ann_return': 'Annualised return (%)',
    'ann_vol': 'Annualised volatility (%)',
    'max_drawdown': 'Maximum drawdown (%)',
    'sharpe': 'Sharpe ratio',
    'ann_crra_ce': 'CRRA certainty equivalent (%)',
    'ann_turnover': 'Annualised turnover',
    'avg_weight_entropy': 'Average weight entropy'}
PERFORMANCE_KIND: dict[str, str] = {
    'Annualised return (%)': 'pct', 'Annualised volatility (%)': 'pct',
    'Maximum drawdown (%)': 'pct', 'Sharpe ratio': 'num',
    'CRRA certainty equivalent (%)': 'pct', 'Annualised turnover': 'num',
    'Average weight entropy': 'num'}

# Mean first, certainty equivalent last. See the module docstring.
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


def table_performance (summary: pd.DataFrame)-> tuple[pd.DataFrame, pd.Series]:
    """Table 1: what each strategy did, with no inference in it."""
    columns: list[str] = [s for s in STRATEGY_ORDER if s in summary.columns]
    frame: pd.DataFrame = summary.loc[list(PERFORMANCE_ROWS), columns]
    frame.index = [PERFORMANCE_ROWS[r] for r in frame.index]
    frame.columns = [STRATEGY_LABEL[c] for c in columns]
    return frame, pd.Series(PERFORMANCE_KIND).reindex(frame.index)


def table_inference (returns: pd.DataFrame, benchmark: str = BENCHMARK,
                     gamma: float = risk_aversion, n_boot: int = 4999,
                     seed: int = 0)-> tuple[pd.DataFrame, pd.Series,
                                            pd.DataFrame]:
    """Table 2: all three difference tests against the benchmark.

    Args:
        returns (pd.DataFrame): `monthly_returns.csv` -- `<strategy>_gross`,
            `<strategy>_net` and `rf`. Net returns are used throughout, and the
            Sharpe test is handed excess returns while the other two take total
            returns, which is what each of them requires.
        benchmark (str): Strategy every difference is taken against.
        gamma (float): Relative risk aversion for the CE test.
        n_boot (int): Bootstrap resamples per test.
        seed (int): Seeds every bootstrap.

    Returns:
        tuple: the numeric frame indexed by `(panel, row)`, the format kind per
            row, and the significance markers.
    """
    rf: pd.Series = returns['rf']
    bench: pd.Series = returns[f'{benchmark}_net']
    names: list[str] = [s for s in STRATEGY_ORDER
                        if s != benchmark and f'{s}_net' in returns.columns]

    values: dict[tuple[str, str], dict[str, float]] = {}
    marks: dict[tuple[str, str], dict[str, str]] = {}
    kinds: dict[tuple[str, str], str] = {}
    for key, panel, kind in TEST_PANEL:
        kinds[(panel, DIFFERENCE)] = kind
        kinds[(panel, T_STAT)] = 'num'
        kinds[(panel, P_VALUE)] = 'pval'
        for row in (DIFFERENCE, T_STAT, P_VALUE):
            values.setdefault((panel, row), {})
            marks.setdefault((panel, row), {})

        for name in names:
            strategy: pd.Series = returns[f'{name}_net']
            if key == 'sharpe':
                out = sharpe_difference_test(strategy-rf, bench-rf,
                                             n_boot=n_boot, seed=seed)
                diff, err = 'sr_diff', 'sr_diff_se'
            elif key == 'crra_ce':
                out = ce_difference_test(strategy, bench, gamma=gamma,
                                         n_boot=n_boot, seed=seed)
                diff, err = 'ce_diff', 'ce_diff_se'
            else:
                out = mean_difference_test(strategy, bench, n_boot=n_boot,
                                           seed=seed)
                diff, err = 'mean_diff', 'mean_diff_se'
            label: str = STRATEGY_LABEL[name]
            se: float = float(out[err])
            pval: float = float(out['pval_boot'])
            values[(panel, DIFFERENCE)][label] = float(out[diff])
            values[(panel, T_STAT)][label] = (float(out[diff])/se if se > 0
                                              else float('nan'))
            values[(panel, P_VALUE)][label] = pval
            marks[(panel, DIFFERENCE)][label] = _stars(pval)

    frame: pd.DataFrame = pd.DataFrame(values).T
    frame.index = pd.MultiIndex.from_tuples(frame.index)
    star_frame: pd.DataFrame = pd.DataFrame(marks).T.reindex(
        frame.index).fillna('')
    return frame, pd.Series(kinds).reindex(frame.index), star_frame


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

    perf, perf_kind = table_performance(summary)
    _write(perf, 'table1_performance')
    _write_tex(render_latex(
        [(perf, perf_kind, None)],
        caption='Performance of each portfolio strategy.',
        label='tab:performance',
        notes=[
            common,
            'The certainty equivalent is the annualised certain return an '
            f'investor with CRRA utility at $\\gamma={gamma:g}$ would accept '
            'in place of the realised series.',
            'Turnover is the annualised sum of absolute weight changes. '
            'Weight entropy is $-\\sum_i w_i \\ln w_i$ averaged over '
            'rebalance dates; a lower value is a more concentrated book, and '
            '$\\exp(\\cdot)$ of it is an effective number of holdings.',
            'Tests of the differences between these strategies and the '
            'equal-weight benchmark are reported in '
            'Table~\\ref{tab:inference}.']),
        'table1_performance')

    infer, infer_kind, infer_star = table_inference(
        returns, gamma=gamma, n_boot=n_boot, seed=seed)
    _write(infer, 'table2_inference')
    _write_tex(render_latex(
        [(infer, infer_kind, infer_star)],
        caption='Difference tests against the equal-weight benchmark.',
        label='tab:inference',
        notes=[
            common+' Each column reports the strategy minus the equal-weight '
            'benchmark, whose levels are in Table~\\ref{tab:performance}.',
            'All tests are two-sided and paired. Each is a delta method over '
            'sample moments with a prewhitened quadratic-spectral HAC '
            'long-run covariance (Andrews, 1991; Andrews and Monahan, 1992); '
            'the reported $p$-value is from a studentized circular block '
            f'bootstrap with block length 5 and {n_boot} resamples '
            '(Ledoit and Wolf, 2008).',
            'The panels are ordered from the statistic that does not adjust '
            'for risk to the one that penalises it most heavily. A mean '
            'return difference is not scale-invariant: it rises with leverage '
            'without the accompanying volatility being charged for. '
            'Significance in Panel A alongside insignificance in Panels B '
            'and C is therefore a statement about volatility, not about '
            'skill.',
            'With three tests across four strategies these are twelve '
            'comparisons; the notes on multiple testing in the text apply.',
            '$^{***}$, $^{**}$ and $^{*}$ denote significance at the 1\\%, '
            '5\\% and 10\\% levels.']),
        'table2_inference')

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
