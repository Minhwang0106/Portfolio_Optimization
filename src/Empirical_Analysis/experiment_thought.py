"""Can a 40x asset be judged significantly *worse* than a 3x one?

A thought experiment on what the three difference tests actually answer, and the
companion to the warning at the end of `mean_inference`: there the mean test
rewarded volatility it should not have, here the Sharpe test punishes an asset
whose terminal wealth is thirteen times the alternative's. Neither is a defect.
They are different questions, and the experiment exists to make the gap between
them impossible to miss.

The headline result, over 10,000 realisations of the pair: the Sharpe test
favours A in 99.9% of them, the mean return test favours B in 94.2%, and the
certainty equivalent test separates the two in 0.2%. Three statistics, three
answers -- A wins, B wins, and no detectable difference -- on identical data.

Two assets, 132 monthly observations -- 11 years, matching the backtest's T:

* **Asset A** -- Gaussian, 10% annual mean and 0.0001 annual *variance*, i.e. 1%
  annual volatility. A steady compounder. One dollar becomes about
  `1.1^11 = 2.85`.
* **Asset B** -- Gaussian, 40% annual mean and 40% annual volatility for the
  first 120 months. The last 12 months are then *deterministic*: whatever the
  first ten years happened to produce, the final year supplies exactly the gross
  return needed to pin terminal wealth at 40x. One dollar becomes forty, with
  certainty, in every realisation.

Note where the certainty sits. `TARGET_WEALTH` is reached exactly, always, but
the path is not fixed: wealth after ten years is random, so the final year's
required return is random too -- and occasionally negative, when the first
decade overshot 40x on its own.

The test statistics are all reported as **A minus B**, so a positive difference
means the steady compounder wins.

Two caveats, stated here because the numbers are quotable and the caveats are
not optional:

* **B's population certainty equivalent does not exist, though at this
  volatility its sample estimate behaves.** For a Gaussian simple return the
  integral `E[(1+R)^(1-gamma)]` diverges at `R = -1` when `gamma > 1`, so the
  population mean utility is infinite and `mu^(k/(1-gamma)) - 1` collapses to
  -1 regardless of the parameters. Whether that bites is a question about the
  sample, not the population: at `B_STD = 0.40` a month of -100% sits about
  nine standard deviations out and is never drawn, so the estimate is stable
  and the CE column can be read as printed. At `B_STD = 0.80` it very much bit
  -- the CE t-statistic had a standard deviation of 38 and a 99th percentile of
  284, driven entirely by each realisation's worst month.
  `report_ce_tail_sensitivity` is the diagnostic for this, and
  `distribution='lognormal'` is the specification to switch to if the
  volatility is ever raised again.
* **The p-values are not strictly valid, and the effect sizes are.** The last 12
  observations are a constant by construction, a deliberate structural break
  that violates the stationarity both the HAC estimator and the circular block
  bootstrap assume. It does not matter at these effect sizes -- the Sharpe
  difference runs to about 14 standard errors -- but quote the t-statistics and
  the point estimates, not the p-values.

Run:
    python -m src.Empirical_Analysis.experiment_thought
    python -m src.Empirical_Analysis.experiment_thought --n-sim 5000
    python -m src.Empirical_Analysis.experiment_thought --distribution lognormal
"""
import argparse
import math
import numpy as np
import pandas as pd

from .metrics import (MONTHS_PER_YEAR, max_drawdown, crra_certainty_equivalent,
                      _annualise_geometric)
from .sharpe_inference import sharpe_difference_test
from .ce_inference import ce_difference_test
from .mean_inference import mean_difference_test

N_MONTH: int = 132              # 11 years.
N_RANDOM: int = 120             # The first 10; the last 12 are deterministic.
GAMMA: float = 5.0
START: str = '2015-01-31'

# Annual parameters as specified. A's 0.0001 is a variance, so 1% volatility.
A_MEAN: float = 0.10
A_VAR: float = 0.0001
B_MEAN: float = 0.40
B_STD: float = 0.40
TARGET_WEALTH: float = 40.0

# A Gaussian simple return can fall below -100%, where gross wealth turns
# negative and CRRA utility does not exist. At B's 11.5% monthly volatility that
# is a 9-sigma event and never actually happens, so it is floored rather than
# rejected -- but the floor is counted and reported, because at a higher
# `B_STD` it would start to bind and the experiment would then be measuring the
# floor instead of the asset.
FLOOR: float = -0.99
DISTRIBUTION: tuple[str, ...] = ('gaussian', 'lognormal')

DESCRIBE_ROWS: tuple[str, ...] = (
    'terminal_wealth', 'ann_return_geom', 'ann_vol', 'max_drawdown',
    'worst_month', 'sharpe', 'ann_crra_ce')
TEST_ROWS: tuple[str, ...] = ('diff', 'diff_se', 't_stat', 'pval_hac',
                              'pval_boot')
# All three difference tests this project implements, in the order they are
# reported. The mean test is included precisely because it disagrees: see the
# module docstring.
TEST_COLUMNS: tuple[str, ...] = ('sharpe', 'crra_ce', 'mean_return')


def _monthly (mean_ann: float, std_ann: float)-> tuple[float, float]:
    """Annual mean and volatility to monthly, under i.i.d. months."""
    return mean_ann/MONTHS_PER_YEAR, std_ann/math.sqrt(MONTHS_PER_YEAR)


def _draw_b (rng: np.random.Generator, distribution: str)-> np.ndarray:
    """B's first 120 random months, at 40% annual mean and 40% annual vol.

    The lognormal branch matches the *arithmetic* mean of the Gaussian one, not
    its log mean, so the two specifications differ only in shape and the CE
    comparison between them is like for like. Its log volatility is set to the
    same monthly figure, which leaves the arithmetic volatility a little higher
    -- unavoidable, since a lognormal cannot match both moments and stay
    lognormal.
    """
    mu, sd = _monthly(B_MEAN, B_STD)
    if distribution == 'gaussian':
        return np.maximum(rng.normal(mu, sd, N_RANDOM), FLOOR)
    # E[exp(N(m, s^2))] = exp(m + s^2/2) = 1 + mu.
    log_mean: float = math.log1p(mu)-0.5*sd**2
    return np.exp(rng.normal(log_mean, sd, N_RANDOM))-1.0


def simulate (seed: int,
              distribution: str = 'gaussian')-> tuple[pd.Series, pd.Series,
                                                      dict]:
    """One 11-year realisation of the pair.

    Args:
        seed (int): Seeds both assets, so a realisation is reproducible.
        distribution (str): Shape of B's 120 random months -- `'gaussian'` as
            specified, or `'lognormal'` for the robustness check described in
            the module docstring. A is Gaussian either way; at 1% annual
            volatility the distinction is invisible.

    Returns:
        tuple: A's monthly returns, B's, and a dict describing how B reached
            40x -- the wealth it had after ten years, the return the final year
            had to deliver, and how many months hit `FLOOR`.
    """
    if distribution not in DISTRIBUTION:
        raise ValueError(f'distribution must be one of {DISTRIBUTION}; '
                         f'got {distribution!r}')
    rng: np.random.Generator = np.random.default_rng(seed)
    idx: pd.DatetimeIndex = pd.date_range(START, periods=N_MONTH, freq='ME')

    a_mu, a_sd = _monthly(A_MEAN, math.sqrt(A_VAR))
    r_a: np.ndarray = rng.normal(a_mu, a_sd, N_MONTH)

    r_b: np.ndarray = np.empty(N_MONTH)
    r_b[:N_RANDOM] = _draw_b(rng, distribution)
    n_floored: int = int(np.isclose(r_b[:N_RANDOM], FLOOR).sum())

    # The last year is whatever it has to be. Wealth after ten years is random,
    # so the required monthly gross return is too -- it is the *terminal* value
    # that is certain, not the path to it.
    wealth_10y: float = float(np.prod(1.0+r_b[:N_RANDOM]))
    gross: float = (TARGET_WEALTH/wealth_10y)**(1.0/MONTHS_PER_YEAR)
    r_b[N_RANDOM:] = gross-1.0

    return (pd.Series(r_a, index=idx, name='asset_A'),
            pd.Series(r_b, index=idx, name='asset_B'),
            {'wealth_10y': wealth_10y, 'final_year_monthly': gross-1.0,
             'final_year_total': TARGET_WEALTH/wealth_10y-1.0,
             'n_floored': n_floored})


def describe (returns: pd.Series, gamma: float = GAMMA)-> pd.Series:
    """Terminal wealth and the performance statistics, one asset.

    The same definitions `metrics` uses for the backtest -- geometric rather
    than arithmetic annualisation, drawdown on compounded wealth, the CE
    inverted from realised CRRA utility -- so a number here means what the same
    name means in `summary.csv`.
    """
    return pd.Series({
        'terminal_wealth': float(np.prod(1.0+returns)),
        'ann_return_geom': _annualise_geometric(returns),
        'ann_vol': float(returns.std(ddof=1))*math.sqrt(MONTHS_PER_YEAR),
        'max_drawdown': max_drawdown(returns),
        'worst_month': float(returns.min()),
        'sharpe': (float(returns.mean()/returns.std(ddof=1))
                   *math.sqrt(MONTHS_PER_YEAR)),
        'ann_crra_ce': crra_certainty_equivalent(returns, gamma)},
        name=returns.name)[list(DESCRIBE_ROWS)]


def compare (r_a: pd.Series, r_b: pd.Series, gamma: float = GAMMA,
             n_boot: int = 4999, seed: int = 0)-> pd.DataFrame:
    """Both difference tests, oriented **A minus B**.

    A is passed as the candidate and B as the benchmark, so a positive `diff`
    means the steady compounder is ahead and a positive `t_stat` points the same
    way. The risk-free rate is zero in this experiment, so total and excess
    returns coincide and the Sharpe test can take these series directly.

    Args:
        r_a (pd.Series): Asset A's monthly returns.
        r_b (pd.Series): Asset B's, on the same index.
        gamma (float): Relative risk aversion for the CE test. Defaults to 5.
        n_boot (int): Bootstrap resamples for each test. Defaults to 4999; 0
            leaves `pval_boot` as NaN and returns only the HAC column, which is
            what the Monte Carlo uses.
        seed (int): Seeds both bootstraps. Defaults to 0.

    Returns:
        pd.DataFrame: `TEST_ROWS` by `['sharpe', 'crra_ce']`. `t_stat` is
            `diff/diff_se`, the statistic `pval_hac` is read from.
    """
    results: tuple[tuple[str, pd.Series, str, str], ...] = (
        ('sharpe', sharpe_difference_test(r_a, r_b, n_boot=n_boot, seed=seed),
         'sr_diff', 'sr_diff_se'),
        ('crra_ce', ce_difference_test(r_a, r_b, gamma=gamma, n_boot=n_boot,
                                       seed=seed), 'ce_diff', 'ce_diff_se'),
        ('mean_return', mean_difference_test(r_a, r_b, n_boot=n_boot,
                                             seed=seed),
         'mean_diff', 'mean_diff_se'))
    out: dict[str, pd.Series] = {}
    for name, res, diff, err in results:
        se: float = float(res[err])
        out[name] = pd.Series({
            'diff': float(res[diff]), 'diff_se': se,
            't_stat': float(res[diff])/se if se > 0 else float('nan'),
            'pval_hac': float(res['pval_hac']),
            'pval_boot': float(res['pval_boot'])})
    return pd.DataFrame(out)[list(TEST_COLUMNS)].loc[list(TEST_ROWS)]


def run_single (seed: int = 0, gamma: float = GAMMA, n_boot: int = 4999,
                distribution: str = 'gaussian')-> tuple[pd.DataFrame,
                                                        pd.DataFrame, dict]:
    """One realisation: the per-asset table and the test table."""
    r_a, r_b, info = simulate(seed, distribution=distribution)
    table: pd.DataFrame = pd.concat(
        [describe(r_a, gamma), describe(r_b, gamma)], axis=1)
    tests: pd.DataFrame = compare(r_a, r_b, gamma=gamma, n_boot=n_boot,
                                  seed=seed)
    return table, tests, info


def run_monte_carlo (n_sim: int = 10000, gamma: float = GAMMA,
                     distribution: str = 'gaussian',
                     n_boot: int = 0)-> pd.DataFrame:
    """The sampling distribution of the two t-statistics, A minus B.

    The bootstrap is off by default. A t-statistic is `diff/se_hac` and needs no
    resampling, so at `n_sim=5000` skipping it is the difference between about a
    minute and about a day; pass `n_boot > 0` only if the bootstrap p-value's
    distribution is itself the object of interest.

    Args:
        n_sim (int): Realisations, each with its own seed. Defaults to 10000.
        gamma (float): Relative risk aversion for the CE test. Defaults to 5.
        distribution (str): Passed to `simulate`.
        n_boot (int): Bootstrap resamples per realisation. Defaults to 0.

    Returns:
        pd.DataFrame: One row per realisation -- the two t-statistics, the two
            differences, the two HAC p-values, and the diagnostics that explain
            them (`wealth_10y`, `final_year_monthly`, each asset's Sharpe, CE,
            drawdown and worst month).
    """
    rows: list[dict] = []
    for sim in range(n_sim):
        r_a, r_b, info = simulate(sim, distribution=distribution)
        tests: pd.DataFrame = compare(r_a, r_b, gamma=gamma, n_boot=n_boot,
                                      seed=sim)
        stat_a: pd.Series = describe(r_a, gamma)
        stat_b: pd.Series = describe(r_b, gamma)
        row: dict = {}
        for name in TEST_COLUMNS:
            row[f't_{name}'] = tests.at['t_stat', name]
            row[f'diff_{name}'] = tests.at['diff', name]
            row[f'pval_{name}'] = tests.at['pval_hac', name]
        rows.append({
            **row,
            'sharpe_a': stat_a['sharpe'], 'sharpe_b': stat_b['sharpe'],
            'ce_a': stat_a['ann_crra_ce'], 'ce_b': stat_b['ann_crra_ce'],
            'mdd_a': stat_a['max_drawdown'], 'mdd_b': stat_b['max_drawdown'],
            'worst_a': stat_a['worst_month'],
            'worst_b': stat_b['worst_month'],
            'wealth_10y': info['wealth_10y'],
            'final_year_monthly': info['final_year_monthly']})
    return pd.DataFrame(rows)


def summarise_monte_carlo (draws: pd.DataFrame,
                           alpha: float = 0.05)-> pd.DataFrame:
    """The t-statistic distribution, one column per test.

    Quantiles rather than a histogram, because each test's distribution sits
    almost entirely on one side of zero and the only questions are which side
    and how far. `share_t_above_0` is the fraction of realisations in which A --
    the 3x asset -- is ahead on the point estimate, and it is the row to read
    first: the Sharpe and CE columns put it at 1, the mean column at 0.

    The two `reject_for_` rows split the rejections by direction, which is what
    makes the disagreement between the three tests legible. A test that rejects
    99% of the time tells you nothing on its own; a test that rejects 99% of the
    time *for the opposite asset* tells you the three statistics are not
    measuring the same thing.
    """
    out: dict[str, pd.Series] = {}
    for name in TEST_COLUMNS:
        t: pd.Series = draws[f't_{name}']
        pval: str = f'pval_{name}'
        out[name] = pd.Series({
            'mean': t.mean(), 'std': t.std(ddof=1), 'min': t.min(),
            'p1': t.quantile(0.01), 'p5': t.quantile(0.05),
            'median': t.median(), 'p95': t.quantile(0.95),
            'p99': t.quantile(0.99), 'max': t.max(),
            'share_t_above_0': float((t > 0).mean()),
            f'reject_at_{alpha:.0%}': float((draws[pval] < alpha).mean()),
            f'reject_for_A_at_{alpha:.0%}': float(
                ((draws[pval] < alpha)&(t > 0)).mean()),
            f'reject_for_B_at_{alpha:.0%}': float(
                ((draws[pval] < alpha)&(t < 0)).mean())})
    return pd.DataFrame(out)


def report_ce_tail_sensitivity (seed: int = 0, gamma: float = GAMMA,
                                n_worst: int = 3,
                                distribution: str = 'gaussian')-> pd.DataFrame:
    """How much of B's certainty equivalent rests on its handful of worst months.

    The diagnostic behind the first caveat in the module docstring. If dropping
    one observation moves the CE by tens of percentage points, the statistic is
    reporting the minimum draw rather than the distribution, and no amount of
    bootstrap will fix that.
    """
    _, r_b, _ = simulate(seed, distribution=distribution)
    utility: pd.Series = (1.0+r_b)**(1.0-gamma)
    worst: pd.Index = utility.sort_values(ascending=False).index[:n_worst]
    rows: list[dict] = [{
        'months_dropped': 0, 'dropped_return': float('nan'),
        'share_of_mean_utility': float('nan'),
        'ann_crra_ce': crra_certainty_equivalent(r_b, gamma)}]
    for k in range(1, n_worst+1):
        rows.append({
            'months_dropped': k,
            'dropped_return': float(r_b[worst[k-1]]),
            'share_of_mean_utility': float(utility[worst[k-1]]/utility.sum()),
            'ann_crra_ce': crra_certainty_equivalent(r_b.drop(worst[:k]),
                                                     gamma)})
    return pd.DataFrame(rows).set_index('months_dropped')


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, default=0,
                        help='realisation shown in the single-run tables')
    parser.add_argument('--n-sim', type=int, default=10000,
                        help='Monte Carlo realisations; 0 skips it')
    parser.add_argument('--n-boot', type=int, default=4999,
                        help='bootstrap resamples for the single run')
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--distribution', choices=DISTRIBUTION,
                        default='gaussian')
    parser.add_argument('--alpha', type=float, default=0.05)
    return parser.parse_args()


if __name__ == '__main__':
    args = _cli()
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:,.4f}'.format):
        table, tests, info = run_single(seed=args.seed, gamma=args.gamma,
                                        n_boot=args.n_boot,
                                        distribution=args.distribution)
        print(f'=== one realisation (seed={args.seed}, '
              f'{args.distribution}) ===\n')
        print(f"B's wealth after 10 random years : "
              f"{info['wealth_10y']:>12,.4f}")
        print(f'required final-year total return : '
              f"{info['final_year_total']:>12.2%}")
        print(f'i.e. per month, for 12 months    : '
              f"{info['final_year_monthly']:>12.2%}")
        if info['n_floored']:
            print(f'months floored at {FLOOR:.0%}           : '
                  f"{info['n_floored']:>12d}")
        print(f'\n--- performance ---\n{table}')
        print(f'\n--- H0: A equals B, two-sided, differences are A minus B '
              f'---\n{tests}')
        print(f'\n--- how much of B\'s CE is the worst few months? ---\n'
              f'{report_ce_tail_sensitivity(args.seed, args.gamma, distribution=args.distribution)}')

        if args.n_sim > 0:
            draws: pd.DataFrame = run_monte_carlo(
                n_sim=args.n_sim, gamma=args.gamma,
                distribution=args.distribution)
            print(f'\n=== monte carlo, {args.n_sim:,} realisations '
                  f'({args.distribution}) ===')
            print(f'\n--- t-statistic distribution, A minus B ---\n'
                  f'{summarise_monte_carlo(draws, alpha=args.alpha)}')
            print(f'\n--- underlying spread ---\n'
                  f"{draws[['sharpe_a', 'sharpe_b', 'ce_a', 'ce_b', 'mdd_a', 'mdd_b', 'worst_a', 'worst_b', 'wealth_10y', 'final_year_monthly']].describe(percentiles=[0.05, 0.5, 0.95])}")
