"""Panel A: two assets with matched path volatility, only terminal risk differs.

The companion to `experiment_thought` (now Panel B). That module's Asset A is a
steady compounder with almost no volatility anywhere in its life (1% annual),
so its Sharpe advantage over Asset B is open to the objection that it is really
about path smoothness, not about terminal-wealth certainty. This module softens
that objection rather than removing it outright: both assets here are built the
same way B was -- a random phase, then a final year solved so terminal wealth
lands on a target exactly -- so both have a genuinely volatile path, not the
near-zero one Panel B's Asset A had. Asset 1 is still the calmer of the two
(16% annual vol against Asset 2's 19%), which is what lets a Sharpe/terminal-
wealth disagreement appear at all; the point is that the gap is far narrower
than Panel B's 1% against 40%, and Asset 1's terminal wealth still carries
genuine path risk rather than none. 16% is chosen so Asset 1's realised mean
Sharpe lands close to 0.93, and its mean return is set to 15% -- Table 1's
`Max Sharpe, B&H N` column, the max-Sharpe-objective variant the Discussion's
paradox is actually about. Asset 2's 25% mean return and ~1.28 realised Sharpe
correspondingly target Table 1's `CRRA, B&H N` column. Both synthetic assets
are calibrated to the paper's own numbers, not picked freehand -- Asset 1 is
not a stand-in for B&H itself (Sharpe 1.02 in Table 1), which plays no role
here.

* **Asset 1** -- the target is a fixed constant: a 15% annualised compound
  return, every realisation, no exception. Its random phase draws at the same
  15% mean, 16% annual volatility.
* **Asset 2** -- the target is itself drawn, once per realisation, from a
  tight Gaussian: 25% mean, 2% annual volatility. So Asset 2 keeps a little
  terminal-wealth risk, far less than its path volatility, but not zero. Its
  random phase draws at the same 25% mean, 19% annual volatility.

Asset 2's expected terminal wealth is far higher than Asset 1's (a 25% versus
15% compound rate over 11 years is 11.6x against 4.7x) and its terminal risk is
small (2% annual volatility on the compounded rate). A long-horizon investor
comparing the two on terminal wealth alone would take Asset 2 without a second
thought. Whether a Sharpe ratio computed on the monthly path agrees is an
empirical question this module answers by simulation, not by construction --
unlike Panel B, where Asset A's near-total absence of path risk makes the
Sharpe result close to foreordained.

No hypothesis test is run here, deliberately: the paper does not yet test
CRRA's realised Sharpe against max-Sharpe's realised Sharpe either, so it would
overstate this illustration to hold it to a higher standard than the result it
illustrates. `win_shares` only counts, out of `n_sim` realisations, how often
each asset has the larger value of each statistic.

Run:
    python -m src.Empirical_Analysis.experiment_thought_panel_a
    python -m src.Empirical_Analysis.experiment_thought_panel_a --n-sim 5000
"""
import argparse
import math
import numpy as np
import pandas as pd

from .metrics import MONTHS_PER_YEAR
from .experiment_thought import describe, GAMMA

N_MONTH: int = 132              # 11 years, matching Panel B and the backtest.
N_RANDOM: int = 120             # First 10 years random; the last 12 forced.
N_FORCED: int = N_MONTH-N_RANDOM
START: str = '2015-01-31'
HORIZON_YEARS: float = N_MONTH/MONTHS_PER_YEAR

# Asset 1: the target is a fixed constant, so it carries no terminal-wealth
# risk at all -- the cleanest possible "necessitas".
ASSET1_MEAN: float = 0.15
ASSET1_STD: float = 0.166
ASSET1_TARGET: float = 0.15

# Asset 2: the target is drawn once per realisation, so a little
# terminal-wealth risk survives -- far less than the path risk, but not zero.
ASSET2_MEAN: float = 0.25
ASSET2_STD: float = 0.19
ASSET2_TARGET_MEAN: float = 0.25
ASSET2_TARGET_STD: float = 0.02

# See `experiment_thought.FLOOR`: the same guard against a sub -100% month, at
# a volatility here (19-20% annual, about 5.5-5.8% monthly) where it is even
# further from ever binding.
FLOOR: float = -0.99


def _monthly (mean_ann: float, std_ann: float)-> tuple[float, float]:
    """Annual mean and volatility to monthly, under i.i.d. months."""
    return mean_ann/MONTHS_PER_YEAR, std_ann/math.sqrt(MONTHS_PER_YEAR)


def _forced_path (rng: np.random.Generator, mean_ann: float, std_ann: float,
                  target_wealth: float)-> tuple[np.ndarray, int]:
    """`N_RANDOM` random months, then `N_FORCED` solved to hit `target_wealth`.

    Args:
        rng (np.random.Generator): Shared across both assets in `simulate`, so
            a seed reproduces the whole realisation.
        mean_ann (float): Annual mean of the random phase.
        std_ann (float): Annual volatility of the random phase.
        target_wealth (float): Terminal wealth the full `N_MONTH` path must
            reach exactly -- fixed for Asset 1, realisation-specific for
            Asset 2.

    Returns:
        tuple: The `N_MONTH` monthly returns, and how many random months hit
            `FLOOR`.
    """
    mu, sd = _monthly(mean_ann, std_ann)
    r: np.ndarray = np.empty(N_MONTH)
    r[:N_RANDOM] = np.maximum(rng.normal(mu, sd, N_RANDOM), FLOOR)
    n_floored: int = int(np.isclose(r[:N_RANDOM], FLOOR).sum())
    wealth_random: float = float(np.prod(1.0+r[:N_RANDOM]))
    gross: float = (target_wealth/wealth_random)**(1.0/N_FORCED)
    r[N_RANDOM:] = gross-1.0
    return r, n_floored


def simulate (seed: int)-> tuple[pd.Series, pd.Series, dict]:
    """One 11-year realisation of Asset 1 and Asset 2.

    Args:
        seed (int): Seeds both assets in sequence -- Asset 1's random phase,
            then Asset 2's target draw, then Asset 2's random phase -- so a
            realisation is reproducible.

    Returns:
        tuple: Asset 1's monthly returns, Asset 2's, and a dict with Asset 2's
            drawn target `r_star`, both terminal-wealth targets, and each
            asset's floored-month count.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    idx: pd.DatetimeIndex = pd.date_range(START, periods=N_MONTH, freq='ME')

    target_1: float = (1.0+ASSET1_TARGET)**HORIZON_YEARS
    r1, floor_1 = _forced_path(rng, ASSET1_MEAN, ASSET1_STD, target_1)

    r_star: float = float(rng.normal(ASSET2_TARGET_MEAN, ASSET2_TARGET_STD))
    target_2: float = (1.0+r_star)**HORIZON_YEARS
    r2, floor_2 = _forced_path(rng, ASSET2_MEAN, ASSET2_STD, target_2)

    return (pd.Series(r1, index=idx, name='asset_1'),
            pd.Series(r2, index=idx, name='asset_2'),
            {'r_star': r_star, 'target_wealth_1': target_1,
             'target_wealth_2': target_2, 'n_floored_1': floor_1,
             'n_floored_2': floor_2})


def run_monte_carlo (n_sim: int = 10000)-> pd.DataFrame:
    """Both assets' `describe` statistics, one row per realisation.

    Args:
        n_sim (int): Realisations, each with its own seed. Defaults to 10000,
            matching Panel B.

    Returns:
        pd.DataFrame: One row per realisation, with each asset's terminal
            wealth, annualised return and Sharpe ratio, plus Asset 2's drawn
            target `r_star`.
    """
    rows: list[dict] = []
    for sim in range(n_sim):
        r1, r2, info = simulate(sim)
        d1: pd.Series = describe(r1, GAMMA)
        d2: pd.Series = describe(r2, GAMMA)
        rows.append({
            'mean_return_1': float(r1.mean()), 'mean_return_2': float(r2.mean()),
            'sharpe_1': d1['sharpe'], 'sharpe_2': d2['sharpe'],
            'crra_ce_1': d1['ann_crra_ce'], 'crra_ce_2': d2['ann_crra_ce'],
            'terminal_wealth_1': d1['terminal_wealth'],
            'terminal_wealth_2': d2['terminal_wealth'],
            'ann_return_1': d1['ann_return_geom'],
            'ann_return_2': d2['ann_return_geom'],
            'r_star': info['r_star']})
    return pd.DataFrame(rows)


def win_shares (draws: pd.DataFrame,
                metrics: tuple[str, ...] = ('mean_return', 'sharpe', 'crra_ce')
                )-> pd.DataFrame:
    """How often each asset has the larger value -- a count, not a test.

    Deliberately not a hypothesis test: see the module docstring. This is the
    same question `experiment_thought.summarise_monte_carlo`'s
    `share_t_above_0` answers for Panel B, asked directly of the statistic
    instead of through a t-statistic.

    Args:
        draws (pd.DataFrame): `run_monte_carlo`'s output.
        metrics (tuple[str, ...]): Column prefixes to compare. Defaults to
            `('mean_return', 'sharpe', 'crra_ce')`, Panel B's own three
            columns -- mean monthly return (`mean_inference`'s definition, the
            plain sample mean, not annualised), Sharpe ratio, and CRRA
            certainty equivalent at the same `GAMMA` Panel B uses.

    Returns:
        pd.DataFrame: One column per metric, rows `asset_1_higher`,
            `asset_2_higher`, `ties` -- fractions in [0, 1], matching
            `experiment_thought.summarise_monte_carlo`'s `share_t_above_0`
            convention. `tables.py` scales to a percentage at render time.
    """
    out: dict[str, pd.Series] = {}
    for metric in metrics:
        a1: pd.Series = draws[f'{metric}_1']
        a2: pd.Series = draws[f'{metric}_2']
        out[metric] = pd.Series({
            'asset_1_higher': float((a1 > a2).mean()),
            'asset_2_higher': float((a2 > a1).mean()),
            'ties': float((a1 == a2).mean())})
    return pd.DataFrame(out)


def describe_draws (draws: pd.DataFrame)-> pd.DataFrame:
    """Mean, spread and percentiles of each tracked statistic, both assets."""
    cols: list[str] = ['mean_return_1', 'mean_return_2', 'sharpe_1', 'sharpe_2',
                       'crra_ce_1', 'crra_ce_2', 'terminal_wealth_1',
                       'terminal_wealth_2', 'ann_return_1', 'ann_return_2']
    return draws[cols].describe(percentiles=[0.05, 0.5, 0.95]).T


def _cli ()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, default=0,
                        help='realisation shown in the single-run tables')
    parser.add_argument('--n-sim', type=int, default=10000,
                        help='Monte Carlo realisations; 0 skips it')
    return parser.parse_args()


if __name__ == '__main__':
    args = _cli()
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:,.4f}'.format):
        r1, r2, info = simulate(args.seed)
        print(f'=== one realisation (seed={args.seed}) ===\n')
        print(f"Asset 2's drawn target R*         : "
              f"{info['r_star']:>12.2%}")
        print(f'Asset 1 terminal-wealth target     : '
              f"{info['target_wealth_1']:>12.4f}")
        print(f'Asset 2 terminal-wealth target     : '
              f"{info['target_wealth_2']:>12.4f}")
        if info['n_floored_1'] or info['n_floored_2']:
            print(f"months floored at {FLOOR:.0%} (1, 2)       : "
                  f"{info['n_floored_1']:>6d}, {info['n_floored_2']:<6d}")
        table: pd.DataFrame = pd.concat(
            [describe(r1, GAMMA), describe(r2, GAMMA)], axis=1)
        print(f'\n--- performance ---\n{table}')

        if args.n_sim > 0:
            draws: pd.DataFrame = run_monte_carlo(args.n_sim)
            print(f'\n=== monte carlo, {args.n_sim:,} realisations ===')
            print(f'\n--- win shares, no hypothesis test (%) ---\n'
                  f'{win_shares(draws)*100}')
            print(f'\n--- underlying spread ---\n{describe_draws(draws)}')
