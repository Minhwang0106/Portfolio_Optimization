"""Run the whole project end to end: collect data, backtest, build tables.

    python main.py                                # everything, from scratch
    python main.py --skip-data                    # panels already collected
    python main.py --skip-data --skip-backtest    # tables only, from a saved run
    python main.py --only equal_weight epo ppp    # skip the slow proposed-model runs
    python main.py --skip-experiment              # skip Table 3's Monte Carlo

Three stages, each a thin wrapper around the module that actually does the
work -- nothing here is reimplemented, so this can't drift from
`python -m src.Data.run`, `python -m src.Empirical_Analysis.run` or
`python -m src.Empirical_Analysis.tables` run by hand:

1. **Data** (`src.Data.run.main`) -- fetches prices from Yahoo Finance,
   accounting data from SEC EDGAR, and rebuilds the point-in-time universe.
   Network-bound, hours on a full run.
2. **Backtest** (`src.Empirical_Analysis.run.run_all`) -- backtests all five
   strategies and writes `Data/Result/raw_backtest/`. The proposed model's two
   variants are the long pole; see `src/Proposed_Model/README.md`'s
   Performance section -- a full run is on the order of ten hours.
3. **Tables** (`src.Empirical_Analysis.tables.build_all`) -- reads
   `raw_backtest/` and writes the three reported tables as CSV and LaTeX.

Each stage can be skipped independently, so an interrupted run resumes
without paying for what already finished: `--skip-data` trusts the panels on
disk, `--skip-backtest` trusts `Data/Result/raw_backtest/`. Finer control than
what's exposed here -- `--resume`-ing an interrupted fetch, `--rebuild`-ing a
backtest from saved weights, a custom output directory -- goes through the
underlying module directly; see `README.md` and each subpackage's own
`README.md` for the full flag reference.
"""
import argparse
import time
from pathlib import Path


def _banner (title: str)-> None:
    print(f'\n{"="*70}\n{title}\n{"="*70}', flush=True)


def _elapsed (start: float)-> str:
    seconds: float = time.time()-start
    return f'{seconds/60:.1f} min' if seconds >= 60 else f'{seconds:.1f} s'


def run_data_stage (resume: bool)-> None:
    from src.Data.run import main as data_main
    argv: list[str] = ['--resume'] if resume else []
    data_main(argv)


def run_backtest_stage (n_workers: int, n_samples: int,
                        only: tuple[str, ...]|None, skip: tuple[str, ...],
                        seed: int)-> None:
    from src.Empirical_Analysis.run import run_all
    _, table = run_all(only=only, skip=skip, n_workers=n_workers,
                       n_samples=n_samples, seed=seed)
    import pandas as pd
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:,.4f}'.format):
        print(table)


def run_tables_stage (n_sim: int, skip_experiment: bool, seed: int)-> None:
    from src.Empirical_Analysis.tables import build_all
    for path in build_all(n_sim=n_sim, seed=seed,
                          skip_experiment=skip_experiment):
        try:
            shown = path.relative_to(Path.cwd())
        except ValueError:
            shown = path
        print(f'  {shown}')


def main (argv: list[str]|None = None)-> None:
    parser = argparse.ArgumentParser(
        description='Run the whole project: collect data, backtest, build tables.')
    parser.add_argument('--skip-data', action='store_true',
                        help='skip data collection; use the panels already on disk')
    parser.add_argument('--data-resume', action='store_true',
                        help="pass --resume to the data stage, picking up an "
                             "interrupted collection instead of starting over")
    parser.add_argument('--skip-backtest', action='store_true',
                        help='skip the backtest; use the results already '
                             'saved to Data/Result/raw_backtest/')
    parser.add_argument('--only', nargs='+', default=None,
                        help='backtest just these strategies (see '
                             'src/Empirical_Analysis/run.py:STRATEGIES)')
    parser.add_argument('--skip', nargs='+', default=(),
                        help='backtest strategies to leave out; ignored if --only is given')
    parser.add_argument('--n-workers', type=int, default=5,
                        help='processes to spread the backtested strategies over')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='simulated paths per ticker in the proposed model; '
                             'lower this to iterate faster (see '
                             'src/Proposed_Model/README.md)')
    parser.add_argument('--skip-tables', action='store_true',
                        help='skip building the reported tables')
    parser.add_argument('--skip-experiment', action='store_true',
                        help="skip Table 3's thought-experiment Monte Carlo, the "
                             'slow part of the tables stage')
    parser.add_argument('--n-sim', type=int, default=10000,
                        help="Monte Carlo realisations for Table 3's Panel C")
    parser.add_argument('--seed', type=int, default=0,
                        help='shared seed for the backtest and the tables stage')
    args = parser.parse_args(argv)

    start: float = time.time()

    if args.skip_data:
        print('Skipping data collection (--skip-data).')
    else:
        _banner('Stage 1/3 -- Data collection')
        stage_start: float = time.time()
        run_data_stage(resume=args.data_resume)
        print(f'\nData collection done in {_elapsed(stage_start)}.')

    if args.skip_backtest:
        print('Skipping the backtest (--skip-backtest).')
    else:
        _banner('Stage 2/3 -- Backtest')
        stage_start = time.time()
        run_backtest_stage(n_workers=args.n_workers, n_samples=args.n_samples,
                           only=tuple(args.only) if args.only else None,
                           skip=tuple(args.skip), seed=args.seed)
        print(f'\nBacktest done in {_elapsed(stage_start)}.')

    if args.skip_tables:
        print('Skipping table building (--skip-tables).')
    else:
        _banner('Stage 3/3 -- Tables')
        stage_start = time.time()
        run_tables_stage(n_sim=args.n_sim, skip_experiment=args.skip_experiment,
                         seed=args.seed)
        print(f'\nTables done in {_elapsed(stage_start)}.')

    _banner(f'Finished in {_elapsed(start)}')


# Load-bearing on Windows, not decoration: `run_all`'s process pool re-imports
# this module under the spawn start method, and without the guard that
# re-import would run the whole pipeline again in every worker. See
# `src/Empirical_Analysis/run.py`'s module docstring.
if __name__ == '__main__':
    main()
