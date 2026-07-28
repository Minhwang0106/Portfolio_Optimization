import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from constant import (testing_period,
                       risk_com,
                       risk_correl_com,
                       risk_n_day,
                       risk_n_month,
                       epo_shrinkage,
                       epo_theta,
                       risk_aversion)
from .utils import init_worker, compute_date


class EPO:
    """Enhanced Portfolio Optimization.

    Call `EPO.Config()` once to precompute the per-date risk inputs, then
    instantiate per formation date. Unlike `PPP` and `RIM_PortOp`, `__init__`
    does *not* configure on your behalf: `Config` runs a process pool over every
    date in `testing_period`, which is far too expensive to trigger as a side
    effect of constructing an object.

    Attributes:
        vol, correl, tsmom (dict): Class-level, keyed by stringified date. Set
            by `Config`.
        date (str): This instance's formation date.
    """
    def __init__(self, date:str) -> None:
        self.date = date

    @classmethod
    def Config(cls, com=risk_com, correl_com=risk_correl_com,
                      n_day=risk_n_day, n_month=risk_n_month,
                      n_workers: int | None = None, force: bool = False):
        """Precompute per-date volatility, correlation, and TSMOM signal for all dates.

        Runs `compute_date` (from `utils`) over every date in `testing_period`,
        in parallel via a process pool (unless `n_workers` <= 1), and stores
        the results as class-level attributes `EPO.vol`, `EPO.correl`,
        `EPO.tsmom` so that `portfolio_weight` can look them up per instance.

        Idempotent: returns immediately if already configured, so a repeat call
        cannot silently redo the whole parallel sweep. Pass `force=True` to
        rebuild with different parameters.

        Args:
            com: EWM center-of-mass for volatility, passed to worker init.
            correl_com: EWM center-of-mass for correlation, passed to worker init.
            n_day (int): Trailing window (days) for vol/correlation.
            n_month (int): Trailing window (months) for the TSMOM signal.
            n_workers (int | None): Number of worker processes; defaults to
                cpu_count - 1. Falls back to serial execution if <= 1.
            force (bool): Recompute even if already configured. Needed whenever
                any of the parameters above differ from the first call, since
                otherwise they are ignored. Defaults to False.

        Returns:
            None. Populates `EPO.vol`, `EPO.correl`, `EPO.tsmom` (each a dict
            keyed by stringified date).

        Example:
            >>> EPO.Config()
            >>> '2020-03-31' in EPO.vol
            True
        """
        if getattr(cls,'_configured',False) and not force:
            return
        std_dict: dict[str, pd.Series] = {}
        correl_dict: dict[str, pd.DataFrame] = {}
        tsmom_dict: dict[str, pd.Series] = {}

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

        if n_workers <= 1:
            init_worker(com, correl_com, n_day, n_month)
            results = (compute_date(date) for date in testing_period)
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker,
                                      initargs=(com, correl_com, n_day, n_month)) as ex:
                results = list(ex.map(compute_date, testing_period))

        for result in results:
            if result is None:
                continue
            key, std, correl, tsmom = result
            std_dict[key] = std
            correl_dict[key] = correl
            tsmom_dict[key] = tsmom

        cls.vol = std_dict
        cls.correl = correl_dict
        cls.tsmom = tsmom_dict
        cls._configured: bool = True

    def portfolio_weight (self):
        """Compute Enhanced Portfolio Optimization (EPO) weights for this instance's date.

        Shrinks the correlation matrix toward the identity matrix (via
        `epo_theta`), builds a shrunk covariance matrix from vol and
        correlation, further blends it toward identity (via `epo_shrinkage`),
        then solves for weights proportional to risk-adjusted TSMOM signal.
        Requires `Config` to have been called first to populate `EPO.vol`,
        `EPO.correl`, `EPO.tsmom`.

        Returns:
            np.ndarray: Portfolio weights, one per ticker, in the same order
                as `EPO.correl[self.date]`'s columns.

        Raises:
            RuntimeError: If `Config` has not been called. Checked explicitly
                because the bare AttributeError names `vol`, which does not
                point at the missing step.
            KeyError: If `self.date` is not one of the dates `Config` covered.

        Example:
            >>> EPO.Config()
            >>> EPO('2020-03-31').portfolio_weight()
            array([0.12, -0.05, ...])
        """
        if not getattr(EPO,'_configured',False):
            raise RuntimeError('EPO.Config() must be called before '
                               'portfolio_weight(); it precomputes the '
                               'per-date vol, correlation and TSMOM inputs')
        correl: np.ndarray = EPO.correl[self.date].to_numpy()
        vol: np.ndarray = np.diag(EPO.vol[self.date].to_numpy())
        signal: np.ndarray = EPO.tsmom[self.date].to_numpy()
        n_stock: int = correl.shape[0]
        
        shrink_correl: np.ndarray = correl*epo_theta+(1-epo_theta
                                                      )*np.identity(n_stock)
        epo_correl: np.ndarray = epo_shrinkage*shrink_correl+(
                                1-epo_shrinkage)*np.identity(n_stock)
        epo_Sigma: np.ndarray = vol@epo_correl@vol
        return np.linalg.solve(epo_Sigma,signal)/risk_aversion