import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from constant import (testing_period,
                       risk_com,
                       risk_correl_com,
                       risk_n_day,
                       risk_n_month,
                       epo_shrinkage,
                       epo_shrinkage_grid,
                       epo_w_min_periods,
                       epo_theta,
                       risk_aversion)
from .utils import init_worker, compute_date


def _mean_variance (weight: np.ndarray, signal: np.ndarray,
                    Sigma: np.ndarray, gamma: float)-> float:
    """Negated mean-variance objective, `-(w'mu - gamma/2 w'Sigma w)`.

    The same objective the unconstrained `Sigma^-1 @ signal / gamma` maximises,
    written out so it can be handed to a constrained solver.
    """
    return float(-(weight@signal-0.5*gamma*(weight@Sigma@weight)))


def _mean_variance_jac (weight: np.ndarray, signal: np.ndarray,
                        Sigma: np.ndarray, gamma: float)-> np.ndarray:
    """Gradient of `_mean_variance`, `-(mu - gamma*Sigma@w)`.

    Supplied explicitly because SLSQP would otherwise difference the objective
    once per variable per iteration, and there are ~150 of them at every
    formation date.
    """
    return -(signal-gamma*(Sigma@weight))


def _long_only_weight (Sigma: np.ndarray, signal: np.ndarray,
                       gamma: float)-> np.ndarray:
    """Maximise the mean-variance objective over the long-only simplex.

    Solves `max w'mu - gamma/2 w'Sigma w` subject to `w >= 0` and `sum(w) == 1`.
    Unlike clipping the unconstrained solution, this leaves the covariance
    matrix doing real work -- a name can still be held down because of what it
    is correlated with rather than only because of its own signal.

    Args:
        Sigma (np.ndarray): Shrunk covariance, shape `(n, n)`.
        signal (np.ndarray): Expected excess returns, shape `(n,)`.
        gamma (float): Risk aversion.

    Returns:
        np.ndarray: Weights, shape `(n,)`, non-negative and summing to one.

    Raises:
        ValueError: If the solver does not converge, which the caller records as
            a failed formation date rather than passing an arbitrary vector on.
    """
    n_stock: int = signal.shape[0]
    start: np.ndarray = np.ones(n_stock)/n_stock
    res = minimize(_mean_variance, start, args=(signal, Sigma, gamma),
                   jac=_mean_variance_jac, method='SLSQP',
                   bounds=[(0.0, 1.0)]*n_stock,
                   constraints={'type': 'eq',
                                'fun': lambda w: float(w.sum()-1.0),
                                'jac': lambda w: np.ones_like(w)},
                   options={'maxiter': 500, 'ftol': 1e-12})
    if not res.success:
        raise ValueError(f'EPO long-only solve did not converge: {res.message}')
    # Clip before renormalising: SLSQP satisfies its bounds to solver tolerance,
    # so a weight can land at -1e-17 and turn `n_holdings` into a count of
    # rounding noise.
    weight: np.ndarray = np.clip(res.x, 0.0, None)
    return weight/weight.sum()


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
        candidate (dict[str, dict[float, pd.Series]]): Class-level. Unconstrained
            closed-form weight per date per candidate shrinkage. Set by `Config`.
        realized (dict[str, pd.Series | None]): Class-level. Next month's
            realised excess return per date, keyed like `candidate`. Set by
            `Config`.
        chosen_w (dict[str, float]): Class-level. The shrinkage `_select_w`
            picked for each date -- see `constant.epo_shrinkage`. Set by
            `Config`.
        date (str): This instance's formation date.
    """
    def __init__(self, date:str) -> None:
        self.date = date

    @classmethod
    def Config(cls, com=risk_com, correl_com=risk_correl_com,
                      n_day=risk_n_day, n_month=risk_n_month,
                      n_workers: int | None = None, force: bool = False):
        """Precompute per-date risk inputs, w-selection inputs, and chosen w.

        Runs `compute_date` (from `utils`) over every date in `testing_period`,
        in parallel via a process pool (unless `n_workers` <= 1), and stores
        the results as class-level attributes `EPO.vol`, `EPO.correl`,
        `EPO.tsmom`, `EPO.candidate`, `EPO.realized` so that `portfolio_weight`
        can look them up per instance. Then runs `_select_w` once over all
        dates to populate `EPO.chosen_w` -- see `constant.epo_shrinkage`.

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
            None. Populates `EPO.vol`, `EPO.correl`, `EPO.tsmom`, `EPO.candidate`,
            `EPO.realized` (each a dict keyed by stringified date) and
            `EPO.chosen_w` (a dict of float keyed the same way).

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
        candidate_dict: dict[str, dict[float, pd.Series]] = {}
        realized_dict: dict[str, pd.Series|None] = {}

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
            key, std, correl, tsmom, candidates, realized = result
            std_dict[key] = std
            correl_dict[key] = correl
            tsmom_dict[key] = tsmom
            candidate_dict[key] = candidates
            realized_dict[key] = realized

        cls.vol = std_dict
        cls.correl = correl_dict
        cls.tsmom = tsmom_dict
        cls.candidate = candidate_dict
        cls.realized = realized_dict
        cls.chosen_w = cls._select_w(candidate_dict, realized_dict)
        cls._configured: bool = True

    @staticmethod
    def _select_w (candidate: dict[str, dict[float, pd.Series]],
                   realized: dict[str, pd.Series|None]
                   )-> dict[str, float]:
        """Out-of-sample shrinkage selection (`constant.epo_shrinkage`'s
        docstring has the full rationale).

        Walks the dates in `candidate` in the order `Config` inserted them --
        chronological, since it iterates `testing_period` -- and at each date
        scores every point in `constant.epo_shrinkage_grid` by the realised
        Sharpe ratio its unconstrained closed-form weight earned over every
        *earlier* formation date. A date's own return never contributes to its
        own score, and a date whose grid score has no history yet falls back
        to `constant.epo_shrinkage`.

        Args:
            candidate (dict[str, dict[float, pd.Series]]): Per date, the
                unconstrained weight at each grid point (`compute_date`'s
                third output).
            realized (dict[str, pd.Series | None]): Per date, next month's
                realised excess return, or None at the last date.

        Returns:
            dict[str, float]: Chosen `w` per stringified formation date.
        """
        chosen: dict[str, float] = {}
        history: list[tuple[dict[float, pd.Series], pd.Series]] = []
        for key, cand in candidate.items():
            if len(history) < epo_w_min_periods:
                chosen[key] = epo_shrinkage
            else:
                best_w, best_sr = epo_shrinkage, -np.inf
                for w in epo_shrinkage_grid:
                    rets: list[float] = []
                    for hist_cand, hist_realized in history:
                        x_w: pd.Series = hist_cand[w]
                        common = x_w.index.intersection(hist_realized.index)
                        if len(common) == 0:
                            continue
                        rets.append(float(x_w[common]@hist_realized[common]))
                    if not rets:
                        continue
                    arr: np.ndarray = np.array(rets)
                    sr: float = arr.mean()/(arr.std(ddof=1)+1e-12)
                    if sr > best_sr:
                        best_sr, best_w = sr, w
                chosen[key] = best_w
            this_realized: pd.Series|None = realized[key]
            if this_realized is not None:
                history.append((cand, this_realized))
        return chosen

    def portfolio_weight (self, long_only: bool = True):
        """Compute Enhanced Portfolio Optimization (EPO) weights for this instance's date.

        Shrinks the correlation matrix toward the identity matrix (via
        `epo_theta`), builds a shrunk covariance matrix from vol and
        correlation, further blends it toward identity by `EPO.chosen_w[date]`
        -- the shrinkage `_select_w` picked out-of-sample for this date, see
        `constant.epo_shrinkage` -- then maximises `w'mu - gamma/2 w'Sigma w`
        over the TSMOM signal. Requires `Config` to have been called first to
        populate `EPO.vol`, `EPO.correl`, `EPO.tsmom`, `EPO.chosen_w`.

        The shrinkage is the weight on the *identity*, so
        `Sigma_w = vol @ ((1-w)C + wI) @ vol`.

        Args:
            long_only (bool): Constrain the book to `w >= 0, sum(w) == 1` and
                solve the objective under those constraints. False returns the
                closed-form unconstrained maximiser `Sigma^-1 @ signal / gamma`,
                whose gross exposure is whatever the covariance implies and
                which is generally close to market neutral. Defaults to True.

        Returns:
            np.ndarray: Portfolio weights, one per ticker, in the same order
                as `EPO.correl[self.date]`'s columns. Non-negative and summing
                to one under `long_only`; otherwise unnormalised.

        Raises:
            RuntimeError: If `Config` has not been called. Checked explicitly
                because the bare AttributeError names `vol`, which does not
                point at the missing step.
            KeyError: If `self.date` is not one of the dates `Config` covered.
            ValueError: If the long-only solve does not converge.

        Example:
            >>> EPO.Config()
            >>> w = EPO('2020-03-31').portfolio_weight()
            >>> float(w.sum()), float(w.min())
            (1.0, 0.0)
        """
        if not getattr(EPO,'_configured',False):
            raise RuntimeError('EPO.Config() must be called before '
                               'portfolio_weight(); it precomputes the '
                               'per-date vol, correlation and TSMOM inputs')
        correl: np.ndarray = EPO.correl[self.date].to_numpy()
        vol: np.ndarray = np.diag(EPO.vol[self.date].to_numpy())
        signal: np.ndarray = EPO.tsmom[self.date].to_numpy()
        n_stock: int = correl.shape[0]
        w: float = EPO.chosen_w[self.date]

        shrink_correl: np.ndarray = correl*epo_theta+(1-epo_theta
                                                      )*np.identity(n_stock)
        epo_correl: np.ndarray = (1-w)*shrink_correl+w*np.identity(n_stock)
        epo_Sigma: np.ndarray = vol@epo_correl@vol
        if not long_only:
            return np.linalg.solve(epo_Sigma,signal)/risk_aversion
        return _long_only_weight(epo_Sigma,signal,risk_aversion)