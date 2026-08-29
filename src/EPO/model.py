import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from constant import (testing_period,
                       risk_n_day,
                       risk_day_per_name,
                       risk_n_month,
                       epo_shrinkage,
                       epo_shrinkage_grid,
                       epo_w_min_periods,
                       epo_theta,
                       risk_aversion)
from .utils import init_worker, compute_date
from .functions import _long_only_weight


class EPO:
    """Enhanced Portfolio Optimization.

    Call `EPO.Config()` once to precompute the per-date risk inputs, then
    instantiate per formation date. Unlike `PPP` and `RIM_PortOp`, `__init__`
    does *not* configure on your behalf: `Config` runs a process pool over every
    date in `testing_period`, which is far too expensive to trigger as a side
    effect of constructing an object.

    `Config(long_only=...)` fixes which book the class holds, and it holds one
    at a time. The long-only default solves `x >= 0, sum(x) == 1` on the
    simplex; `long_only=False` returns the paper's own unconstrained closed
    form, eq. (16). Everything downstream -- the candidate books, the shrinkage
    selection, `portfolio_weight` -- follows that one setting, because a `w`
    ranked on one book is close to uninformative about the other and a run that
    held both at once could not say which EPO its numbers describe.

    Attributes:
        vol, correl, signal (dict): Class-level, keyed by stringified date. Set
            by `Config`. `correl` is the raw correlation estimate; the 5%
            pre-shrink of `constant.epo_theta` is applied downstream of it.
        candidate (dict[str, dict[float, pd.Series]]): Class-level. The weight
            per date per candidate shrinkage, in whichever book `Config` was
            asked for. Set by `Config`.
        realized (dict[str, pd.Series | None]): Class-level. Next month's
            realised excess return per date, keyed like `candidate`. Set by
            `Config`.
        chosen_w (dict[str, float]): Class-level. The shrinkage `_select_w`
            picked for each date, scored on that same book -- see
            `constant.epo_shrinkage`. Set by `Config`.
        long_only (bool): Class-level. Which book the above describe, and which
            one `portfolio_weight` solves. Set by `Config`.
        date (str): This instance's formation date.
    """
    def __init__(self, date:str) -> None:
        self.date = date

    @classmethod
    def Config(cls, n_day=risk_n_day, day_per_name=risk_day_per_name,
                      n_month=risk_n_month, long_only: bool = True,
                      n_workers: int | None = None, force: bool = False):
        """Precompute per-date risk inputs, w-selection inputs, and chosen w.

        Runs `compute_date` (from `utils`) over every date in `testing_period`,
        in parallel via a process pool (unless `n_workers` <= 1), and stores
        the results as class-level attributes `EPO.vol`, `EPO.correl`,
        `EPO.signal`, `EPO.candidate`, `EPO.realized` so that `portfolio_weight`
        can look them up per instance. Then runs `_select_w` once over all dates
        to populate `EPO.chosen_w` -- see `constant.epo_shrinkage`.

        `long_only` decides which book all of that describes, and only that one
        is built. Solving both would mean selecting two shrinkages and holding
        one, and the unused half is not a free by-product: the long-only book
        costs one SLSQP solve per date per grid point against one
        `np.linalg.solve` for the closed form.

        Idempotent: returns immediately if already configured *in the same
        mode*, so a repeat call cannot silently redo the whole parallel sweep,
        while a call that asks for the other book rebuilds rather than handing
        back one that answers a different question. Pass `force=True` to rebuild
        after changing any other parameter.

        Args:
            n_day (int): Floor for the trailing daily covariance window.
            day_per_name (int): Daily observations required per name; the window
                is `max(n_day, day_per_name * n_t)`. See `functions.cov_window`.
            n_month (int): Trailing window (months) for the momentum signal.
            long_only (bool): Solve `x >= 0, sum(x) == 1` on the simplex rather
                than the unconstrained closed form, and select `w` against that
                book. Defaults to True, matching `PPP` and `RIM_PortOp` and the
                run-level `long_only` of `Empirical_Analysis.run`. False gives
                eq. (16) as the paper states it -- see `portfolio_weight`.
            n_workers (int | None): Number of worker processes; defaults to
                cpu_count - 1. Falls back to serial execution if <= 1.
            force (bool): Recompute even if already configured in this mode.
                Needed whenever any of the window parameters differ from the
                first call, since otherwise they are ignored. Defaults to False.

        Returns:
            None. Populates `EPO.vol`, `EPO.correl`, `EPO.signal`,
            `EPO.candidate`, `EPO.realized` (each a dict keyed by stringified
            date), `EPO.chosen_w` (a dict of float keyed the same way) and
            `EPO.long_only`.

        Example:
            >>> EPO.Config()
            >>> '2020-03-31' in EPO.vol
            True
        """
        # The guard weighs *what* was configured, not merely that something
        # was. A second call in the same process asking for the other book has
        # to rebuild: the cached candidates, the cached `chosen_w` and the
        # solve in `portfolio_weight` are all one book's, and handing back the
        # wrong one would silently answer a different question.
        if (getattr(cls,'_configured',False) and not force
                and getattr(cls,'long_only',None) == long_only):
            return
        std_dict: dict[str, pd.Series] = {}
        correl_dict: dict[str, pd.DataFrame] = {}
        signal_dict: dict[str, pd.Series] = {}
        candidate_dict: dict[str, dict[float, pd.Series]] = {}
        realized_dict: dict[str, pd.Series|None] = {}

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

        if n_workers <= 1:
            init_worker(n_day, day_per_name, n_month, long_only)
            results = (compute_date(date) for date in testing_period)
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker,
                                      initargs=(n_day, day_per_name, n_month,
                                                long_only)) as ex:
                results = list(ex.map(compute_date, testing_period))

        for result in results:
            if result is None:
                continue
            key, std, correl, signal, candidates, realized = result
            std_dict[key] = std
            correl_dict[key] = correl
            signal_dict[key] = signal
            candidate_dict[key] = candidates
            realized_dict[key] = realized

        cls.vol = std_dict
        cls.correl = correl_dict
        cls.signal = signal_dict
        cls.candidate = candidate_dict
        cls.realized = realized_dict
        cls.chosen_w = cls._select_w(candidate_dict, realized_dict)
        cls.long_only: bool = long_only
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
        Sharpe ratio its weight earned over every *earlier* formation date. A
        date's own return never contributes to its own score, and a date whose
        grid score has no history yet falls back to `constant.epo_shrinkage`.

        Agnostic about which book it is ranking, and it ranks whichever one
        `Config` built. Scoring the closed form and applying the winner to the
        simplex solve would not be a shortcut but a different question: the two
        books correlate ~0 at every `w` below 0.75, so a `w` ranked on one is
        close to uninformative about the other.

        Args:
            candidate (dict[str, dict[float, pd.Series]]): Per date, the weight
                at each grid point -- `compute_date`'s `candidates`, in the book
                `init_worker`'s `long_only` selected.
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

    def portfolio_weight (self):
        """Compute Enhanced Portfolio Optimization (EPO) weights for this instance's date.

        Applies the risk model's 5% pre-shrink to the correlation matrix (via
        `epo_theta`), builds a covariance matrix from vol and correlation,
        further blends it toward identity by the shrinkage `_select_w` picked
        out-of-sample for this date (see `constant.epo_shrinkage`), then
        maximises `w'mu - gamma/2 w'Sigma w` over the `sigma * XSMOM` signal.
        Requires `Config` to have been called first to populate `EPO.vol`,
        `EPO.correl`, `EPO.signal` and `EPO.chosen_w`.

        Takes no constraint argument: which book this is was fixed by
        `Config(long_only=...)`, and `EPO.chosen_w` holds the shrinkage ranked
        on that same book. The two are not separable -- ranking `w` on the
        closed form and applying the winner to the simplex solve scores a
        portfolio that is close to uncorrelated with the one held.

        Under `EPO.long_only` the book is the maximiser of the same objective
        over `x >= 0, sum(x) == 1`; otherwise it is eq. (16) itself, the
        closed-form `Sigma_w^-1 @ signal / gamma`, whose gross exposure is
        whatever the covariance implies and which is generally close to market
        neutral. Eq. (16) is not a convenience form of EPO, it is EPO:
        Proposition 2 (p.13, proved p.36) derives it as the exact solution to a
        robust max-min problem over an ellipsoidal uncertainty region for
        expected returns, and that problem carries no sign or budget
        constraint. Adding one is this repo's doing, so that the EPO row
        compares like-for-like against `PPP` and `Proposed_Model`.

        The two shrinkages are different objects and compose in a fixed order:
        `epo_theta` belongs to the *risk model* and is applied first, `w` is
        EPO's own and is selected per date. The shrinkage is the weight on the
        *identity*, so `Sigma_w = vol @ ((1-w)C_tilde + wI) @ vol`.

        Returns:
            np.ndarray: Portfolio weights, one per ticker, in the same order as
                `EPO.correl[self.date]`'s columns. Non-negative and summing to
                one under `EPO.long_only`; otherwise unnormalised.

        Raises:
            RuntimeError: If `Config` has not been called -- checked explicitly
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
                               'per-date vol, correlation and signal inputs')
        correl: np.ndarray = EPO.correl[self.date].to_numpy()
        vol: np.ndarray = np.diag(EPO.vol[self.date].to_numpy())
        signal: np.ndarray = EPO.signal[self.date].to_numpy()
        n_stock: int = correl.shape[0]
        w: float = EPO.chosen_w[self.date]

        shrink_correl: np.ndarray = correl*epo_theta+(1-epo_theta
                                                      )*np.identity(n_stock)
        epo_correl: np.ndarray = (1-w)*shrink_correl+w*np.identity(n_stock)
        epo_Sigma: np.ndarray = vol@epo_correl@vol
        if not EPO.long_only:
            return np.linalg.solve(epo_Sigma,signal)/risk_aversion
        return _long_only_weight(epo_Sigma,signal,risk_aversion)