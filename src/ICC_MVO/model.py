"""Bielstein & Hanauer (2019): mean-variance optimisation on implied cost of capital.

Bielstein, P. and Hanauer, M. X. (2019), "Mean-variance optimization using
forward-looking return estimates", *Review of Quantitative Finance and
Accounting* 52, 815-840. Their argument is that mean-variance optimisation
fails on its inputs, above all the expected returns, and that an implied cost
of capital (Gebhardt, Lee & Swaminathan 2001) is a better expected return than
a historical average. The construction, at each formation date:

1. The GLS implied cost of capital per stock (`icc_adjusted.icc_gls`).
2. Plus 12-1 momentum standardised onto the ICC's cross-sectional dispersion,
   both winsorised at 1%/99%, minus the risk-free rate
   (`icc_adjusted.expected_excess_return`).
3. Ledoit-Wolf (2004) constant-correlation covariance over 60 months of
   returns (`volatility.ledoit_wolf_cc`).
4. Maximum Sharpe ratio, long-only and fully invested, at most 5% in any name,
   weights under 0.01% set to zero (`utils.max_sharpe_weight`).

In this paper it is the point-estimate benchmark of the ex post table: the B&H
construction run on this repo's own panels.

**Lookahead: a benchmark, not a strategy.** B&H read the explicit earnings
years from IBES consensus forecasts, which this repo does not have. It reads the
*realised* EPS instead -- the `constant.icc_explicit_years` (11) four-quarter
blocks after the formation date's accounting cutoff -- so the book formed at `t`
has seen as much future earnings as the panel holds, the same window
`proposed_ex_post` elicits its moments from. That makes it an ICC under perfect
earnings foresight, the counterpart of the proposed model's ex post rows and
never a tradeable result. The panel ends in 2026, so from the 2015-09-30
formation date on, the later years run past it and are extrapolated
(`inputs.complete_eps`); how many years each name had realised is kept in
`Icc_Mvo.inputs['n_oracle_years']`.

Every other substitute for B&H's data -- dividends from adjusted closes, the
industry ROE over every SEC filer's firm-years grouped by SIC code, the
Fama-French risk-free rate -- is listed in `src/ICC_MVO/README.md`.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from constant import (
    accounting_path, monthly_price_path, daily_price_path, splits_path,
    industry_path, industry_roe_pool_path, icc_cov_month, icc_weight_cap,
    icc_weight_floor, risk_aversion,
)
from ..panel import read_csv_file, book_equity
from ..Proposed_Model.data_generator import _adjusted_shares, accounting_cutoff
from ..Empirical_Analysis.backtest.data import monthly_return, risk_free
from .icc_adjusted import (
    gls_path, icc_gls, expected_excess_return, MONTHS_PER_YEAR,
)
from .volatility import ledoit_wolf_cc
from .inputs import (
    realised_eps, trailing_eps, complete_eps, dividend_events,
    monthly_dividends, trailing_dps, payout_ratio, annual_roe,
    industry_target_roe, read_industry, panel_roe_pool, read_roe_pool,
    momentum, return_window,
)
from .utils import max_sharpe_weight, quadratic_utility_weight, \
    ICC_MVO_OBJECTIVES, _at


class Icc_Mvo:
    """The ICC-MVO book at one formation date.

    `Config()` builds the panels once per process, as `RIM_PortOp`, `EPO` and
    `PPP` do. An instance assembles one date's cross-section in its
    constructor -- cheap, a few hundred root solves -- and `weight()` solves
    the portfolio.

    Attributes:
        date (pd.Timestamp): Formation date.
        cutoff (pd.Timestamp): `accounting_cutoff(date)`, the last quarter the
            accounting inputs read. The realised EPS blocks start after it.
        inputs (pd.DataFrame): Per candidate ticker, dropped ones included:
            'price', 'b0' (book equity per share), 'e0' (trailing EPS), 'dps',
            'payout', 'roe_target', 'roe_source' ('ff48' or 'all'),
            'fy1'..'fy11', 'n_oracle_years', 'icc' and 'momentum'.
        dropped (dict[str, list[str]]): The tickers each screen removed, each
            ticker under the first screen it failed.
        tickers (pd.Index): The names that pass every screen.
        rf (float): Annual risk-free rate taken off every mean.
        shrinkage (float | None): Ledoit-Wolf intensity, once `covariance` has
            run.
    """

    @classmethod
    def Config (cls, force: bool = False, industry_file: Path = industry_path,
                pool_file: Path|None = industry_roe_pool_path)-> None:
        """Build the class-level panels. Idempotent unless `force`.

        Everything per share is on today's split basis: EPS, book equity and
        total assets divide by `adj_shares`, the close and the recovered
        dividends are split-adjusted by Yahoo, and the ICC is solved on all of
        them together. See `src.Data.split_adjust`.

        Sets `roe_pool`, the firm-years the industry median is taken over, and
        `roe_pool_source`: 'sec' for every SEC filer (`Data.industry_pool`),
        'panel' for the S&P 500 panel's own firm-years.

        Args:
            force (bool): Rebuild even if already built -- after `Data.run` or
                `Data.industry` has rewritten a file mid-process, or to switch
                `pool_file`. Defaults to False.
            industry_file (Path): The ticker -> FF48 table. Defaults to
                `constant.industry_path`; without it every firm fades to the
                all-firm median ROE, with a warning.
            pool_file (Path | None): The SEC filer pool. Defaults to
                `constant.industry_roe_pool_path`; if it is missing the panel
                stands in, with a warning. None asks for the panel pool
                outright, the earlier specification, as a robustness check.
        """
        if getattr(cls, '_configured', False) and not force:
            return
        acc: pd.DataFrame = read_csv_file(accounting_path)
        shares: pd.Series = _adjusted_shares(acc, splits_path)

        def wide (values: pd.Series)-> pd.DataFrame:
            # A zero share count turns into inf; NaN is what every screen reads.
            return (values.replace([np.inf, -np.inf], np.nan)
                    .unstack(level='ticker').sort_index())

        equity: pd.Series = book_equity(acc)
        cls.eps_q: pd.DataFrame = wide(acc['net_income']/shares)
        cls.bps_q: pd.DataFrame = wide(equity/shares)
        cls.aps_q: pd.DataFrame = wide(acc['total_assets']/shares)
        cls.close: pd.DataFrame = (read_csv_file(monthly_price_path)['close']
                                   .unstack(level='ticker').sort_index())
        cls.returns: pd.DataFrame = monthly_return()
        cls.rf_monthly: pd.Series = risk_free()
        cls.dividends: pd.DataFrame = monthly_dividends(
            dividend_events(read_csv_file(daily_price_path)))
        cls.industry: pd.Series = read_industry(industry_file)
        pool: pd.DataFrame|None = (read_roe_pool(pool_file)
                                   if pool_file is not None else None)
        cls.roe_pool_source: str = 'panel' if pool is None else 'sec'
        cls.roe_pool: pd.DataFrame = (
            pool if pool is not None
            else panel_roe_pool(annual_roe(wide(acc['net_income']),
                                           wide(equity)), cls.industry))
        cls._configured: bool = True

    def __init__ (self, date: str|pd.Timestamp, tickers: list[str])-> None:
        """Assemble the cross-section at `date`: inputs, ICC, momentum, screens.

        Args:
            date (str | pd.Timestamp): Formation date, a month end.
            tickers (list[str]): Candidate universe, typically
                `Empirical_Analysis.backtest.data.universe()[date]`.

        Raises:
            ValueError: If the ROE panel does not yet hold the industry median's
                minimum number of years at `date`.
        """
        Icc_Mvo.Config()
        cls = Icc_Mvo
        self.date: pd.Timestamp = pd.Timestamp(date)
        self.cutoff: pd.Timestamp = accounting_cutoff(self.date)
        names: pd.Index = pd.Index(list(dict.fromkeys(tickers)), name='ticker')

        price: pd.Series = _at(cls.close, self.date, names)
        b0: pd.Series = _at(cls.bps_q, self.cutoff, names)
        assets: pd.Series = _at(cls.aps_q, self.cutoff, names)
        eps_q: pd.DataFrame = cls.eps_q.reindex(columns=names)
        e0: pd.Series = trailing_eps(eps_q, self.cutoff)
        # Over the same four quarters as E0, so the payout ratio sets a year's
        # distributions against that year's earnings.
        dps: pd.Series = trailing_dps(cls.dividends, self.cutoff, names)
        payout: pd.Series = payout_ratio(dps, e0, assets)

        # The last calendar year all four of whose quarters are public.
        last_year: int = (self.cutoff.year if self.cutoff.month == 12
                          else self.cutoff.year-1)
        per_industry, overall = industry_target_roe(cls.roe_pool, last_year)
        target: pd.Series = cls.industry.reindex(names).map(per_industry)
        source: pd.Series = pd.Series(np.where(target.notna(), 'ff48', 'all'),
                                      index=names)
        target = target.fillna(overall)

        eps, n_oracle = complete_eps(realised_eps(eps_q, self.cutoff), e0,
                                     (1.0-payout)*target)
        roe, book = gls_path(eps.to_numpy(), b0.to_numpy(), payout.to_numpy(),
                             target.to_numpy())
        icc: pd.Series = pd.Series(icc_gls(price.to_numpy(), roe, book),
                                   index=names)
        mom: pd.Series = momentum(cls.returns, self.date, names)
        window: pd.DataFrame = return_window(cls.returns, self.date, names,
                                             icc_cov_month)
        complete: pd.Series = window.notna().all() & (window.std() > 0)

        self.inputs: pd.DataFrame = pd.DataFrame({
            'price': price, 'b0': b0, 'e0': e0, 'dps': dps, 'payout': payout,
            'roe_target': target, 'roe_source': source, **eps,
            'n_oracle_years': n_oracle, 'icc': icc, 'momentum': mom})

        screens: list[tuple[str, pd.Series]] = [
            ('no price', ~(price > 0)),
            ('book equity <= 0', ~(b0 > 0)),
            ('no earnings', eps.isna().any(axis=1)),
            ('no payout ratio', payout.isna()),
            ('no ICC root', icc.isna()),
            ('no 12-1 momentum', mom.isna()),
            (f'under {icc_cov_month} months of returns', ~complete)]
        keep: pd.Series = pd.Series(True, index=names)
        self.dropped: dict[str, list[str]] = {}
        for reason, fails in screens:
            hit: pd.Series = keep & fails
            if hit.any():
                self.dropped[reason] = names[hit.to_numpy()].tolist()
            keep &= ~fails
        self.tickers: pd.Index = names[keep.to_numpy()]

        # The Fama-French one-month bill rate of the formation month,
        # compounded to a year to sit against the annual ICC.
        rf_month: float = float(cls.rf_monthly.loc[:self.date].iloc[-1])
        self.rf: float = (1.0+rf_month)**MONTHS_PER_YEAR-1.0
        self.window: pd.DataFrame = window.loc[:, self.tickers]
        self.shrinkage: float|None = None

    def expected_return (self)-> pd.Series:
        """B&H's expected monthly excess return on the names that passed.

        Returns:
            pd.Series: Indexed by ticker.

        Raises:
            ValueError: If fewer than two names passed, which leaves no
                cross-section to winsorise or standardise.
        """
        if len(self.tickers) < 2:
            raise ValueError(f'{self.date.date()}: {len(self.tickers)} name(s) '
                             f'passed the screens; dropped {self.dropped}')
        passed: pd.DataFrame = self.inputs.loc[self.tickers]
        return expected_excess_return(passed['icc'], passed['momentum'],
                                      self.rf)

    def covariance (self)-> pd.DataFrame:
        """Ledoit-Wolf constant-correlation covariance of the names that passed.

        Returns:
            pd.DataFrame: Monthly covariance, ticker by ticker. The shrinkage
                intensity is left on `self.shrinkage`.
        """
        sigma, self.shrinkage = ledoit_wolf_cc(self.window.to_numpy())
        return pd.DataFrame(sigma, index=self.tickers, columns=self.tickers)

    def weight (self, cap: float = icc_weight_cap, long_only: bool = True,
                min_weight: float = icc_weight_floor,
                objective: str = 'max_sharpe',
                risk_aversion: float = risk_aversion)-> pd.Series:
        """B&H's book under a chosen objective. See `max_sharpe_weight`,
        `quadratic_utility_weight`.

        Args:
            cap (float): Largest weight on one name. Defaults to
                `constant.icc_weight_cap`.
            long_only (bool): Defaults to True, B&H's book.
            min_weight (float): Dust threshold. Defaults to
                `constant.icc_weight_floor`.
            objective (str): `'max_sharpe'` -- B&H's own rule -- or
                `'quadratic_utility'`, maximum mean-variance utility on the
                same `mu`/`sigma`: this repo's own comparison point, not part
                of B&H's specification. Defaults to `'max_sharpe'`.
            risk_aversion (float): The mean-variance risk-aversion
                coefficient, used only under `objective='quadratic_utility'`.
                Defaults to `constant.risk_aversion`.

        Returns:
            pd.Series: One weight per name that passed the screens, summing to
                one; the ones the optimiser does not hold are exactly 0.

        Raises:
            ValueError: If `objective` is not one of `ICC_MVO_OBJECTIVES`, or
                whatever the chosen solver raises.
        """
        if objective not in ICC_MVO_OBJECTIVES:
            raise ValueError(f'objective must be one of {ICC_MVO_OBJECTIVES}; '
                             f'got {objective!r}')
        mu: pd.Series = self.expected_return()
        sigma: pd.DataFrame = self.covariance()
        if objective == 'max_sharpe':
            weight: np.ndarray = max_sharpe_weight(
                mu.to_numpy(), sigma.to_numpy(), cap=cap,
                min_weight=min_weight, long_only=long_only)
        else:
            weight = quadratic_utility_weight(
                mu.to_numpy(), sigma.to_numpy(), risk_aversion=risk_aversion,
                cap=cap, min_weight=min_weight, long_only=long_only)
        return pd.Series(weight, index=mu.index, name='weight')
