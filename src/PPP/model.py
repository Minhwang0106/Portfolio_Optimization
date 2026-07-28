import pandas as pd
import numpy as np
from constant import (
    testing_period, n_month, n_quarter, risk_aversion
)
from .input_generator import generator
from scipy.optimize import minimize
from .utils import ppp_op, ppp_weight

class PPP:
    """Parametric Portfolio Policy (Brandt, Santa-Clara & Valkanov 2009).

    Rather than forecasting each stock's moments, the portfolio weight is
    parameterised directly as a function of firm characteristics::

        w_it = w_bar_it + (1/N_t) * theta' * x_hat_it

    where `w_bar` is an equal-weighted benchmark and `x_hat` are the
    cross-sectionally standardized characteristics -- momentum (`mom`),
    log book-to-market (`btm`) and log size (`me`). The three coefficients in
    `theta` are fitted once per formation date by maximising realised CRRA
    utility over a trailing window.

    `PPP.Config()` builds the class-level panels; `__init__` calls it if it has
    not run, so `PPP()` works standalone. Instantiate once, then call
    `portfolio_weight` per formation date.

    Attributes:
        characteristics (pd.DataFrame): Class-level. Indexed (ticker, date) with
            columns ['mom', 'btm', 'me'], unstandardized. Set by `Config`.
        return_df (pd.DataFrame): Class-level. Monthly simple returns, indexed
            by date with one column per ticker. Set by `Config`.
        theta (dict[str, np.ndarray]): Fitted coefficients keyed by
            stringified formation date.
        weight (dict[str, np.ndarray]): Portfolio weights keyed by
            stringified formation date.
        ticker (list[str]): The universe that survived the last `find_theta`
            call -- narrower than the argument passed in.

    Example:
        >>> PPP.Config()
        >>> model = PPP()
        >>> model.portfolio_weight('2020-03-31', ['AAPL', 'MSFT', 'JPM'])
        array([[0.41527..., 0.36118..., 0.22354...]])
    """
    def __init__(self) -> None:
        """Create an unfitted model with empty per-date result stores.

        Takes no arguments: the data panels live on the class. `Config` is
        called here if it has not run, so a bare `PPP()` works standalone;
        call it explicitly only to force a rebuild.

        Example:
            >>> model = PPP()
            >>> model.theta, model.weight
            ({}, {})
        """
        PPP.Config()
        self.theta: dict[str, np.ndarray] = {}
        self.weight: dict[str, np.ndarray] = {}
    @classmethod
    def Config (cls, force: bool = False):
        """Build the class-level characteristic and return panels. Idempotent.

        Returns immediately if the panels are already built, so repeat calls
        cost nothing and cannot reload the data a second time. `force` rebuilds
        anyway, which is what you want after `Data.run` rewrites a CSV (together
        with `panel.clear_panel_cache`, since the parse is cached too).

        Reads the raw inputs via `generator()` and derives:

        * `mom` -- 11-month momentum, `P(t-2)/P(t-13) - 1`, skipping the most
          recent month to avoid the short-term reversal effect.
        * `btm` -- `log(1 + book_value / market_cap)`.
        * `me` -- `log(market_cap)`.

        Characteristics are stored *raw*, not standardized: the z-score has to
        be computed on each formation date's investable universe, which is only
        known inside `find_theta`. Infinities and NaNs from the logs
        (`market_cap == 0`, or `BE/ME <= -1`) are normalised to NaN so a single
        bad name cannot poison a whole date's cross-sectional moments.

        Args:
            force (bool): Rebuild even if already configured. Defaults to False.

        Returns:
            None. Populates `PPP.characteristics` and `PPP.return_df`.

        Example:
            >>> PPP.Config()
            >>> PPP.characteristics.loc['A'].tail(2)
                             mom       btm         me
            date
            2025-10-31  0.183044  0.121993  24.263241
            2025-11-30  0.201776  0.119438  24.281905
            >>> PPP.return_df.iloc[-1, :3]
            ticker
            A      -0.014327
            AAPL    0.052881
            ABBV    0.007194
            Name: 2025-11-30 00:00:00, dtype: float64
        """
        if getattr(cls,'_configured',False) and not force:
            return
        input_df: pd.DataFrame = generator()
        col: list[str] = ['mom','btm','me']
        col_val: list[pd.Series] = []
        
        #mom
        col_val.append(input_df['adj_close'].groupby(
            level='ticker').shift(2)/input_df['adj_close'].groupby(
            level='ticker').shift(13)-1)
        #btm
        btm: pd.Series = input_df['book_value']/input_df['market_cap']
        col_val.append(btm.apply(lambda x: np.log(1+x)))
        
        #me
        col_val.append(input_df['market_cap'].apply(lambda x: np.log(x)))
        
        df:pd.DataFrame = pd.concat(col_val,keys=col,axis=1).sort_index()
        # log() can emit +/-inf or NaN (market_cap == 0, or BE/ME <= -1);
        # treat those as missing so they don't poison a whole date's moments.
        # Kept raw here: standardization must happen on the *investable*
        # universe of each formation date, which is only known in find_theta
        # (see _standardize).
        cls.characteristics: pd.DataFrame = df.replace([np.inf,-np.inf],np.nan)
        cls.return_df: pd.DataFrame = input_df['adj_close'
                                               ].unstack(level=0).pct_change()
        cls._configured: bool = True
    @staticmethod
    def _standardize (char: np.ndarray):
        """Cross-sectional standardization (Brandt-Santa-Clara-Valkanov 2009;
        PPP_2009.md sec.3): z-score each characteristic across the ``N_t``
        stocks present at a formation date -- per-date, never over time, and
        only over the stocks that survived the section-2 filters. This is what
        makes ``sum(x_hat) == 0``, so the tilts cancel and the weights sum to
        one. Accepts either a ``(n_t, n_char)`` cross-section or a stacked
        ``(n_period, n_t, n_char)`` panel; the stock axis is always ``-2``."""
        mean: np.ndarray = char.mean(axis=-2, keepdims=True)
        std: np.ndarray = char.std(axis=-2, keepdims=True, ddof=1)
        return (char - mean)/std
    def find_theta (self, date:str|pd.Timestamp,ticker:list[str],
                    n_month: int = 60):
        """Fit the policy coefficients on the window ending just before `date`.

        Screens `ticker` down to the names with a complete record of all three
        characteristics over the whole estimation window, standardizes them
        cross-sectionally per date, then maximises realised CRRA utility over
        the window by minimising `ppp_op`.

        The window is strictly backward-looking: characteristics are read over
        `[date - (n_month+1) months, date - 1 month]` and paired with the
        returns they earn over `[date - n_month months, date]`, so each
        cross-section is matched with the *next* month's return.

        Args:
            date (str | pd.Timestamp): Formation date. Strings are parsed with
                `pd.Timestamp`; should be a month-end.
            ticker (list[str]): Candidate universe (e.g. the index members
                applicable at `date`). Narrowed by the completeness screen --
                the surviving list is stored on `self.ticker`.
            n_month (int): Length of the estimation window in months. Defaults
                to 60.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                * theta, shape `(3,)` -- coefficients on mom, btm, me in the
                  column order of `PPP.characteristics`.
                * benchmark, shape `(1, n_t)` -- equal weights over the
                  surviving universe.

            Also records theta in `self.theta[str(date)]` and the surviving
            universe in `self.ticker`.

        Note:
            `theta` is seeded with `np.random.random(3)`, so results vary run to
            run unless the global numpy seed is fixed.

        Example:
            >>> PPP.Config()
            >>> model = PPP()
            >>> theta, benchmark = model.find_theta('2020-03-31',
            ...                                     ['AAPL', 'MSFT', 'JPM'])
            >>> theta
            array([ 0.42317..., -0.18402...,  0.07655...])
            >>> benchmark
            array([[0.33333333, 0.33333333, 0.33333333]])
            >>> model.ticker
            ['AAPL', 'JPM', 'MSFT']
        """
        if isinstance(date,str):
            date = pd.Timestamp(date)
        idx = pd.IndexSlice
        
        date_m1: pd.Timestamp = date - pd.offsets.MonthEnd(1)
        date_0: pd.Timestamp = date - pd.offsets.MonthEnd(n_month+1)
        date_1: pd.Timestamp = date - pd.offsets.MonthEnd(n_month)
        
        input: pd.DataFrame = PPP.characteristics.loc[idx[ticker,date_0:date_m1],:]
        # Keep only tickers observed on *every* characteristic at *every* date
        # of the window. Unstacking first matters: it turns a ticker that is
        # simply absent on some date into an explicit NaN, which a check on the
        # stacked frame would never see. Screening all three characteristics
        # rather than btm alone also catches a missing mom (a ticker without the
        # full 13-month lookback), which btm does not imply.
        # Transposed so the (characteristic, ticker) pairs are rows and dropna
        # can work on its default axis: a pair clean at every date survives, so
        # a fully observed ticker is left once per characteristic -- hence the
        # count against the characteristic total below.
        kept: pd.Index = input.unstack(level='ticker').T.dropna().index
        n_kept: pd.Series = kept.get_level_values('ticker').value_counts()
        # sorted(), not list(set(...)): set iteration order varies between
        # processes, and this order is what aligns char_arr with r_arr.
        ticker = sorted(set(n_kept.index[n_kept == input.shape[1]]
                            ).intersection(ticker))

        n_t: int = len(ticker)
        benchmark_port: np.ndarray = np.ones(shape=(1,n_t))/n_t
        theta: np.ndarray = np.random.random((3))

        # Build the panel date-major and with an explicit column order rather
        # than reshaping the stacked frame: PPP.characteristics is indexed
        # (ticker, date), so a plain reshape would interleave one ticker's time
        # series into what the optimizer reads as a single cross-section.
        char_arr: np.ndarray = self._standardize(np.stack(
            [input[c].unstack(level='ticker').reindex(columns=ticker).to_numpy()
             for c in PPP.characteristics.columns], axis=-1))
        r_arr: np.ndarray = self.return_df.loc[date_1:date,ticker].to_numpy()

        res = minimize(ppp_op, theta, args=(benchmark_port,char_arr,r_arr))
        self.theta[str(date)] = res.x
        self.ticker = ticker
        return res.x, benchmark_port
    def portfolio_weight (self, date:str|pd.Timestamp, ticker:list[str],
                          n_month: int = 60):
        """Fit theta and apply it to `date`'s cross-section to get weights.

        Calls `find_theta` for the coefficients, then standardizes the
        characteristics observed *on* `date` (over the surviving universe only)
        and tilts the equal-weighted benchmark by them.

        Args:
            date (str | pd.Timestamp): Formation date; the weights are the ones
                to hold from `date` into the following month.
            ticker (list[str]): Candidate universe, before the completeness
                screen.
            n_month (int): Estimation window length in months, passed straight
                through to `find_theta`. Defaults to 60.

        Returns:
            np.ndarray: Weights of shape `(1, n_t)` summing to one, ordered to
                match `self.ticker` (sorted alphabetically, and shorter than
                the `ticker` argument whenever names were screened out).
                Individual weights may be negative -- the policy is
                unconstrained, so shorts are allowed. Also stored in
                `self.weight[str(date)]`.

        Example:
            >>> PPP.Config()
            >>> model = PPP()
            >>> w = model.portfolio_weight('2020-03-31', ['AAPL', 'MSFT', 'JPM'])
            >>> w
            array([[0.41527..., 0.22354..., 0.36118...]])
            >>> model.ticker
            ['AAPL', 'JPM', 'MSFT']
            >>> w.sum()
            np.float64(1.0)
        """
        if isinstance(date,str):
            date = pd.Timestamp(date)
        # `ticker` is the raw applicable universe; find_theta narrows it to the
        # names with usable characteristics and stores the result on self, so
        # read self.ticker afterwards rather than the argument.
        theta, benchmark = self.find_theta(date,ticker,n_month)
        char: np.ndarray = self._standardize(PPP.characteristics.xs(
            date, level='date').reindex(self.ticker).to_numpy())
        weight: np.ndarray = ppp_weight(theta,benchmark,char)
        self.weight[str(date)] = weight
        return weight
        
        
        
        
        
        