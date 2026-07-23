import pandas as pd
import numpy as np
from constant import (
    testing_period, n_month, n_quarter, risk_aversion
)
from .input_generator import generator
from scipy.optimize import minimize
from .utils import ppp_op, ppp_weight

class PPP:
    def __init__(self) -> None:
        self.theta: dict[str, np.ndarray] = {}
        self.weight: dict[str, np.ndarray] = {}
        pass
    @classmethod
    def Config (cls):
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
        
        
        
        
        
        