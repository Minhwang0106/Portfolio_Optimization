import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from .data_generator import (
    generator, take_training_data, accounting_cutoff
)
from pyvinecopulib import FitControlsBicop
from pyvinecopulib import BicopFamily
from .utils.dependence_structure import copula_structure
from .utils.simulation import sample_joint, CHARACTERISTICS
from .utils.sampling_distribution import (
    lag_moments, signal_cal, conditional_moments, sample_conditional
)
from .utils.theta import find_theta
from .utils.variable_tranformation import Ros_Trans, Ate_Trans, Ato_Trans
from constant import (
    n_quarter, n_quarter_ahead, min_char_obs, accounting_path,
    monthly_price_path
)
from .utils.elicitation import parameter_derived
from .utils import cost_of_fund as cof
from .utils.implied_return import rim_mapping, terminal_val, implied_return
from .utils.port_optimize import port_weight
class RIM_PortOp:
    """Residual-income portfolio optimisation over simulated characteristics.

    Call `RIM_PortOp.Config()` once to build the class-level panels, then
    instantiate per formation date. `__init__` calls `Config` itself if it has
    not run, so a bare `RIM_PortOp(date, ticker)` works standalone; call it
    explicitly only when you want non-default data paths, and do that before the
    first instance exists.

    Attributes:
        general_data (pd.DataFrame): Class-level. The characteristic panel,
            indexed (ticker, date). Set by `Config`.
        general_pb, general_bvps, general_rps, general_price (pd.DataFrame):
            Class-level. The RIM inputs, each unstacked wide. Set by `Config`.
        ticker_across_time (dict[str, list[str]]): Class-level record of which
            universe each formation date resolved to, for inspection across a
            backtest. Not authoritative -- an instance reads its own
            `self.ticker`, because two instances can share a date.
        ticker (list[str]): This instance's universe, narrowed by
            `take_training_data` to the names the panel actually carries.
    """
    bicop_controls = FitControlsBicop(selection_criterion='aic',
                            family_set=[BicopFamily.gaussian,#type: ignore
                                        BicopFamily.frank, #type: ignore
                                        BicopFamily.clayton]) #type: ignore

    ticker_across_time: dict[str, list[str]] = dict()
    # Indexed by CHARACTERISTICS, so the padding is load bearing: 'g' leads
    # that tuple and is modelled untransformed, while the other three are
    # generated on the scale data_generator put them on and have to come back.
    activate_inverse_func: list = [None, Ate_Trans.inverse,
                                   Ato_Trans.inverse, Ros_Trans.inverse]

    @classmethod
    def Config (cls, acc_path: Path = accounting_path,
                price_path: Path = monthly_price_path, force: bool = False):
        """Build the class-level panels. Idempotent.

        Returns immediately if the panels are already built, so calling it in a
        loop costs nothing and cannot reload the data a second time. `force`
        rebuilds anyway, which is what you want after `Data.run` rewrites a CSV
        (together with `panel.clear_panel_cache`, since the parse is cached too).

        Args:
            acc_path (Path): Accounting panel CSV. Defaults to
                `constant.accounting_path`.
            price_path (Path): Monthly price panel CSV. Defaults to
                `constant.monthly_price_path`.
            force (bool): Rebuild even if already configured. Defaults to False.

        Returns:
            None. Populates `general_data`, `general_pb`, `general_bvps`,
            `general_rps` and `general_price` on the class.

        Example:
            >>> RIM_PortOp.Config()
            >>> RIM_PortOp.general_data.columns.tolist()
            ['ros', 'ato', 'ate', 'g']
        """
        if getattr(cls,'_configured',False) and not force:
            return
        (cls.general_data, cls.general_pb, cls.general_bvps,
         cls.general_rps, cls.general_price) = generator(acc_path,price_path)
        cls._configured: bool = True
        cls.ff5_factor: pd.DataFrame = cof.generator()

    def __init__(self, date:str|pd.Timestamp,ticker:list[str],
                 min_obs:int=min_char_obs) -> None:
        """Slice the panels around one formation date.

        Args:
            date (str | pd.Timestamp): Formation date. Strings are parsed with
                `pd.Timestamp`.
            ticker (list[str]): Candidate universe. Narrowed to the names the
                panel carries and then to those with at least
                `constant.min_char_obs` quarters of complete characteristics;
                read the survivors from `self.ticker`.
            min_obs (int): Quarters of complete characteristics a ticker needs
                to survive. Defaults to `constant.min_char_obs`.

        Warns:
            RuntimeWarning: If any candidate is dropped for thin characteristics,
                naming it and its count.
        """
        RIM_PortOp.Config()
        if isinstance(date, str):
            self.date: pd.Timestamp = pd.Timestamp(date)
        else:
            self.date: pd.Timestamp = date
        cache: tuple[pd.DataFrame,pd.
                     DataFrame,list[str]] = take_training_data(
            RIM_PortOp.general_data,self.date,ticker)
        # The universe lives on the instance. It used to live only in the
        # class-level dict below, which is keyed by date alone -- so a second
        # instance for the same date overwrote the first's universe, and the
        # first then fitted copulas against tickers it never asked for. The
        # dict is still written for cross-date inspection, but nothing reads
        # it back.
        self.train_df, self.future_df, self.ticker = cache

        # Having rows is not the same as having *usable* rows. `Data.exclusion`
        # screens the raw accounting columns; the characteristics are those
        # columns after the transforms in `data_generator`, and a ratio outside
        # a transform's domain leaves NaN behind. `copula_structure` cannot fit
        # a ticker whose rows all carry one -- it raises 'no association
        # available', which took the whole formation date down at 2024-12-31,
        # where AVB, EXR and ESS have no complete quarter at all. Counted on all
        # four characteristics jointly, since that is what the inner tree needs.
        usable: pd.Series = self.train_df[list(CHARACTERISTICS)].notna().all(
            axis=1).groupby(level='ticker').sum()
        thin: list[str] = [t for t in self.ticker
                           if int(usable.get(t,0)) < min_obs]
        if thin:
            warnings.warn(
                f'{self.date.date()}: dropping {len(thin)} ticker(s) with fewer '
                f'than {min_obs} complete quarters of characteristics: '
                f'{ {t: int(usable.get(t,0)) for t in thin} }',
                RuntimeWarning)
            self.ticker = [t for t in self.ticker if t not in set(thin)]
            idx = pd.IndexSlice
            self.train_df = self.train_df.loc[idx[self.ticker,:],:]
            # Intersected first, because `take_training_data` narrows the
            # universe on the *training* window alone -- narrowing it on the
            # future window too would pick the universe partly on data the
            # formation date cannot see. So a name can be in `self.ticker` and
            # carry no future row at all (XEL at 2019-12-31), and slicing by the
            # full list raises KeyError and takes the whole formation date down.
            # Only reachable when `thin` is non-empty, which is why it survived
            # the single-date checks and only surfaced across a full backtest.
            in_future: set[str] = set(
                self.future_df.index.get_level_values('ticker'))
            self.future_df = self.future_df.loc[
                idx[[t for t in self.ticker if t in in_future],:],:]
        RIM_PortOp.ticker_across_time[str(self.date)] = self.ticker
    def depedence_structure (self):
        controls = RIM_PortOp.bicop_controls
        tickers: list[str] = self.ticker

        outer_df: pd.DataFrame = self.train_df.loc[tickers,'g'
                                                   ].unstack(level='ticker'
                                                             ).dropna()
        inner_df: pd.DataFrame = self.train_df.loc[tickers,CHARACTERISTICS]
        outer_copula: list[dict] = copula_structure(outer_df, controls)
        inner_copula: dict[str,list] = {}
        
        for ticker in tickers:
            temp = inner_df.xs(ticker, level='ticker').dropna()
            if not isinstance(temp,pd.DataFrame):
                raise ValueError('Inner temp should be pandas DataFrame')
            inner_copula[ticker] = copula_structure(temp,controls)
        return outer_copula, inner_copula
    
    def sampling (self, n_samples: int = 10000,n_lags:int=4,forward:bool=False,
                  common_theta:bool=True,seed:int|None=None):
        outer_copula, inner_copula = self.depedence_structure()
        # `sample_joint` already takes a seed; passing it through is what makes a
        # reported figure reproducible. Left None by default, so the unseeded
        # behaviour is unchanged unless a caller asks for repeatability.
        uniform_sample: dict[str,np.ndarray] = sample_joint(
            outer_copula,inner_copula,n_quarter_ahead+1,n_samples,seed
        )
        tickers: list[str] = list(uniform_sample.keys())
        large_traindata: pd.DataFrame = self.train_df.loc[tickers,CHARACTERISTICS]
        if forward:
            # A name can sit in `self.ticker` and carry no future row at all --
            # `take_training_data` picks the universe on the *training* window
            # alone, deliberately, so the formation date never selects on data
            # it cannot see. Slicing by the full list raises KeyError and takes
            # the whole date down (FITB's date, 2018-06-30, has one such name).
            # Those tickers fall back to their training moments rather than
            # being dropped, so the forward run's universe stays identical to
            # the baseline it exists to be compared against.
            in_future: set[str] = set(
                self.future_df.index.get_level_values('ticker'))
            eli_traindata: pd.DataFrame = self.future_df.loc[
                [t for t in tickers if t in in_future],CHARACTERISTICS]
        else:
            in_future = set(tickers)
            eli_traindata: pd.DataFrame = large_traindata.copy()
        sample: dict[str, np.ndarray] = dict()

        train_by_ticker: dict[str,pd.DataFrame] = {}
        for ticker in tickers:
            frame = large_traindata.xs(ticker,level='ticker')
            if not isinstance(frame,pd.DataFrame):
                raise ValueError(f"{ticker} in train data have only 1 point of data")
            train_by_ticker[ticker] = frame

        # One decay per characteristic for the whole universe. A training
        # window this short cannot identify a decay per ticker -- fitted that
        # way most columns land on 0 or 1, which says the likelihood is flat
        # between them, not that the characteristic is that persistent.
        # Pooling keeps the characteristics apart, which is where persistence
        # genuinely differs, and backs each estimate with the whole universe.
        pooled: np.ndarray|None = None
        if common_theta:
            pooled = np.asarray(find_theta(train_by_ticker,n_lags))

        for ticker in tickers:
            train_df: pd.DataFrame = train_by_ticker[ticker]
            eli_df = (eli_traindata.xs(ticker,level='ticker')
                      if ticker in in_future else train_df)
            ticker_uniform: np.ndarray = uniform_sample[ticker]
            T, N_s, d = ticker_uniform.shape
            if not isinstance(eli_df,pd.DataFrame):
                raise ValueError(f"{ticker} in elicitation data have only 1 point of data")
            mean_x, var_x = parameter_derived(eli_df)
            # An unfittable column falls back to its unconditional variance, so
            # that variance has to be usable itself. A characteristic that never
            # moves across the elicitation window has variance exactly 0 -- MAS's
            # `ate` over the 12 quarters after 2015-03-31 -- and substituting it
            # puts a 0 into `sample_conditional`, which raises and costs the whole
            # formation date for one ticker in 172. Only reachable when the
            # elicitation window is not the training window, i.e. under
            # `forward=True`; the training law is the only other unconditional
            # one available, and is non-degenerate wherever the column fitted.
            fallback_var: np.ndarray = np.asarray(var_x,dtype=np.float64)
            if forward:
                _, train_var = parameter_derived(train_df)
                fallback_var = np.where(fallback_var > 0, fallback_var,
                                        np.asarray(train_var,dtype=np.float64))
            correl: np.ndarray = lag_moments(train_df,n_lags).corr
            if pooled is not None:
                thetas: np.ndarray = pooled
            else:
                thetas = np.asarray(find_theta(train_df,n_lags))
            # Both loop invariant. Taken from conditional_moments rather than
            # divided out by hand so eq (18) keeps its guard: the pairwise
            # deletion in lag_moments can carry it outside (0, inf), and the
            # square root below would turn that into a silent NaN path.
            beta, cond_var = conditional_moments(correl,var_x,thetas)

            # find_theta reports a column it cannot fit as NaN -- too little
            # history, or an eq (18) infeasible right across [0, 1] -- and
            # conditional_moments carries that through. Such a column has no
            # usable signal, so it falls back to its unconditional law rather
            # than taking the whole ticker down with it. beta = 0 is what
            # makes that fallback: it leaves conditional_mean at mu_x. The
            # theta is replaced too, since a NaN weight would otherwise reach
            # the signal and 0*NaN is still NaN.
            #
            # A non-positive conditional variance is unfittable in the same
            # sense as a non-finite one and takes the same fallback, or
            # `sample_conditional` raises and the whole formation date is lost.
            # It is reachable whenever the variance and the lag correlation are
            # elicited from different windows -- i.e. under `forward=True`,
            # where eq (18) goes infeasible because the realised future is less
            # dispersed than the training correlation implies. `var_x` is the
            # unconditional law of whichever window was elicited, so it remains
            # the right thing to fall back to.
            unfit: np.ndarray = ~(np.isfinite(beta) & np.isfinite(cond_var)
                                  & (cond_var > 0) & (fallback_var > 0))
            if unfit.any():
                warnings.warn(
                    f'{ticker}: no conditional fit for '
                    f'{[c for c,u in zip(CHARACTERISTICS,unfit) if u]}; '
                    f'drawing them from the unconditional law',
                    RuntimeWarning)
                thetas = np.where(unfit,1.0,thetas)
                beta = np.where(unfit,0.0,beta)
                cond_var = np.where(unfit,fallback_var,cond_var)

            # Eq (10) reads realised x, so the first simulated quarter has a
            # signal of its own -- built from the last observed ones. Each (d,)
            # row is broadcast across simulations so the buffer holds a single
            # shape, and a gap is filled with the unconditional mean rather
            # than left to send NaN down every path.
            history: np.ndarray = train_df.to_numpy(dtype=np.float64)[-n_lags:]
            history = np.where(np.isnan(history),mean_x,history)
            real_val_list: list[np.ndarray] = [
                np.broadcast_to(row,(N_s,d)).copy() for row in history]

            for i in range(T):
                # Signal first, append after: period i never reads its own draw.
                signal: np.ndarray|np.float64 = signal_cal(
                    real_val_list,n_lags,thetas)
                real_val: np.ndarray = np.asarray(sample_conditional(
                    ticker_uniform[i],signal,mean_x,beta,cond_var))
                real_val_list.append(real_val)
            # The seed rows are history, not simulation, so they are dropped.
            real_val_arr: np.ndarray = np.array(real_val_list[len(history):])
            for j, inverse in enumerate(RIM_PortOp.activate_inverse_func):
                if inverse is not None:
                    real_val_arr[:,:,j] = inverse(real_val_arr[:,:,j])
            sample[ticker] = real_val_arr
        return sample
    def joint_return (self, n_samples: int = 10000,n_lags:int=4,forward:bool=False,
                  common_theta:bool=True,seed:int|None=None)->pd.DataFrame:
        """Annualised implied return per simulated path, per ticker.

        Args:
            n_samples (int): Simulations per ticker. Defaults to 10000.
            n_lags (int): Lags the persistence fit reads. Defaults to 4.
            forward (bool): Elicit the unconditional moments from the future
                window rather than the training window. Defaults to False.
            common_theta (bool): Pool the decay estimate across the universe.
                Defaults to True.
            seed (int | None): Seed for the copula draw. Defaults to None, i.e.
                fresh entropy and a different answer on every call.

        Returns:
            pd.DataFrame: Indexed by ticker in `self.ticker` order, one column
                per simulation. NaN where a path implies a non-positive payoff,
                which has no real annualised growth rate.
        """
        sample: dict[str, np.ndarray] = self.sampling(n_samples,n_lags,forward,
                                                      common_theta,seed)
        # `self.ticker` order, not `sample.keys()`. The returned array carries no
        # labels, so a caller has nothing to read row i by except this instance's
        # universe -- and `sample` comes back keyed in the order `sample_joint`
        # fitted the copulas in, which agrees with `self.ticker` on 2 of 220
        # positions at 2016-03-31. Anything zipping the rows against
        # `self.ticker` (a portfolio optimiser mapping weights back to names,
        # say) was silently attaching each expected return to a different stock.
        tickers: list[str] = self.ticker
        missing: list[str] = [t for t in tickers if t not in sample]
        if missing:
            raise KeyError(f'sampling returned no paths for {missing}')
        # Two cutoffs, because the two kinds of input become knowable at
        # different times. The accounting panel is lagged for filing, by the same
        # rule `take_training_data` slices the training window with -- a flat
        # three months instead disagreed with it on the 88 mid-quarter dates of
        # `constant.testing_period`, compounding simulated growth off a revenue
        # base one quarter ahead of where the simulation starts, and reading a
        # quarter that may not have been filed. Market data carries no such lag:
        # price and the factor returns are observable at the formation date, and
        # reading them three months stale put a quarter of realised return into
        # the denominator of the implied return, where it would masquerade as a
        # momentum signal.
        acc_date: pd.Timestamp = accounting_cutoff(self.date)
        mkt_date: pd.Timestamp = self.date

        ff5_df: pd.DataFrame = RIM_PortOp.ff5_factor.loc[:mkt_date]
        re_ser: pd.Series = cof.re_cal(ff5_df,tickers)

        # The last *observed* figure per ticker, not the last row. A filer on an
        # offset fiscal calendar has no row on a calendar quarter end -- AAP
        # reports at 2019-12-31 and 2020-06-30 but not 2020-03-31 -- and
        # `.iloc[-1]` reads that gap as NaN, which takes the ticker's whole
        # simulation with it (6.9% of the universe at 2020-03-31). One period of
        # carry is all the misalignment a quarterly panel can produce; a longer
        # one would be inventing a figure rather than reading a stale one.
        rps: pd.DataFrame = RIM_PortOp.general_rps.loc[:acc_date,tickers]
        current_rps: pd.Series = rps.ffill(limit=1).iloc[-1]

        # Extrema over the same trailing window `Data.exclusion` screens on, not
        # over all history. That screen drops a ticker whose book equity goes
        # negative in the last `n_quarter` quarters, but a survivor can still
        # carry a negative P/B from before the window -- CLX passes at
        # 2020-06-30 with a trailing minimum of +20.6 and a 2012 minimum of
        # -469, and the full-history read hands that -469 to `terminal_val` as a
        # terminal multiple. Six of the 273 applicable tickers are in that
        # position at that date.
        pb_df: pd.DataFrame = RIM_PortOp.general_pb.loc[:acc_date,tickers
                                                        ].iloc[-n_quarter:]
        extrema_pb: pd.DataFrame = pd.concat([pb_df.min(),pb_df.max()]
                                             ,keys=['min','max'],axis=1)
        price_df: pd.DataFrame = RIM_PortOp.general_price.loc[:mkt_date,tickers]
        current_price: pd.Series = price_df.ffill(limit=1).iloc[-1]
        
        return_list: list[np.ndarray] = []
        for ticker in tickers:
            arr: np.ndarray = sample[ticker]
            re: np.float64 = re_ser.loc[ticker]
            min_pb = extrema_pb.loc[ticker,'min']
            max_pb = extrema_pb.loc[ticker,'max']
            revenue = current_rps.loc[ticker]
            # From the last row, not `price_df[ticker]`: the RIM panels are
            # unstacked wide, so `.loc[ticker]` on one of them is a lookup for a
            # date named after a ticker.
            price = current_price.loc[ticker]
            T = arr.shape[0]
            # `terminal_val` takes the *discounted* residual income and
            # `implied_return` the ROE-compounded one. They are the same cash
            # flows on two bases, and the split is deliberate: `be0 +
            # discounted_ri` is the residual income valuation V0, which is the
            # identity the Gordon branch inside `terminal_val` is built on, so
            # handing it the compounded figure would leave it valuing nothing.
            discounted_ri, be0, beT, future_bn = rim_mapping(arr,revenue,re)
            tv_T: np.ndarray = terminal_val(min_pb, max_pb, discounted_ri,#type: ignore
                                            be0, beT,T-1,re)
            annual_return: np.ndarray = implied_return(future_bn,be0,beT,tv_T,
                                                       T-1,re, price)
            return_list.append(annual_return)
        # Labelled, so the ticker a row belongs to travels with the row. The
        # array alone left that to a convention the caller had to know, and a
        # caller that paired it with the wrong list -- a portfolio optimiser
        # mapping positional weights back to names, say -- got a silently
        # scrambled cross-section rather than an error.
        return pd.DataFrame(np.array(return_list),index=tickers)
    def weight (self, n_samples: int = 10000,n_lags:int=4,forward:bool=False,
                  common_theta:bool=True,seed:int|None=None, long_only:bool=True):
        return_df: pd.DataFrame = self.joint_return(n_samples,n_lags,
                                                    forward,common_theta,seed)
        weight: pd.Series = port_weight(return_df,long_only=long_only)
        return weight
        
            
                
                    
                    
            
            
            
                
                
        
        
            
            