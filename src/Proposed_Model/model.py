import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from .data_generator import generator, take_training_data
from pyvinecopulib import FitControlsBicop
from pyvinecopulib import BicopFamily
from .utils.dependence_structure import copula_structure
from .utils.simulation import sample_joint, CHARACTERISTICS
from .utils.sampling_distribution import (
    lag_moments, signal_cal, conditional_moments, sample_conditional
)
from .utils.theta import find_theta
from .utils.variable_tranformation import Ros_Trans, Ate_Trans, Ato_Trans
from constant import n_quarter_ahead
from .utils.elicitation import parameter_derived
class RIM_PortOp:
    bicop_controls = FitControlsBicop(selection_criterion='aic',
                            family_set=[BicopFamily.gaussian,#type: ignore
                                        BicopFamily.frank, #type: ignore
                                        BicopFamily.clayton]) #type: ignore

    general_data,general_pb, general_bvps, general_rps, general_price = generator()
    ticker_across_time: dict[str, list[str]] = dict()
    # Indexed by CHARACTERISTICS, so the padding is load bearing: 'g' leads
    # that tuple and is modelled untransformed, while the other three are
    # generated on the scale data_generator put them on and have to come back.
    activate_inverse_func: list = [None, Ate_Trans.inverse,
                                   Ato_Trans.inverse, Ros_Trans.inverse]
    def __init__(self, date:str|pd.Timestamp,ticker:list[str]) -> None:
        if isinstance(date, str):
            self.date: pd.Timestamp = pd.Timestamp(date)
        else:
            self.date: pd.Timestamp = date
        cache: tuple[pd.DataFrame,pd.
                     DataFrame,list[str]] = take_training_data(
            RIM_PortOp.general_data,self.date,ticker)
        # Keyed on the parsed date, never on the argument: '2020-06-30' and the
        # Timestamp it parses to are the same formation date but not the same
        # string, and `depedence_structure` reads this back through self.date.
        self.train_df, self.future_df, RIM_PortOp.ticker_across_time[
            str(self.date)] = cache
    def depedence_structure (self):
        controls = RIM_PortOp.bicop_controls
        tickers: list[str] = RIM_PortOp.ticker_across_time[str(self.date)]
        
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
    
    def sampling (self, n_samples: int = 10000,n_lags:int=2,forward:bool=False,
                  common_theta:bool=True):
        outer_copula, inner_copula = self.depedence_structure()
        uniform_sample: dict[str,np.ndarray] = sample_joint(
            outer_copula,inner_copula,n_quarter_ahead,n_samples
        )
        tickers: list[str] = list(uniform_sample.keys())
        large_traindata: pd.DataFrame = self.train_df.loc[tickers,CHARACTERISTICS]
        if forward:
            eli_traindata: pd.DataFrame = self.future_df.loc[tickers,CHARACTERISTICS]
        else:
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
            eli_df = eli_traindata.xs(ticker,level='ticker')
            ticker_uniform: np.ndarray = uniform_sample[ticker]
            T, N_s, d = ticker_uniform.shape
            if not isinstance(eli_df,pd.DataFrame):
                raise ValueError(f"{ticker} in elicitation data have only 1 point of data")
            mean_x, var_x = parameter_derived(eli_df)
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
            unfit: np.ndarray = ~(np.isfinite(beta) & np.isfinite(cond_var))
            if unfit.any():
                warnings.warn(
                    f'{ticker}: no conditional fit for '
                    f'{[c for c,u in zip(CHARACTERISTICS,unfit) if u]}; '
                    f'drawing them from the unconditional law',
                    RuntimeWarning)
                thetas = np.where(unfit,1.0,thetas)
                beta = np.where(unfit,0.0,beta)
                cond_var = np.where(unfit,var_x,cond_var)

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
                
                    
                    
            
            
            
                
                
        
        
            
            