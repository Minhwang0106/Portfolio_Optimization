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
    def __init__(self, date:str) -> None:
        self.date = date

    @classmethod
    def class_config(cls, com=risk_com, correl_com=risk_correl_com,
                      n_day=risk_n_day, n_month=risk_n_month,
                      n_workers: int | None = None):
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
    def portfolio_weight (self):
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