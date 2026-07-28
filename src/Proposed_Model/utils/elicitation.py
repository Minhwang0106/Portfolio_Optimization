import numpy as np
import pandas as pd
from scipy.stats import norm

def get_quantile (data: pd.DataFrame, 
                  q_list: list|np.ndarray=[0.25,0.5,0.75]):
    quantile_list: list[np.ndarray] = []
    for q in q_list:
        quantile_list.append(data.quantile(q).to_numpy())
    return tuple(quantile_list)

def parameter_derived (data: pd.DataFrame, 
                  q_list: list|np.ndarray=[0.25,0.5,0.75]):
    q_value: tuple = get_quantile(data,q_list)
    mean: np.ndarray = q_value[1]
    std: np.ndarray = (q_value[-1]-q_value[0])/(
        norm.ppf(q_list[-1])-norm.ppf(q_list[0]))
    return mean, std**2