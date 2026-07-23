import numpy as np
from constant import risk_aversion

def crra_utility (r:np.ndarray, risk_aversion:float = risk_aversion):
    utility: np.ndarray = (1+r)**(1-risk_aversion)/(1-risk_aversion)
    return np.mean(utility)

def ppp_weight (theta:np.ndarray, benchmark: np.ndarray, 
                char: np.ndarray):
    # Stock axis is -2, never 1: `char` arrives as a (n_t, n_char) cross-section
    # from portfolio_weight but as a (n_period, n_t, n_char) panel from
    # find_theta. shape[1] is n_t only in the 3-D case; in the 2-D case it is
    # the characteristic count, which would scale every tilt by n_t/n_char.
    n_t:int = char.shape[-2]
    return benchmark+char@theta/n_t

def ppp_op (theta:np.ndarray, benchmark: np.ndarray, 
                char: np.ndarray,r_arr:np.ndarray):
    weight: np.ndarray = ppp_weight(theta,benchmark,char)
    r_vec: np.ndarray = np.sum(weight*r_arr, axis=1)
    return -crra_utility(r_vec)