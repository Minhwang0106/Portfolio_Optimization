import numpy as np
from math import exp

from constant import gordon_den_floor, reinvest_rate_cap

def rim_mapping (arr: np.ndarray, rps: float, re:float,
                 rate_cap: float = reinvest_rate_cap):
    """Turn one ticker's simulated characteristic paths into RIM quantities.

    Args:
        arr (np.ndarray): this array is 3 dimension array with
        shape = (T, Ns, d) T is time, Ns is number of simulation
        d is the characteristics (g, ate, ato, and ros), on the axis-2 order of
        `simulation.CHARACTERISTICS` and already inverted back onto their
        natural scale by `RIM_PortOp.sampling`.

        rps (float): revenue per share, the last reported one
        re (float): cost of funds, per quarter
        rate_cap (float): largest per-quarter ROE residual income is compounded
            forward at, applied symmetrically. Defaults to
            `constant.reinvest_rate_cap`.

    Returns:
        tuple[np.ndarray, ...]: `(discounted_ri, be0, beT, future_bn)`, each of
            shape (Ns,) -- the residual income of periods 1..T-1 discounted back
            to period 0, the book equity per share at the first and last
            simulated period, and the terminal-period payoff with every
            component carried forward at ROE.

            The two residual income figures are on different bases on purpose.
            `discounted_ri` is the present value `terminal_val` needs, because
            `be0 + discounted_ri` is the residual income valuation V0 and that
            identity is what its Gordon branch is built on. `future_bn` is the
            same cash flows compounded to the terminal period at the rate the
            firm earns, which is what `implied_return` divides by price.
    """
    g_arr: np.ndarray = arr[:,:,0]
    ate_arr: np.ndarray = arr[:,:,1]
    ato_arr: np.ndarray = arr[:,:,2]
    ros_arr: np.ndarray = arr[:,:,3]
    # DuPont, all three factors: ROE = (NI/revenue)(revenue/assets)(assets/BE).
    # Squaring `ate` and dropping `ato` still returns something ROE-shaped, and
    # the `ate*ato` in be_arr below does not cancel it.
    roe_arr: np.ndarray = ate_arr*ato_arr*ros_arr

    cum_g: np.ndarray = np.cumsum(g_arr,axis=0)
    rps_arr: np.ndarray = rps*np.exp(cum_g)
    be_arr: np.ndarray = rps_arr/(ate_arr*ato_arr)
    excess_return: np.ndarray = roe_arr - re
    be0: np.ndarray = be_arr[0]
    beT: np.ndarray = be_arr[-1]
    #Note: we should remember that the most recent current quarter is not reported which means that
    #if we are standing at time t, we only have data of time t-1 and so the the t_period have to be
    #lengthen by 1 which are not implemented in our code. Which means if i want to create 12 quarter
    #I have to simulation 13 quarter
    time: np.ndarray = np.arange(1,arr.shape[0]).reshape(-1,1)
    # RI_t = (ROE_t - re) * BE_{t-1}: residual income is earned on the book
    # equity carried into the period, so book equity lags the excess return by
    # one row. Reversing it instead pairs period 1 with the terminal book value.
    residual_income: np.ndarray = excess_return[1::]*be_arr[:-1]
    discounted_ri: np.ndarray = np.sum(residual_income*np.exp(-re*time),axis=0)

    # Carry each component forward to the terminal period at the rate the firm
    # earns rather than at `re`. A quantity sitting at period t compounds over
    # the ROE of every period *after* it, so the exponent has to exclude t's own
    # ROE: appending a zero row before the reversed cumulative sum shifts the
    # whole schedule by one and gives the terminal period a factor of exp(0)=1.
    # Row 0 of the stack is `be0`, which is carried over the full path.
    # Capped before it reaches the exponent. `Ate_Trans.inverse` is unbounded,
    # so the ROE draw has a right tail that is a simulation artefact rather than
    # a business: uncapped, `exp(sum(roe))` reaches 1e254 on single paths and the
    # cross-sectional mean implied return runs to 1e46. See
    # `constant.reinvest_rate_cap`. Symmetric, so a path that destroys equity
    # implausibly fast is bounded too.
    capped_roe: np.ndarray = np.clip(roe_arr[1:],-rate_cap,rate_cap)
    reinvest_rate: np.ndarray = np.vstack(
        [capped_roe, np.zeros(shape=(1,residual_income.shape[1]))])
    reinvest_rate = np.exp(np.cumsum(reinvest_rate[::-1],axis=0))[::-1]

    # Summed over time: this is one payoff per path, not a payoff per period.
    # Left 2-D it makes `implied_return` return a (T, Ns) matrix rather than the
    # (Ns,) vector every caller expects.
    future_bn: np.ndarray = np.sum(
        np.concat([be0[None,:],residual_income],axis=0)*reinvest_rate, axis=0)
    return discounted_ri, be0, beT, future_bn

def terminal_val (min_pb: float, max_pb: float,
                  discounted_ri: np.ndarray, be0:np.ndarray,
                  beT:np.ndarray, n_period:int,re:float,
                  den_floor:float=gordon_den_floor):
    """This function calculate the terminal value at the end of simulation not the start

    Args:
        min_pb (float): lowest price to book the ticker has traded at, used
            where the paths accumulate to more than terminal book equity.
        max_pb (float): highest price to book, used in the other branch.
        discounted_ri (np.ndarray): shape (Ns,), from `rim_mapping`.
        be0 (np.ndarray): shape (Ns,), book equity per share at period 0.
        beT (np.ndarray): shape (Ns,), book equity per share at period T.
        n_period (int): quarters between period 0 and the terminal period.
        re (float): cost of funds, per quarter.
        den_floor (float): smallest value the Gordon denominator is allowed to
            take. Defaults to `constant.gordon_den_floor`.

    Returns:
        np.ndarray: shape (Ns,), terminal value per share at `n_period`.

    Note:
        The denominator is clamped from below, not in magnitude, so a path whose
        simulated growth outruns the discount rate is pulled back across zero
        rather than held on the negative branch. See `constant.gordon_den_floor`
        for why the sign is what matters here and a magnitude floor is not.
    """
    condition: np.ndarray = np.sign((be0+discounted_ri)*exp(re*n_period)-beT)
    # min on the +1 branch, max on the -1 branch. Written as
    # `+max_pb*((condition-1)/2)` this returns -max_pb whenever the sign is
    # negative, i.e. a negative price to book. An exact tie (sign 0, which no
    # float path realistically hits) averages the two.
    choosen_pb: np.ndarray = min_pb*((condition+1)/2)-max_pb*((condition-1)/2)
    g_factor: np.ndarray = (beT+choosen_pb*beT*exp(re*n_period)
                            )/(be0+discounted_ri+choosen_pb*beT*exp(re*n_period))
    # `g_factor` carries simulated growth, so nothing holds
    # `g_factor*exp(-re*n_period)` under 1 and the perpetuity is only defined
    # while it is. Flooring leaves the converging paths untouched -- the median
    # denominator is 0.285, five times the floor -- and replaces the diverging
    # ones with the largest terminal value the formula can still express, rather
    # than the arbitrarily large one it returns just the other side of zero.
    denominator: np.ndarray = np.maximum(1-g_factor*exp(-re*n_period),den_floor)
    iv0: np.ndarray = (be0+discounted_ri-beT*exp(-re*n_period))/denominator
    return iv0*g_factor

def implied_return (future_bn:np.ndarray, be0:np.ndarray, 
                    beT:np.ndarray, tv: np.ndarray,
                    n_period:int, re:float, price:float):
    """Annualised implied return per simulated path.

    Args:
        future_bn (np.ndarray): shape (Ns,), from `rim_mapping` -- the opening
            book equity plus every period's residual income, each carried to the
            terminal period at ROE. It already contains `be0`, so nothing here
            may add the opening book a second time.
        be0 (np.ndarray): shape (Ns,), book equity per share at period 0. Kept
            in the signature for callers and diagnostics; the payoff itself
            comes through `future_bn`.
        beT (np.ndarray): shape (Ns,), book equity per share at period T.
        tv (np.ndarray): shape (Ns,), from `terminal_val`.
        n_period (int): quarters the return is earned over.
        re (float): cost of funds, per quarter. Unused in the payoff now that it
            compounds at ROE; retained so the signature still documents the rate
            the residual income was discounted at upstream.
        price (float): price per share paid at the formation date.

    Returns:
        np.ndarray: shape (Ns,), annualised return per path. NaN where the
            payoff is negative, which no fractional power can annualise.
    """
    # Renamed off `terminal_val`, which is the module-level function above.
    tv_excess: np.ndarray = (tv-beT)*0# this factor is dropped in our final equation, therefore it is multiplied by zero
    # Negative components are moved to the denominator rather than netted off
    # the numerator: a path that destroys value is an extra outlay, not a
    # smaller payoff. `future_bn` carries `be0`, so it is not added again.
    numerator: np.ndarray = (np.maximum(future_bn,0)
                             +np.maximum(tv_excess,0))
    denominator: np.ndarray = (price-np.minimum(future_bn,0)
                               -np.minimum(tv_excess,0))
    # A path whose payoff or outlay is non-positive has no real growth rate; it
    # is dropped rather than fed to the power below, where a negative base under
    # a fractional exponent is NaN anyway -- but a NaN that arrives with a
    # RuntimeWarning from inside numpy rather than from a decision made here.
    with np.errstate(divide='ignore', invalid='ignore'):
        R: np.ndarray = np.where(denominator>0,numerator/denominator,np.nan)
    R = np.where(R>=0,R,np.nan)
    # R is the gross return over `n_period` quarters, so the exponent that puts
    # it on a yearly basis is 4/n_period. n_period/4 compounds it instead: at
    # the default 12-quarter horizon it cubes the 3-year return.
    return R**(4/n_period)-1