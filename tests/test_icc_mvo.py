"""Tests for `src.ICC_MVO`: the GLS implied cost of capital, Ledoit-Wolf
constant-correlation shrinkage, the maximum-Sharpe solve, and the inputs built
from the panels -- plus the FF48 mapping in `src.Data.industry`.

Everything here runs on synthetic data; none of it reads the raw panels.

The `test_regression_*` cases are named after defects the first version of the
package had, so a refactor that brings one back fails by name.
"""
import numpy as np
import pandas as pd
import pytest

from src.ICC_MVO.icc_adjusted import (
    gls_path, gls_value, icc_gls, winsorize, rescaled_momentum,
    expected_excess_return,
)
from src.ICC_MVO.volatility import ledoit_wolf_cc
from src.ICC_MVO.utils import max_sharpe_weight, _floor_and_cap
from src.ICC_MVO.inputs import (
    realised_eps, trailing_eps, complete_eps, dividend_events,
    monthly_dividends, trailing_dps, payout_ratio, annual_roe,
    industry_target_roe, panel_roe_pool, read_roe_pool, momentum,
)
import src.Data.industry as industry_mod
import src.Data.industry_pool as pool_mod
from src.Data.industry import (
    parse_siccodes, sic_to_ff48, fetch_sic_by_cik, unmatched_sic_codes,
)
from src.Data.industry_pool import (
    frame_values, common_equity, firm_year_roe, build_pool,
    EQUITY_WITH_NCI_TAG,
)


# --- the GLS implied cost of capital -----------------------------------------

# GLS's two worked examples (their Appendices A and B): three years of EPS
# forecasts, opening book value per share, payout ratio, industry target ROE,
# price, and the ICC they report. Horizon T = 12.
GM = dict(eps=(6.75, 7.73, 8.29), b0=17.01, payout=0.196, target=0.16,
          price=48.50, icc=0.1394)
JNJ = dict(eps=(3.68, 4.18, 4.70), b0=11.08, payout=0.362, target=0.18,
           price=86.63, icc=0.0712)


def _path (*cases: dict)-> tuple[np.ndarray, np.ndarray]:
    return gls_path(np.array([c['eps'] for c in cases]),
                    np.array([c['b0'] for c in cases]),
                    np.array([c['payout'] for c in cases]),
                    np.array([c['target'] for c in cases]))


@pytest.mark.parametrize('case', [GM, JNJ], ids=['GM', 'JNJ'])
def test_icc_reproduces_the_gls_worked_examples(case):
    roe, book = _path(case)
    assert icc_gls(np.array([case['price']]), roe, book)[0] == pytest.approx(
        case['icc'], abs=5e-4)


def test_gls_path_matches_the_gm_example_line_by_line():
    roe, book = _path(GM)
    assert book[0, 1] == pytest.approx(22.437, abs=1e-3)   # 17.01+6.75*0.804
    assert roe[0, 2] == pytest.approx(0.289, abs=1e-3)     # 8.29 / B2
    # The fade lands on the target in year T, in equal steps from year 3.
    assert roe[0, -1] == pytest.approx(GM['target'])
    steps = np.diff(roe[0, 2:])
    assert steps == pytest.approx(np.full_like(steps, steps[0]))


def test_regression_terminal_value_is_a_perpetuity_from_year_t():
    """The first version divided by `r * r**T`, not `r * (1+r)**(T-1)`."""
    roe, book = _path(GM)
    rate, horizon = 0.10, roe.shape[1]
    explicit = ((roe[0, :-1]-rate)*book[0, :-1]
                /(1+rate)**np.arange(1, horizon)).sum()
    # The same year-T residual income, paid every year for 5,000 years.
    tail = ((roe[0, -1]-rate)*book[0, -1]
            /(1+rate)**np.arange(horizon, horizon+5000)).sum()
    assert float(gls_value(rate, roe[0], book[0])) == pytest.approx(
        book[0, 0]+explicit+tail, rel=1e-10)


def test_regression_each_firm_is_solved_for_its_own_price():
    """The first version minimised the *negative* squared error, jointly --
    maximising the pricing error -- and sliced firms where it meant years."""
    roe, book = _path(GM, JNJ)
    price = np.array([GM['price'], JNJ['price']])
    icc = icc_gls(price, roe, book)
    assert icc == pytest.approx([GM['icc'], JNJ['icc']], abs=5e-4)
    assert gls_value(icc, roe, book) == pytest.approx(price, rel=1e-8)


def test_regression_explicit_years_are_read_along_the_row():
    """`eps_arr[i]` used to pick firm i where it meant year i."""
    eps = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    roe, book = gls_path(eps, np.array([10.0, 100.0]), np.zeros(2),
                         np.array([0.1, 0.1]))
    # Full retention: B_t = B_0 + cumulative EPS, and ROE_t = EPS_t / B_{t-1}.
    assert book[0, :3] == pytest.approx([10.0, 11.0, 13.0])
    assert roe[0, :3] == pytest.approx([0.1, 2/11, 3/13])
    assert book[1, :3] == pytest.approx([100.0, 110.0, 130.0])


def test_icc_is_nan_without_positive_book_or_a_root():
    eps = np.array([[-20.0, -20.0, -20.0], GM['eps']])
    roe, book = gls_path(eps, np.array([10.0, GM['b0']]),
                         np.array([0.0, GM['payout']]),
                         np.array([0.15, GM['target']]))
    # Firm 0 loses its book value; firm 1 is priced far below its value at
    # 100%, so the value never crosses the price inside the bracket.
    assert np.isnan(icc_gls(np.array([50.0, 1e-3]), roe, book)).all()


# --- expected return ----------------------------------------------------------

def test_winsorize_clips_each_tail_at_its_quantile():
    out = winsorize(pd.Series(np.arange(101, dtype=float)), 0.01)
    assert (out.min(), out.max()) == (pytest.approx(1.0), pytest.approx(99.0))


def test_regression_expected_return_is_excess_of_rf_and_monthly():
    """The first version added momentum to the ICC and stopped: no risk-free
    rate, no winsorising, annual means against a monthly covariance."""
    idx = pd.Index(list('ABCDEFGHIJ'))
    icc = pd.Series(np.linspace(0.05, 0.14, 10), index=idx)
    mom = pd.Series(np.linspace(-0.3, 0.6, 10), index=idx)
    resc = rescaled_momentum(mom, icc)
    assert resc.mean() == pytest.approx(0.0, abs=1e-12)
    assert resc.std() == pytest.approx(icc.std())
    er = expected_excess_return(icc, mom, rf=0.02, winsor=0.0)
    assert er.to_numpy() == pytest.approx(((icc+resc-0.02)/12).to_numpy())


def test_regression_momentum_is_twelve_minus_one_not_the_whole_window():
    """The first version took the product of raw returns over all 60 months."""
    months = pd.date_range('2019-01-31', periods=14, freq='ME')
    r = pd.DataFrame({'X': 0.01}, index=months)
    r.loc[months[-1], 'X'] = 0.50      # the formation month: skipped
    r.loc[months[-13], 'X'] = -0.50    # month t-12: before the window
    assert momentum(r, months[-1], ['X'])['X'] == pytest.approx(1.01**11-1)
    r.loc[months[-5], 'X'] = np.nan
    assert np.isnan(momentum(r, months[-1], ['X'])['X'])


# --- covariance ---------------------------------------------------------------

def _lw_by_the_formulas (x: np.ndarray)-> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) straight from the definitions, entry by entry."""
    t, n = x.shape
    y = x-x.mean(axis=0)
    s = y.T@y/t
    sd = np.sqrt(np.diag(s))
    rbar = sum(s[i, j]/(sd[i]*sd[j]) for i in range(n) for j in range(n)
               if i != j)/(n*(n-1))
    f = np.array([[s[i, i] if i == j else rbar*sd[i]*sd[j]
                   for j in range(n)] for i in range(n)])
    pi = np.array([[np.mean((y[:, i]*y[:, j]-s[i, j])**2) for j in range(n)]
                   for i in range(n)])

    def theta (k, i, j):
        return np.mean((y[:, k]**2-s[k, k])*(y[:, i]*y[:, j]-s[i, j]))

    rho = np.trace(pi)+sum(
        rbar/2*(np.sqrt(s[j, j]/s[i, i])*theta(i, i, j)
                +np.sqrt(s[i, i]/s[j, j])*theta(j, i, j))
        for i in range(n) for j in range(n) if i != j)
    gamma = ((f-s)**2).sum()
    delta = max(0.0, min(1.0, (pi.sum()-rho)/gamma/t))
    return delta*f+(1-delta)*s, delta


def _factor_returns (seed: int, t: int, n: int)-> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.2, 1.5, size=n)
    return 0.03*rng.normal(size=(t, 1))*loadings+0.05*rng.normal(size=(t, n))


def _two_block_returns (seed: int, t: int, n: int)-> np.ndarray:
    """Two industries: high correlation within each, none across -- as far
    from constant correlation as a panel gets, so the intensity is interior
    rather than clipped at 1 as it is on near-exchangeable noise."""
    rng = np.random.default_rng(seed)
    blocks = np.zeros((2, n))
    blocks[0, :n//2] = blocks[1, n//2:] = 1.0
    return 0.04*rng.normal(size=(t, 2))@blocks+0.02*rng.normal(size=(t, n))


def test_ledoit_wolf_matches_the_paper_formulas_entry_by_entry():
    x = _two_block_returns(1, 60, 5)
    sigma, delta = ledoit_wolf_cc(x)
    ref_sigma, ref_delta = _lw_by_the_formulas(x)
    assert 0 < delta < 1          # an interior intensity, so the test bites
    assert delta == pytest.approx(ref_delta, rel=1e-10)
    assert sigma == pytest.approx(ref_sigma, rel=1e-10)


def test_ledoit_wolf_is_symmetric_positive_definite_with_sample_variances():
    x = _factor_returns(1, 60, 40)
    sigma, delta = ledoit_wolf_cc(x)
    assert sigma == pytest.approx(sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0
    assert 0 <= delta <= 1
    assert np.diag(sigma) == pytest.approx(((x-x.mean(axis=0))**2).mean(axis=0))


def test_regression_covariance_is_between_assets_not_between_months():
    """The first version handed sklearn an (N, T) array: a T x T matrix."""
    assert ledoit_wolf_cc(_factor_returns(2, 60, 8))[0].shape == (8, 8)


def test_ledoit_wolf_rejects_an_incomplete_panel():
    x = _factor_returns(3, 60, 4)
    x[3, 2] = np.nan
    with pytest.raises(ValueError, match='complete'):
        ledoit_wolf_cc(x)


# --- the maximum-Sharpe solve ------------------------------------------------

def _covariance (seed: int, n: int)-> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))*0.03
    return a@a.T+np.diag(np.full(n, 0.002))


def test_uncapped_solve_is_the_tangency_portfolio_when_that_is_long_only():
    sigma = _covariance(3, 8)
    target = np.random.default_rng(4).uniform(0.05, 0.2, size=8)
    target /= target.sum()
    # Sigma^-1 mu is `target` itself, all positive, so long-only does not bind.
    w = max_sharpe_weight(sigma@target, sigma, cap=1.0, min_weight=0.0)
    assert w == pytest.approx(target, abs=1e-6)


def test_long_short_form_is_the_closed_form_tangency():
    sigma = _covariance(5, 6)
    mu = np.array([0.01, -0.004, 0.006, 0.002, -0.001, 0.008])
    raw = np.linalg.solve(sigma, mu)
    assert max_sharpe_weight(mu, sigma, cap=1.0, long_only=False) == \
        pytest.approx(raw/raw.sum())


def test_the_cap_binds_on_the_name_that_would_otherwise_dominate():
    n = 30
    sigma = np.diag(np.full(n, 0.004))+0.001
    mu = np.full(n, 0.005)
    mu[0] = 0.05
    w = max_sharpe_weight(mu, sigma, cap=0.05)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.05, abs=1e-9)
    assert (w <= 0.05).all()
    # The other 29 are exchangeable, so they split the rest evenly.
    assert w[1:] == pytest.approx(np.full(29, 0.95/29), abs=1e-6)


def test_the_solution_ignores_the_scale_of_mu_and_sigma():
    sigma = _covariance(6, 25)
    mu = np.random.default_rng(6).uniform(-0.002, 0.01, size=25)
    assert max_sharpe_weight(mu, sigma, cap=0.2) == pytest.approx(
        max_sharpe_weight(12*mu, 3*sigma, cap=0.2), abs=1e-6)


def test_floor_zeroes_dust_and_keeps_the_cap_exact():
    out = _floor_and_cap(np.array([0.6, 0.3, 0.09995, 0.00005]), cap=0.6,
                         min_weight=1e-4)
    assert out[3] == 0.0
    assert out.sum() == pytest.approx(1.0)
    # Renormalising alone would put the first name at 0.60003.
    assert out[0] == pytest.approx(0.6, abs=1e-15)


def test_raises_without_a_positive_expected_excess_return():
    with pytest.raises(ValueError, match='positive'):
        max_sharpe_weight(np.full(25, -0.01), np.eye(25)*0.004, cap=0.05)


def test_raises_when_the_cap_cannot_be_met():
    with pytest.raises(ValueError, match='cap'):
        max_sharpe_weight(np.full(10, 0.01), np.eye(10)*0.004, cap=0.05)


# --- inputs from the panels ---------------------------------------------------

@pytest.fixture
def eps_q():
    quarters = pd.date_range('2019-03-31', periods=16, freq='QE')
    return pd.DataFrame({'A': np.arange(1.0, 17.0),
                         'B': np.arange(1.0, 17.0)}, index=quarters)


def test_realised_eps_sums_complete_blocks_after_the_cutoff(eps_q):
    eps_q.iloc[9, 1] = np.nan                     # a hole in B's second year
    out = realised_eps(eps_q, pd.Timestamp('2019-12-31'))
    assert out.loc['A'].tolist() == [26.0, 42.0, 58.0]   # 5-8, 9-12, 13-16
    assert np.isnan(out.loc['B', 'fy2'])
    # One quarter later the third block runs past the panel.
    assert np.isnan(realised_eps(eps_q, pd.Timestamp('2020-03-31'))
                    .loc['A', 'fy3'])
    assert trailing_eps(eps_q, pd.Timestamp('2019-12-31'))['A'] == 10.0


def test_regression_missing_years_compound_at_one_plus_g():
    """The first version grew a missing year as `eps * g`, not `eps * (1+g)`."""
    realised = pd.DataFrame({'fy1': [2.0, np.nan, 2.0],
                             'fy2': [np.nan, np.nan, np.nan],
                             'fy3': [np.nan, np.nan, 5.0]},
                            index=['one', 'none', 'gap'])
    e0 = pd.Series({'one': 1.0, 'none': 1.5, 'gap': 1.0})
    eps, n_oracle = complete_eps(realised, e0,
                                 pd.Series(0.1, index=realised.index))
    assert eps.loc['one'].tolist() == pytest.approx([2.0, 2.2, 2.42])
    assert eps.loc['none'].tolist() == pytest.approx([1.65, 1.815, 1.9965])
    # A realised year after a missing one is not used: the run is broken.
    assert eps.loc['gap'].tolist() == pytest.approx([2.0, 2.2, 2.42])
    assert n_oracle.to_dict() == {'one': 1, 'none': 0, 'gap': 1}


def test_dividends_are_recovered_from_the_adjusted_close_factor():
    dates = pd.bdate_range('2023-01-02', periods=6)
    close = pd.Series([50.0, 51.0, 50.5, 50.2, 50.8, 51.1], index=dates)
    # Yahoo scales every price before an ex-date by 1 - D / C_{e-1}: here
    # 0.30 going ex on day 1 and 0.40 on day 3, compounding backwards.
    factor = pd.Series(1.0, index=dates)
    factor.iloc[:3] *= 1-0.40/close.iloc[2]
    factor.iloc[:1] *= 1-0.30/close.iloc[0]
    daily = pd.DataFrame({'close': close.to_numpy(),
                          'adj_close': (close*factor).to_numpy()},
                         index=pd.MultiIndex.from_product(
                             [['X'], dates], names=['ticker', 'date']))
    events = dividend_events(daily).loc['X']
    assert events.round(10).to_dict() == {dates[1]: 0.30, dates[3]: 0.40}


def test_trailing_dividends_cover_twelve_months_and_zero_for_non_payers():
    idx = pd.MultiIndex.from_tuples(
        [('X', pd.Timestamp(d)) for d in
         ('2019-02-15', '2019-05-15', '2019-08-15', '2019-11-15',
          '2020-02-14')], names=['ticker', 'date'])
    monthly = monthly_dividends(pd.Series([0.1, 0.1, 0.1, 0.1, 0.2],
                                          index=idx))
    dps = trailing_dps(monthly, pd.Timestamp('2019-12-31'), ['X', 'Y'])
    assert dps['X'] == pytest.approx(0.4)
    assert dps['Y'] == 0.0


def test_regression_payout_is_dividends_over_earnings_with_the_gls_fallback():
    """The first version passed net income where the retention ratio belongs."""
    dps = pd.Series({'profit': 1.0, 'loss': 1.2, 'none': 0.0, 'lavish': 5.0})
    e0 = pd.Series({'profit': 4.0, 'loss': -2.0, 'none': -1.0, 'lavish': 2.0})
    assets = pd.Series({'profit': 50.0, 'loss': 100.0, 'none': 10.0,
                        'lavish': 20.0})
    k = payout_ratio(dps, e0, assets)
    assert k['profit'] == pytest.approx(0.25)
    assert k['loss'] == pytest.approx(1.2/(0.06*100.0))
    assert k['none'] == 0.0
    assert k['lavish'] == 1.0


def test_annual_roe_is_calendar_income_over_the_prior_december_book():
    quarters = pd.date_range('2018-03-31', periods=12, freq='QE')
    ni = pd.DataFrame({'A': 1.0}, index=quarters)
    be = pd.DataFrame({'A': [10.0]*4+[20.0]*4+[40.0]*4}, index=quarters)
    roe = annual_roe(ni, be)
    assert np.isnan(roe.loc[2018, 'A'])            # no 2017 book to start from
    assert roe.loc[2019, 'A'] == pytest.approx(0.4)
    assert roe.loc[2020, 'A'] == pytest.approx(0.2)


def test_industry_target_drops_losses_and_thin_industries():
    roe = pd.DataFrame({'A': 0.10, 'B': 0.20, 'C': -0.50, 'D': 0.30},
                       index=range(2010, 2020))
    roe.loc[2010:2012, 'D'] = np.nan
    industry = pd.Series({'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 2.0})
    pool = panel_roe_pool(roe, industry)
    per, overall = industry_target_roe(pool, last_year=2019,
                                       years=(5, 10), min_obs=10)
    assert per[1.0] == pytest.approx(0.15)    # A and B; C's losses left out
    assert 2.0 not in per.index               # 7 firm-years, under 10
    assert overall == pytest.approx(0.20)
    with pytest.raises(ValueError, match='needs 5'):
        industry_target_roe(pool[pool['year'] >= 2016], last_year=2019)


def test_industry_target_uses_every_row_of_the_pool_not_only_the_universe():
    # Filers outside the S&P 500 panel have no ticker, only a CIK and an FF48
    # code; they count towards their industry's median all the same.
    pool = pd.DataFrame({'year': np.repeat(range(2015, 2020), 3),
                         'cik': ['1', '2', '3']*5,
                         'roe': [0.10, 0.30, 0.50]*5,
                         'ff48': [7.0, 7.0, np.nan]*5})
    per, overall = industry_target_roe(pool, last_year=2019, min_obs=10)
    assert per[7.0] == pytest.approx(0.20)
    assert overall == pytest.approx(0.30)     # the unclassified filer counts here


def test_a_missing_pool_file_warns_and_leaves_the_fallback_to_the_caller(
        tmp_path):
    with pytest.warns(RuntimeWarning, match='S&P 500'):
        assert read_roe_pool(tmp_path/'industry_roe_pool.csv') is None


# --- the SEC filer pool --------------------------------------------------------

def _frame (values: dict[int, float])-> pd.DataFrame:
    return frame_values({'data': [{'cik': cik, 'entityName': f'firm {cik}',
                                   'val': val} for cik, val in values.items()]})


def test_frame_values_give_one_row_per_ten_digit_cik():
    out = frame_values({'data': [
        {'cik': 320193, 'entityName': 'Apple', 'val': 5},
        {'cik': 320193, 'entityName': 'Apple', 'val': 7}]})
    assert out.index.tolist() == ['0000320193']
    assert out.loc['0000320193', 'val'] == 7.0
    assert frame_values({}).empty


def test_common_equity_is_parent_equity_else_total_less_nci_then_less_preferred():
    out = common_equity(parent=pd.Series({'a': 100.0}),
                        with_nci=pd.Series({'a': 999.0, 'b': 80.0}),
                        nci=pd.Series({'b': 30.0}),
                        preferred=pd.Series({'a': 10.0, 'b': 5.0}))
    assert out['a'] == 90.0                   # parent equity wins over the total
    assert out['b'] == 45.0                   # 80 - 30 NCI - 5 preferred


def test_firm_year_roe_needs_both_figures_and_equity_above_the_floor():
    out = firm_year_roe(pd.Series({'a': 2e6, 'b': 1e6, 'c': 1e6}),
                        pd.Series({'a': 20e6, 'b': 5e6, 'd': 50e6}),
                        min_equity=10e6)
    assert out.index.tolist() == ['a']
    assert out.loc['a', 'roe'] == pytest.approx(0.1)


def test_pool_divides_the_years_income_by_the_prior_year_end_equity(
        monkeypatch):
    frames = {
        ('NetIncomeLoss', 'CY2020'): {1: 30e6},
        ('ProfitLoss', 'CY2020'): {1: 99e6, 2: 5e6},   # 2 has no NetIncomeLoss
        ('StockholdersEquity', 'CY2019Q4I'): {1: 200e6},
        (EQUITY_WITH_NCI_TAG, 'CY2019Q4I'): {2: 60e6, 3: 1e9},
        ('MinorityInterest', 'CY2019Q4I'): {2: 10e6},
        ('PreferredStockValue', 'CY2019Q4I'): {1: 50e6},
    }
    asked = []

    def fake (session, tag, period, header, pause=0.0):
        asked.append((tag, period))
        return _frame(frames.get((tag, period), {}))

    monkeypatch.setattr(pool_mod, 'fetch_frame', fake)
    pool = build_pool({}, [2020, 2021], min_equity=10e6, pause=0.0)
    got = pool.set_index('cik')
    assert got.loc['0000000001', 'roe'] == pytest.approx(30/150)
    assert got.loc['0000000002', 'roe'] == pytest.approx(5/50)
    assert '0000000003' not in got.index      # equity but no income
    assert pool['year'].unique().tolist() == [2020]   # 2021 not reported yet
    assert ('StockholdersEquity', 'CY2020Q4I') not in asked


def test_sic_lookups_resume_from_the_cache(tmp_path, monkeypatch):
    cache = tmp_path/'sic_by_cik.csv'
    pd.DataFrame({'cik': ['0000000001'], 'sic': [1311]}).to_csv(cache,
                                                                index=False)
    asked = []

    class Reply:
        status_code = 200

        def json (self):
            return {'sic': '2834'}

        def raise_for_status (self):
            pass

    def fake_get (session, url, header):
        asked.append(url)
        return Reply()

    monkeypatch.setattr(industry_mod, 'sec_get', fake_get)
    out = fetch_sic_by_cik(['0000000001', '0000000002', '0000000002'], {},
                           pause=0.0, cache=cache)
    assert out == {'0000000001': 1311, '0000000002': 2834}
    assert len(asked) == 1 and '0000000002' in asked[0]
    stored = pd.read_csv(cache, dtype={'cik': str})
    assert stored['cik'].tolist() == ['0000000001', '0000000002']
    assert stored['sic'].tolist() == [1311, 2834]


# --- Fama-French 48 ------------------------------------------------------------

SICCODES_EXCERPT = """\
 1 Agric  Agriculture
          0100-0199 Agricultural production - crops
          0200-0299 Agricultural production - livestock
          2048-2048 Prepared feeds for animals

 2 Food   Food Products
          2000-2009 Food and kindred products
          2010-2019 Meat products

48 Other  Almost Nothing
          4950-4959 Sanitary services
"""


def test_sic_ranges_map_to_their_industry_and_the_rest_to_other():
    table = parse_siccodes(SICCODES_EXCERPT, expect=3)
    assert table['ff48_short'].unique().tolist() == ['Agric', 'Food', 'Other']
    ff = sic_to_ff48(pd.Series([150, 2048, 2015, 4955, 9999, np.nan]), table)
    assert ff.iloc[:5].tolist() == [1, 1, 2, 48, 48]
    assert np.isnan(ff.iloc[5])
    assert np.isnan(sic_to_ff48(pd.Series([9999]), table, other=None).iloc[0])


def test_siccodes_parser_refuses_a_file_it_does_not_recognise():
    with pytest.raises(ValueError, match='48 industries'):
        parse_siccodes(SICCODES_EXCERPT)


def test_unmatched_sic_codes_excludes_codes_with_their_own_other_range():
    table = parse_siccodes(SICCODES_EXCERPT, expect=3)
    # 4955 has its own range under ff48=48 'Other'; 9999 and NaN do not.
    sic = pd.Series([150, 4955, 9999, 9999, np.nan])
    assert unmatched_sic_codes(sic, table) == [9999]


def test_sic_override_wins_over_the_range_table_and_over_other():
    table = parse_siccodes(SICCODES_EXCERPT, expect=3)
    out = sic_to_ff48(pd.Series([150, 9999]), table, override={150: 2, 9999: 7})
    assert out.tolist() == [2, 7]
    assert unmatched_sic_codes(pd.Series([9999]), table,
                               override={9999: 7}) == []


def test_parse_siccodes_warns_on_ranges_two_industries_both_claim():
    clashing = SICCODES_EXCERPT+'\n 3 Dup    Duplicate\n          2005-2020 Overlap\n'
    with pytest.warns(RuntimeWarning, match='overlaps'):
        table = parse_siccodes(clashing, expect=4)   # 1, 2, 3, 48
    # first-range-wins: the pre-existing Food range (2 Food) still claims 2015.
    assert sic_to_ff48(pd.Series([2015]), table).iloc[0] == 2


