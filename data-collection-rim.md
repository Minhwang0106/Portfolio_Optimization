---
title: "Master Data Collection Spec — NCKH AEL 2026 (S&P 500, free sources)"
created: 2026-07-16
project: NCKH-AEL-2026
tags: [data, RIM, EPO, PPP, clean-surplus, EDGAR, US-GAAP]
---

# Master Data Collection Spec — thu thập TRONG MỘT LƯỢT

> **Mục đích:** một danh sách tổng thể mọi loại dữ liệu cần cho **cả 4 model** — Proposed (RIM+FF3/5+MIRR), baseline **EPO**, baseline **PPP**, benchmark **1/N** — cộng phần **evaluation**. Đọc §1 (bảng inventory) trước, rồi thu theo §2 (theo từng nguồn).

**Giả định khung (chốt 2026-07-16):**
- Universe: **S&P 500 point-in-time**, 2010–2025 (free path; EDGAR XBRL bắt đầu ~2009).
- Nguồn: 100% free (Ken French + EDGAR + yfinance/Stooq + GitHub membership).
- Survivorship bias = documented limitation.

---

## 1. MASTER INVENTORY — toàn bộ raw data (bảng "một lượt")

| # | Raw data | Nguồn | Tần suất | Proposed | EPO | PPP | Eval |
|---|---|---|---|:--:|:--:|:--:|:--:|
| 1 | Stock total return (gồm cổ tức) | yfinance | Monthly | ✔ (β) | ✔ (Σ, μ) | ✔ (r, mom) | ✔ |
| 2 | Stock price $P_t$ (point-in-time) | yfinance | Monthly/rebalance | ✔ (MIRR) | ~ (anchor) | ✔ (mkt cap) | ✔ |
| 3 | Shares outstanding | EDGAR | Quý | ✔ | | ✔ (size) | |
| 4 | Total equity, NCI, Preferred → **BV common-parent** | EDGAR | Quý | ✔ | | ✔ (B/M) | |
| 5 | Net income, **Comprehensive income**, AOCI | EDGAR | Quý | ✔ (RIM) | | | |
| 6 | Common dividends, buybacks, issuance → $D_t$ | EDGAR | Quý | ✔ | | | |
| 7 | FF factors (MKT, SMB, HML, RMW, CMA) | Ken French | Monthly | ✔ (cost of equity) | | | ✔ (alpha) |
| 8 | Risk-free rate (1-month T-bill) | Ken French | Monthly | ✔ | ✔ | ✔ | ✔ |
| 9 | S&P 500 point-in-time membership | GitHub/Wiki | Monthly | ✔ | ✔ | ✔ | ✔ |
| 10 | Benchmark return (S&P 500 / value-weight) | yfinance / French | Monthly | | | ~ (bench weight) | ✔ |

**Đọc bảng:** Chỉ có **4 nguồn** (yfinance, EDGAR, Ken French, GitHub). Toàn bộ nhu cầu của 3 baseline + evaluation **nằm gọn trong** nhu cầu data của Proposed — thêm EPO/PPP gần như **không tốn data mới**, chỉ tốn code. EPO chỉ cần return panel; PPP thêm market cap + book equity (đã có sẵn cho RIM).

---

## 2. Kế hoạch thu thập THEO NGUỒN (làm tuần tự)

### Nguồn A — Ken French Data Library (tải 1 lần, nhanh nhất)
- FF5 factors + momentum + **RF** (monthly, CSV). Dùng: cost of equity (Proposed), alpha & Sharpe (Eval), excess return (mọi model).
- Công cụ: `pandas_datareader.data.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')`.

### Nguồn B — S&P 500 point-in-time membership
- GitHub `fja05680/sp500` (`S&P 500 Historical Components & Changes.csv`) hoặc Wikipedia "changes" table → dựng danh sách constituent theo từng tháng.
- Ra **danh sách ticker** cho mỗi kỳ rebalance → đây là universe để kéo giá & fundamentals.

### Nguồn C — yfinance (giá & return)
- Với **mọi ticker** xuất hiện trong danh sách lịch sử (kể cả đã rời index): kéo adjusted close monthly → total return.
- Thử Stooq cho tên delisted. Ghi lại tên không lấy được (đo mức survivorship bias).
- Ra: return panel (cho Σ/μ của EPO, momentum của PPP, β của Proposed, evaluation) + $P_t$.

### Nguồn D — SEC EDGAR XBRL (fundamentals, theo CIK)
- Map ticker → CIK (`company_tickers.json`; tên delisted tra `cik-lookup-data.txt`).
- Với mỗi CIK, gọi `companyfacts` hoặc `companyconcept` API kéo các tag ở §4.1 (bảng XBRL).
- Header `User-Agent` + email, ≤10 req/s. Coverage **2009+**.
- Ra: BV common-parent, comprehensive income, shares, dividends → RIM (Proposed) + book-to-market & size (PPP).

---

## 3. Derived variables — công thức từ raw data

| Derived | Công thức | Dùng cho | Raw inputs (#) |
|---|---|---|---|
| Market cap | Shares outstanding × $P_t$ | PPP size, value-weight anchor | 2, 3 |
| Book-to-market (B/M) | BV common-parent ÷ Market cap | PPP characteristic | 2, 3, 4 |
| Momentum | $\prod_{s=t-12}^{t-2}(1+r_s)-1$ (skip tháng gần nhất) | PPP characteristic | 1 |
| FF beta / cost of equity $r_e$ | Hồi quy $r_{i}-r_f$ trên factors → $r_e = r_f + \sum \beta_k \lambda_k$ | Proposed (discount rate) | 1, 7, 8 |
| Covariance $\Sigma$ | Sample cov của return panel (rolling) | EPO | 1 |
| Expected-return signal $s$ | Historical mean return (rolling) | EPO | 1 |
| ROE (clean-surplus) | $CI^{\text{avail-common}}_t / BV^{\text{common}}_{t-1}$ | Proposed RIM (project) | 4, 5 |
| Net dividends $D_t$ | Dividends + buybacks − issuance | Proposed RIM (CSR) | 6 |

---

## 4. Chi tiết từng model

### 4.1 Proposed — RIM inputs (US GAAP, Clean Surplus)

Nguyên tắc: mọi biến ở dạng **COMMON equity, attributable to PARENT** để khớp giá $P_t$ và thỏa Clean Surplus Relation (CSR):
$$BV_t = BV_{t-1} + \text{Earnings}_t - D_t$$

**Book Value — dạng cần: common equity attributable to parent**
$$
BV^{\text{parent}}_t = \underbrace{BV^{\text{total}}_t}_{\text{gồm NCI}} - \underbrace{NCI_t}_{\text{Minority interest}}, \qquad
\boxed{BV^{\text{common}}_t = BV^{\text{parent}}_t - \text{Preferred}_t}
$$
- AOCI **vẫn nằm trong** $BV^{\text{common}}$ — không strip.
- $BVPS_t = BV^{\text{common}}_t / \text{Shares outstanding}_t$

**Net income → Earnings — dạng cần: comprehensive income available to common**
$$
NI^{\text{parent}}_t = NI^{\text{consolidated}}_t - NI^{\text{NCI}}_t
$$
$$
NI^{\text{avail-common}}_t = NI^{\text{parent}}_t - \text{Preferred dividends}_t
$$
$$
\boxed{CI^{\text{avail-common}}_t = NI^{\text{avail-common}}_t + \underbrace{OCI^{\text{parent}}_t}_{=\Delta AOCI}}
$$

**Residual income:** $RI_t = CI^{\text{avail-common}}_t - r_e\,BV^{\text{common}}_{t-1}$, và $V_0 = BV_0 + \sum RI_t/(1+r_e)^t$.

**Net dividends:** $D_t = \text{Common div} + \text{Buybacks} - \text{Issuance}$, hoặc plug $D_t = CI^{\text{avail-common}}_t - \Delta BV^{\text{common}}_t$.

#### Ví dụ Apple Inc. (6 tháng, kết thúc 28/3/2026) — base case (NCI = 0, Preferred = 0)

| Khoản (triệu USD) | Kỳ này | Kỳ trước (YE 9/2025) |
|---|---|---|
| Net income (6T) | 71,675 | — |
| Total OCI (6T) | 196 | — |
| **Comprehensive income** | **71,871** | — |
| AOCI | (5,375) | (5,571) |
| **Total equity = $BV^{\text{common}}$** | **106,491** | **73,733** |
| Shares outstanding (nghìn) | 14,667,688 | 14,773,260 |

- $OCI = \Delta AOCI = -5{,}375 - (-5{,}571) = +196$ ✓
- $CI = 71{,}675 + 196 = 71{,}871$ ✓
- $BVPS = 106{,}491 / 14{,}667.688 = \$7.26$ (shares đơn vị **nghìn**!)

**Kiểm chứng CSR — tại sao phải dùng CI chứ không NI:**
$\Delta BV = 106{,}491 - 73{,}733 = 32{,}758$; suy $D = CI - \Delta BV = 71{,}871 - 32{,}758 = 39{,}113$.
Nếu dùng **NI**: $71{,}675 - 39{,}113 = 32{,}562 \neq 32{,}758$, thiếu đúng **196 = OCI**. Dùng **CI** khớp chính xác.

#### Bảng tag XBRL EDGAR

| Khái niệm                  | Tag                                                                      | Ghi chú            |
| -------------------------- | ------------------------------------------------------------------------ | ------------------ |
| NI consolidated            | `ProfitLoss`                                                             | KHÔNG dùng cho RIM |
| **NI parent**              | `NetIncomeLoss`                                                          | đã là parent       |
| NI của NCI                 | `NetIncomeLossAttributableToNoncontrollingInterest`                      |                    |
| NI available to common     | `NetIncomeLossAvailableToCommonStockholdersBasic`                        |                    |
| **CI parent**              | `ComprehensiveIncomeNetOfTax`                                            | dùng cho CSR       |
| CI total                   | `...IncludingPortionAttributableToNoncontrollingInterest`                |                    |
| **Total equity**           | `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` |                    |
| **Equity parent**          | `StockholdersEquity`                                                     | đã loại NCI        |
| NCI                        | `MinorityInterest`                                                       |                    |
| Preferred (carrying)       | `PreferredStockValue`                                                    |                    |
| Preferred dividends        | `PreferredStockDividendsAndOtherAdjustments`                             |                    |
| AOCI                       | `AccumulatedOtherComprehensiveIncomeLossNetOfTax`                        |                    |
| Shares outstanding (cover) | `dei:EntityCommonStockSharesOutstanding`                                 | point-in-time      |
| Shares outstanding (BS)    | `us-gaap:CommonStockSharesOutstanding`                                   |                    |
| Weighted-avg shares        | `WeightedAverageNumberOfSharesOutstandingBasic`                          | chỉ cho EPS        |
| Common dividends paid      | `PaymentsOfDividendsCommonStock`                                         | $D_t$              |
| Share repurchases          | `PaymentsForRepurchaseOfCommonStock`                                     | $D_t$              |
| Share issuance             | `ProceedsFromIssuanceOfCommonStock`                                      | $D_t$              |

---

### 4.2 Proposed — FF3/5 cost of equity ($r_e$) + MIRR

- **Cost of equity:** hồi quy time-series excess return của từng firm trên FF factors (nguồn #1 + #7 + #8), lấy $\beta_k$ → $r_e = r_f + \sum_k \beta_k \lambda_k$ (λ = factor premia).
- **MIRR:** kết hợp dòng tiền từ RIM (§4.1) + $r_e$ + giá hiện tại $P_t$ (#2). *Wiring cụ thể (reinvestment/finance rate, horizon T) chưa chốt — xem argument-map.md.*
- Data mới so với RIM: **không có** — chỉ dùng lại return panel + factors + price.

---

### 4.3 Baseline EPO — Pedersen, Babu & Levine (2021)

**Model:** MVO với ma trận tương quan bị shrink:
$$w_{\text{EPO}} \propto \Sigma_{\text{shrunk}}^{-1}\, s, \qquad \Sigma_{\text{shrunk}} = D\big[(1-\omega)\Omega + \omega I\big]D$$
($\Omega$ = correlation, $D$ = diag vol, $\omega$ = shrinkage, $s$ = expected-return signal.)

**Data cần:**

| Input | Lấy từ | Ghi chú |
|---|---|---|
| Covariance $\Sigma$ | Return panel (#1), rolling window (vd 60T) | |
| Expected return $s$ | Historical mean return (#1) | đại diện "camp historical-return" |
| (tùy chọn) anchor portfolio | 1/N hoặc value-weight (#2,#3) | cho anchored EPO |

→ **Không cần data mới.** Tham số $\omega$ (shrinkage), risk aversion = *tune trên train*, không thu thập.

---

### 4.4 Baseline PPP — Brandt, Santa-Clara & Valkanov (2009)

**Model:** trọng số = benchmark + hàm tuyến tính của characteristics chuẩn hóa:
$$w_{i,t} = \bar{w}_{i,t} + \frac{1}{N_t}\,\theta^\top \hat{x}_{i,t}$$
$\theta$ fit bằng **maximize CRRA utility trên realized return**:
$$\max_\theta \frac{1}{T}\sum_t u\!\Big(\textstyle\sum_i w_{i,t}\, r_{i,t+1}\Big)$$

**Ba characteristics gốc $\hat{x}$** (chuẩn hóa cross-section mỗi kỳ):
1. **Size** = log market cap = log(shares × price) — #2, #3
2. **Book-to-market** = BV common-parent ÷ market cap — #2, #3, #4
3. **Momentum** = return t−12..t−2 (skip 1 tháng) — #1

**Data cần:**

| Input | Lấy từ |
|---|---|
| Realized return $r_{i,t+1}$ | #1 |
| Market cap (size) | Shares (#3) × price (#2) |
| Book equity (cho B/M) | BV common-parent (#4) |
| Momentum | Return history (#1) |
| Benchmark weight $\bar{w}$ | value-weight (market cap) hoặc 1/N |

→ Data mới so với EPO: chỉ **book equity** — mà RIM **đã thu** rồi. Chi phí biên ≈ 0.

---

### 4.5 Baseline 1/N + Evaluation

- **1/N:** chỉ cần danh sách constituent (#9) → trọng số bằng nhau.
- **Evaluation metrics:** return panel (#1) + RF (#8) → Sharpe; FF factors (#7) → alpha; benchmark (#10) → so sánh.
- **Bắt buộc (quy tắc bất biến):** out-of-sample (train/test split hoặc rolling), test ý nghĩa (Jobson-Korkie cho Sharpe / bootstrap).

---

## 5. Edge cases / cờ cần xử lý

1. **NCI ≠ 0 / Preferred ≠ 0** (§4.1) — áp đủ 2 bước bóc tách. Apple = 0 nhưng nhiều tập đoàn khác có.
2. **Dirty surplus vượt cả CI** — cumulative-effect adjustment (ASC 606/842/CECL) hoặc restatement ghi thẳng retained earnings → bước nhảy BV không qua CI. Xử lý: cộng vào earnings năm chuyển đổi, hoặc dùng plug $D_t$.
3. **Tension forecast vs CSR** — OCI transitory & nhiễu → **forecast** ROE bằng *core NI* nhưng **reconcile BV bằng CI** để không rò rỉ giá trị. Ghi rõ trong Method.
4. **Đơn vị shares = nghìn** trên face → dễ sai 1000×.
5. **Multi-class shares** (GOOGL/GOOG) — cộng đủ class + match giá từng class.
6. **Survivorship bias** — kéo giá cho cả tên đã delisted; ghi lại tỉ lệ không lấy được; documented limitation.
7. **Ticker ↔ CIK ↔ tên đổi theo thời gian** — phần fiddly nhất về định danh khi ghép EDGAR với giá.

---

## 6. Checklist thu thập (theo nguồn)

**Ken French:** ☐ FF5 factors ☐ Momentum ☐ RF (monthly)
**Membership:** ☐ danh sách constituent theo tháng (GitHub/Wiki)
**yfinance/Stooq:** ☐ monthly adjusted close mọi ticker lịch sử ☐ ghi tên delisted không lấy được
**EDGAR (mỗi firm × quý):**
- ☐ `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`, `MinorityInterest`, `PreferredStockValue` → $BV^{\text{common}}$
- ☐ `NetIncomeLoss`, `ComprehensiveIncomeNetOfTax`, `AccumulatedOtherComprehensiveIncomeLossNetOfTax`, `PreferredStockDividendsAndOtherAdjustments` → $CI^{\text{avail-common}}$
- ☐ `EntityCommonStockSharesOutstanding` (+ mọi class) → market cap, BVPS
- ☐ `PaymentsOfDividendsCommonStock`, `PaymentsForRepurchaseOfCommonStock`, `ProceedsFromIssuanceOfCommonStock` → $D_t$

**Derived (sau khi có raw):** ☐ market cap ☐ B/M ☐ momentum ☐ FF beta/$r_e$ ☐ Σ ☐ signal $s$ ☐ ROE ☐ $D_t$
