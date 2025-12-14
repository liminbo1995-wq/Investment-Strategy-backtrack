import pandas as pd
import numpy as np
import datetime as dt

# ========== 参数 ==========
CSV_PATH = "MacroTrends_Data_Download_QQQ.csv" # 你的文件名/路径
START_REQ = "2000-01-01"
END_REQ   = "2025-12-24"   # 若数据不够，会自动截到最后一个交易日
MONTHLY_CONTRIB = 100.0
FEE_ANNUAL = 0.0098
TRADING_DAYS = 252
LEVERAGE = 2.0

# ========== 读入 QQQ 数据（MacroTrends 前 14 行是声明） ==========
qqq = pd.read_csv(CSV_PATH, skiprows=14)
qqq["date"] = pd.to_datetime(qqq["date"])
qqq = qqq.sort_values("date")

start_req = pd.Timestamp(START_REQ)
end_req = pd.Timestamp(END_REQ)
qqq = qqq[(qqq["date"] >= start_req) & (qqq["date"] <= end_req)].copy()

start_actual = qqq["date"].min()
end_actual = qqq["date"].max()
print(f"Using data range: {start_actual.date()} -> {end_actual.date()} (requested end={end_req.date()})")

# ========== 用 QQQ 合成 QLD：2x 日收益复利 + 日度费率拖累 ==========
prices = qqq.set_index("date")["close"].astype(float)
r = prices.pct_change().fillna(0.0)

fee_factor = np.exp(-FEE_ANNUAL / TRADING_DAYS)
qld_factor = (1.0 + LEVERAGE * r) * fee_factor
qld_factor.iloc[0] = 1.0
qld_price = qld_factor.cumprod()
qld_price.name = "qld_synth"

# ========== XIRR ==========
def xnpv(rate, cashflows):
    t0 = cashflows[0][1]
    return sum(cf / ((1 + rate) ** (((d - t0).days) / 365.0)) for cf, d in cashflows)

def xirr(cashflows):
    low, high = -0.9999, 10.0
    f_low, f_high = xnpv(low, cashflows), xnpv(high, cashflows)
    it = 0
    while f_low * f_high > 0 and it < 50:
        high *= 2
        f_high = xnpv(high, cashflows)
        it += 1
    if f_low * f_high > 0:
        return np.nan
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = xnpv(mid, cashflows)
        if abs(f_mid) < 1e-8:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2

# ========== 定投模拟（按月 high / low / avg 价格成交） ==========
def simulate_dca(qld_price: pd.Series, monthly_contribution=100.0, method="high"):
    idx = qld_price.index
    g = qld_price.groupby([idx.year, idx.month])

    if method == "high":
        monthly_price = g.max()
    elif method == "low":
        monthly_price = g.min()
    elif method in ("avg", "mean", "average"):
        monthly_price = g.mean()
    else:
        raise ValueError("method must be one of: high, low, avg")

    # 月末交易日作为现金流日期
    month_end = g.apply(lambda x: x.index.max())

    # 月键（用于对齐）
    month_key = pd.to_datetime([dt.date(y, m, 1) for (y, m) in monthly_price.index])
    monthly_price.index = month_key
    month_end.index = month_key

    shares_by_month = (monthly_contribution / monthly_price).astype(float)

    inc = pd.Series(shares_by_month.values, index=month_end.values).sort_index()
    cum_shares = inc.cumsum().reindex(qld_price.index, method="ffill").fillna(0.0)
    port_value = cum_shares * qld_price

    cashflows = [(-monthly_contribution, d.to_pydatetime().date()) for d in inc.index]
    cashflows.append((float(port_value.iloc[-1]), qld_price.index[-1].to_pydatetime().date()))

    total_contrib = monthly_contribution * len(shares_by_month)
    final_value = float(port_value.iloc[-1])
    return port_value, cashflows, total_contrib, final_value

def yearly_stats(port_value: pd.Series, monthly_contribution=100.0):
    idx = port_value.index
    g = port_value.groupby([idx.year, idx.month])
    month_end_dates = g.apply(lambda x: x.index.max()).values

    months_per_year = pd.Series(1, index=month_end_dates).groupby(pd.Index(month_end_dates).year).sum()
    contrib_per_year = months_per_year * monthly_contribution

    eoy_value = port_value.groupby(port_value.index.year).apply(lambda x: float(x.iloc[-1]))
    eoy_value.index.name = "year"

    years = eoy_value.index
    start_value = pd.Series(index=years, dtype=float)
    prev = 0.0
    for y in years:
        start_value[y] = prev
        prev = eoy_value[y]
    start_value.index.name = "year"

    profit = eoy_value - start_value - contrib_per_year.reindex(years).fillna(0.0)
    roi = profit / (start_value + contrib_per_year.reindex(years).fillna(0.0)).replace(0, np.nan)

    return pd.DataFrame({
        "start_value": start_value,
        "contrib": contrib_per_year.reindex(years).fillna(0.0),
        "end_value": eoy_value,
        "profit": profit,
        "roi_on_start_plus_contrib": roi
    })

# ========== 跑三种价格口径 ==========
scenarios = ["high", "low", "avg"]
summary = []
yearly_all = None

years_total = (qld_price.index[-1] - qld_price.index[0]).days / 365.0

for m in scenarios:
    pv, cfs, total_contrib, final_value = simulate_dca(qld_price, MONTHLY_CONTRIB, m)
    mw = xirr(cfs)
    total_profit = final_value - total_contrib
    total_return = final_value / total_contrib - 1
    cagr_vs_contrib = (final_value / total_contrib) ** (1 / years_total) - 1

    summary.append([m, total_contrib, final_value, total_profit, total_return, mw, cagr_vs_contrib])

    y = yearly_stats(pv, MONTHLY_CONTRIB).reset_index()
    y = y[["year", "end_value", "profit", "roi_on_start_plus_contrib"]].copy()
    y.columns = ["year", f"{m}_end_value", f"{m}_profit", f"{m}_roi"]
    yearly_all = y if yearly_all is None else yearly_all.merge(y, on="year", how="left")

summary_df = pd.DataFrame(summary, columns=[
    "method","total_contributed","final_value","total_profit","total_return_pct",
    "xirr_money_weighted","cagr_vs_total_contrib"
])

# 输出
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
print("\n=== Summary ===")
print(summary_df)

print("\n=== Yearly (profit & ROI) ===")
print(yearly_all)

# 保存 CSV
summary_df.to_csv("qld_dca_summary.csv", index=False)
yearly_all.to_csv("qld_dca_yearly.csv", index=False)
print("\nSaved: qld_dca_summary.csv, qld_dca_yearly.csv")

