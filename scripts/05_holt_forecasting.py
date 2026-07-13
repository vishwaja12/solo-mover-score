
import pandas as pd
import numpy as np

# ── Holt's linear exponential smoothing is in statsmodels ────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS = True
except ImportError:
    STATSMODELS = False

# ── 1. Data completeness check ───────────────────────────────────────────────
ALL_YEARS = {2020, 2021, 2022, 2023, 2024}
_country_years = (
    cost_of_living_ts
    .groupby("Country")["Year"]
    .apply(set)
)
_year_counts = _country_years.apply(len)

print("=== Data completeness per country ===")
print(f"All 5 years (2020–2024): {(_year_counts == 5).sum()} countries")
print(f"4 of 5 years:            {(_year_counts == 4).sum()} countries")
print(f"3 of 5 years:            {(_year_counts == 3).sum()} countries")
print(f"Fewer than 3 years:      {(_year_counts  < 3).sum()} countries")
print(f"Total countries:         {len(_year_counts)}")
print(f"\nCountries with >=4 years (eligible for Holt): {(_year_counts >= 4).sum()}")
print()

# ── 2. Fit Holt + naive per eligible country ─────────────────────────────────
_eligible = _year_counts[_year_counts >= 4].index.tolist()

def _manual_holt(vals):
    _a, _b = 0.3, 0.1
    _l = vals[0]
    _t = np.mean(np.diff(vals[:min(4, len(vals))]))
    for _v in vals[1:]:
        _lp, _tp = _l, _t
        _l = _a * _v + (1 - _a) * (_lp + _tp)
        _t = _b * (_l - _lp) + (1 - _b) * _tp
    return _l + _t, _l + 2 * _t

_rows = []
for _c in sorted(_eligible):
    _ts = (
        cost_of_living_ts[cost_of_living_ts["Country"] == _c]
        [["Year", "Cost of Living Index"]]
        .drop_duplicates("Year").sort_values("Year")
        .set_index("Year")["Cost of Living Index"]
    )
    _ts = _ts.reindex(pd.Index(range(2020,2025),name="Year")).interpolate("linear").bfill().ffill()
    _a24, _a23 = float(_ts[2024]), float(_ts[2023])

    if STATSMODELS:
        _fc = ExponentialSmoothing(_ts.values, trend="add").fit(optimized=True).forecast(2)
        _h25, _h26 = float(_fc[0]), float(_fc[1])
    else:
        _h25, _h26 = _manual_holt(_ts.values)

    def _dir(d): return "Flat" if abs(d) < 0.5 else ("Rising" if d > 0 else "Falling")
    _ht = _dir(_h25 - _a24)
    _step = _a24 - _a23
    _n25, _n26 = _a24 + _step, _a24 + 2*_step
    _nt = _dir(_n25 - _a24)
    _ag = "AGREE" if _ht == _nt else "DISAGREE"

    _rows.append({
        "Country": _c, "2024 Actual": round(_a24,2),
        "2025 Forecast": round(_h25,2), "2026 Forecast": round(_h26,2), "Holt Trend": _ht,
        "2025 Naive": round(_n25,2),    "2026 Naive": round(_n26,2),    "Naive Trend": _nt,
        "Agreement": _ag, "Direction": _ht if _ag=="AGREE" else "UNSTABLE",
    })

holt_forecasts = pd.DataFrame(_rows)
holt_forecasts.index = range(1, len(holt_forecasts)+1)

# ── 3. Print full table — split into two halves to beat 5k char truncation ───
_cmp = holt_forecasts[["Country","2024 Actual","Holt Trend","Naive Trend","Agreement","Direction"]]

def _print_rows(df, title):
    print(title)
    print(f"{'#':>2}  {'Country':<26} {'Actual':>6}  {'Holt':>8}  {'Naive':>8}  {'Agree?':>9}  {'Direction':>9}")
    print("-"*76)
    for _idx, _row in df.iterrows():
        print(f"{_idx:>2}  {_row['Country']:<26} {_row['2024 Actual']:>6.2f}  {_row['Holt Trend']:>8}  {_row['Naive Trend']:>8}  {_row['Agreement']:>9}  {_row['Direction']:>9}")
    print()

_print_rows(_cmp.iloc[:41],  "=== Trend Direction Comparison: Holt vs Naive Momentum (1–41) ===")
_print_rows(_cmp.iloc[41:],  "=== Trend Direction Comparison: Holt vs Naive Momentum (42–81) ===")

_na = (holt_forecasts["Agreement"]=="AGREE").sum()
_nd = (holt_forecasts["Agreement"]=="DISAGREE").sum()
print(f"Total: {len(holt_forecasts)}  |  AGREE: {_na} ({100*_na/len(holt_forecasts):.1f}%)  |  DISAGREE: {_nd} ({100*_nd/len(holt_forecasts):.1f}%)")
print()

# ── 4. DISAGREE detail ────────────────────────────────────────────────────────
_dis = holt_forecasts[holt_forecasts["Agreement"]=="DISAGREE"][
    ["Country","2024 Actual","2025 Forecast","Holt Trend","2025 Naive","Naive Trend"]]
print(f"=== DISAGREE countries ({len(_dis)} directionally unstable) ===")
print(f"{'#':>2}  {'Country':<26} {'Actual':>6}  {'Holt25':>7}  {'HoltDir':>8}  {'Naive25':>7}  {'NaiveDir':>9}")
print("-"*76)
for _idx, _row in _dis.iterrows():
    print(f"{_idx:>2}  {_row['Country']:<26} {_row['2024 Actual']:>6.2f}  {_row['2025 Forecast']:>7.2f}  {_row['Holt Trend']:>8}  {_row['2025 Naive']:>7.2f}  {_row['Naive Trend']:>9}")
