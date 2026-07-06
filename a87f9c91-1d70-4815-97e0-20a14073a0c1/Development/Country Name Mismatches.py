
import pandas as pd
from scipy import stats

# ── Inner join: city_qol ✕ happiness_current on standardized country name ─────
city_master = city_qol.merge(
    happiness_current,
    left_on="Country",
    right_on="Country name",
    how="inner"
).drop(columns=["Country name"])

print(f"city_master rows: {len(city_master)}")
print(f"city_qol rows:    {len(city_qol)}")
print(f"Rows dropped by inner join: {len(city_qol) - len(city_master)}")

whr_countries = set(happiness_current["Country name"])
dropped_mask  = ~city_qol["Country"].isin(whr_countries)
dropped_cities = city_qol.loc[dropped_mask, ["City", "Country"]].sort_values("Country").reset_index(drop=True)

if dropped_cities.empty:
    print("\nNo cities were dropped.")
else:
    print(f"\nCities dropped by the join ({len(dropped_cities)} rows):")
    print(dropped_cities.to_string(index=False))

print(f"\ncost_of_living_ts untouched: {cost_of_living_ts.shape}")

# ── Region mapping ─────────────────────────────────────────────────────────────
REGION_MAP = {
    "Canada": "North America", "Mexico": "North America", "United States": "North America",
    "Brazil": "Latin America", "Colombia": "Latin America", "Ecuador": "Latin America",
    "Panama": "Latin America", "Uruguay": "Latin America",
    "Austria": "Europe", "Belgium": "Europe", "Bosnia And Herzegovina": "Europe",
    "Bulgaria": "Europe", "Croatia": "Europe", "Cyprus": "Europe",
    "Czech Republic": "Europe", "Denmark": "Europe", "Estonia": "Europe",
    "Finland": "Europe", "France": "Europe", "Germany": "Europe",
    "Greece": "Europe", "Hungary": "Europe", "Iceland": "Europe",
    "Ireland": "Europe", "Italy": "Europe", "Latvia": "Europe",
    "Lithuania": "Europe", "Luxembourg": "Europe", "Netherlands": "Europe",
    "North Macedonia": "Europe", "Norway": "Europe", "Poland": "Europe",
    "Portugal": "Europe", "Romania": "Europe", "Russia": "Europe",
    "Serbia": "Europe", "Slovakia": "Europe", "Slovenia": "Europe",
    "Spain": "Europe", "Sweden": "Europe", "Switzerland": "Europe",
    "Ukraine": "Europe", "United Kingdom": "Europe",
    "Armenia": "Middle East & Caucasus", "Azerbaijan": "Middle East & Caucasus",
    "Georgia": "Middle East & Caucasus", "Israel": "Middle East & Caucasus",
    "Jordan": "Middle East & Caucasus", "Kuwait": "Middle East & Caucasus",
    "Oman": "Middle East & Caucasus", "Saudi Arabia": "Middle East & Caucasus",
    "Turkey": "Middle East & Caucasus", "United Arab Emirates": "Middle East & Caucasus",
    "Australia": "Asia-Pacific", "China": "Asia-Pacific", "Hong Kong": "Asia-Pacific",
    "India": "Asia-Pacific", "Japan": "Asia-Pacific", "Malaysia": "Asia-Pacific",
    "New Zealand": "Asia-Pacific", "Pakistan": "Asia-Pacific", "Singapore": "Asia-Pacific",
    "South Korea": "Asia-Pacific", "Taiwan": "Asia-Pacific",
    "South Africa": "Africa",
}

city_master = city_master.copy()
city_master["Region"] = city_master["Country"].map(REGION_MAP)

# ── Summary stats ──────────────────────────────────────────────────────────────
STAT_COLS = [
    "Quality of Life Index", "Safety Index", "Health Care Index",
    "Cost of Living Index", "Purchasing Power Index",
    "Life evaluation (3-year average)", "Explained by: Social support",
]
_present = [c for c in STAT_COLS if c in city_master.columns]
_stats = city_master[_present].agg(["mean","median","std","min","max"]).T.round(3)

# ── Region summary ─────────────────────────────────────────────────────────────
_region_summary = (
    city_master.groupby("Region", sort=False)
    .agg(city_count=("City","count"), country_count=("Country","nunique"))
    .sort_values("city_count", ascending=False)
    .reset_index()
)

# ── Spearman correlations ──────────────────────────────────────────────────────
_country_agg = (
    city_master.groupby("Country", as_index=False)
    .agg(
        mean_ppi=("Purchasing Power Index", "mean"),
        mean_coli=("Cost of Living Index", "mean"),
        social_support=("Explained by: Social support", "first"),
    )
)
n = len(_country_agg)
rho_ppi,  p_ppi  = stats.spearmanr(_country_agg["mean_ppi"],  _country_agg["social_support"])
rho_coli, p_coli = stats.spearmanr(_country_agg["mean_coli"], _country_agg["social_support"])

def _sig(p):
    if p < 0.001: return "p < 0.001 ✓✓✓"
    if p < 0.01:  return "p < 0.01  ✓✓"
    if p < 0.05:  return "p < 0.05  ✓"
    return "not significant (p ≥ 0.05)"

# ══════════════════════════════════════════════════════════════════════════════
# Kruskal-Wallis: Safety Index and Health Care Index across Regions
# ══════════════════════════════════════════════════════════════════════════════

REGIONS_ORDERED = [
    "Europe", "North America", "Asia-Pacific",
    "Middle East & Caucasus", "Latin America", "Africa"
]

_region_groups = {r: city_master.loc[city_master["Region"] == r] for r in REGIONS_ORDERED}
_region_n      = {r: len(g) for r, g in _region_groups.items()}

print("\n" + "="*60)
print("KRUSKAL-WALLIS TESTS — across 6 Regions")
print("="*60)
print("\nObservations per region:")
for r, cnt in _region_n.items():
    flag = "  ⚠️  FEWER THAN 5 OBSERVATIONS" if cnt < 5 else ""
    print(f"  {r:<28} n = {cnt}{flag}")

_kw_results = {}
for metric in ["Safety Index", "Health Care Index"]:
    _groups_arrays = [_region_groups[r][metric].values for r in REGIONS_ORDERED]
    _H, _p = stats.kruskal(*_groups_arrays)
    _df_kw = len(REGIONS_ORDERED) - 1
    _medians = {r: round(_region_groups[r][metric].median(), 2) for r in REGIONS_ORDERED}
    _kw_results[metric] = {"H": _H, "p": _p, "df": _df_kw, "medians": _medians}

    print(f"\n{'─'*60}")
    print(f"Test: {metric}")
    print(f"{'─'*60}")
    print(f"  H-statistic          : {_H:.4f}")
    print(f"  Degrees of freedom   : {_df_kw}  (k − 1, k = 6 regions)")
    print(f"  p-value              : {_p:.4f}  →  {_sig(_p)}")
    print(f"\n  Median {metric} per region:")
    for r, med in _medians.items():
        flag = "  ⚠️  n < 5" if _region_n[r] < 5 else ""
        print(f"    {r:<28} {med}{flag}")

print("\n" + "="*60)
print("SUMMARY: Spearman correlations (for reference)")
print("="*60)
print(f"n (countries): {n}")
print(f"PPI  vs Social support : ρ = {rho_ppi:+.4f}  p = {p_ppi:.4f}  →  {_sig(p_ppi)}")
print(f"COLI vs Social support : ρ = {rho_coli:+.4f}  p = {p_coli:.6f}  →  {_sig(p_coli)}")
