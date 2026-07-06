
import pandas as pd

# ── Load raw files ──────────────────────────────────────────────────────────
qol_raw    = pd.read_csv("quality_of_life_indices_by_country.csv")
whr_raw    = pd.read_excel("WHR26_Data_Figure_2.1.xlsx")
cities_raw = pd.read_csv("livable_cities.csv")

print("=== quality_of_life_indices_by_country.csv ===")
print("Shape:", qol_raw.shape)
print("Columns:", qol_raw.columns.tolist())
print("Year dtype:", qol_raw["Year"].dtype if "Year" in qol_raw.columns else "NO YEAR COL")
print("Year unique (first 30):", sorted(qol_raw["Year"].astype(str).unique())[:30])
_country_col_qol = [c for c in qol_raw.columns if "country" in c.lower() or "nation" in c.lower()]
print("Country col candidates:", _country_col_qol)
print(qol_raw.head(5).to_string())

print("\n=== WHR26_Data_Figure_2.1.xlsx ===")
print("Shape:", whr_raw.shape)
print("Columns:", whr_raw.columns.tolist())
_year_cols_whr = [c for c in whr_raw.columns if "year" in str(c).lower()]
print("Year col candidates:", _year_cols_whr)
if _year_cols_whr:
    print("Year unique:", whr_raw[_year_cols_whr[0]].unique())
_country_col_whr = [c for c in whr_raw.columns if "country" in str(c).lower() or "nation" in str(c).lower()]
print("Country col candidates:", _country_col_whr)
print(whr_raw.head(5).to_string())

print("\n=== livable_cities.csv ===")
print("Shape:", cities_raw.shape)
print("Columns:", cities_raw.columns.tolist())
_country_col_city = [c for c in cities_raw.columns if "country" in c.lower() or "nation" in c.lower()]
print("Country col candidates:", _country_col_city)
print(cities_raw.head(5).to_string())
