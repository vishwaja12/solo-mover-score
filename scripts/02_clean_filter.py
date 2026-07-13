
import pandas as pd

# ── Step 1: cost_of_living_ts ────────────────────────────────────────────────
# Keep only rows where Year is a plain integer (drop Year containing "/2")
qol_filtered = qol_raw[~qol_raw["Year"].astype(str).str.contains("/")]
# Keep only 2020–2024
qol_filtered = qol_filtered[qol_filtered["Year"].astype(int).between(2020, 2024)]
cost_of_living_ts = qol_filtered.copy()
cost_of_living_ts["Year"] = cost_of_living_ts["Year"].astype(int)

# ── Step 2: happiness_current ────────────────────────────────────────────────
happiness_current = whr_raw[whr_raw["Year"] == 2025].copy()

# ── Step 3: city_qol ─────────────────────────────────────────────────────────
city_qol = cities_raw.copy()

print("cost_of_living_ts:", cost_of_living_ts.shape, "| Years:", sorted(cost_of_living_ts["Year"].unique()))
print("happiness_current:", happiness_current.shape)
print("city_qol:", city_qol.shape)
