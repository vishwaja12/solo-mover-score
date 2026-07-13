
import re
import pandas as pd

# ── Utility: split CamelCase-concatenated names ──────────────────────────────
def split_camel(name: str) -> str:
    """Insert spaces before uppercase letters that follow lowercase ones,
    e.g. SouthKorea -> South Korea, BosniaAndHerzegovina -> Bosnia And Herzegovina."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

# ── Utility: clean a single country string ───────────────────────────────────
def clean_name(name: str) -> str:
    name = str(name).strip()
    name = split_camel(name)
    # Remove parenthetical suffixes like "(China)" in "HongKong(China)"
    name = re.sub(r"\s*\(.*?\)", "", name)
    # Collapse multiple spaces
    name = re.sub(r" {2,}", " ", name).strip()
    # Title case
    name = name.title()
    return name

# ── Canonical mapping: all source variants -> one shared spelling ─────────────
# Strategy: use the most common/recognizable English form.
CANONICAL = {
    # WHR-specific long-forms
    "Republic Of Korea":           "South Korea",
    "Republic Of Moldova":         "Moldova",
    "Russian Federation":          "Russia",
    "Hong Kong Sar Of China":      "Hong Kong",
    "Taiwan Province Of China":    "Taiwan",
    "Türkiye":                     "Turkey",
    "Viet Nam":                    "Vietnam",
    "Czechia":                     "Czech Republic",
    "Lao Pdr":                     "Laos",
    "State Of Palestine":          "Palestine",
    "Dr Congo":                    "Dem. Rep. Congo",
    "Côte D'Ivoire":               "Ivory Coast",
    # livable_cities camelCase residuals (after split_camel)
    "Hong Kong(China)":            "Hong Kong",   # before parens removal
    "United States":               "United States",
    # QoL uses "Bosnia And Herzegovina" — WHR has "Bosnia and Herzegovina"
    # After title-casing WHR's value becomes "Bosnia And Herzegovina" — same, fine.
    # Make explicit just in case
    "Bosnia And Herzegovina":      "Bosnia And Herzegovina",
}

def standardize(name: str) -> str:
    cleaned = clean_name(name)
    return CANONICAL.get(cleaned, cleaned)

# ── Apply to each dataframe ───────────────────────────────────────────────────
cost_of_living_ts = cost_of_living_ts.copy()
cost_of_living_ts["Country"] = cost_of_living_ts["Country"].map(standardize)

happiness_current = happiness_current.copy()
happiness_current["Country name"] = happiness_current["Country name"].map(standardize)

city_qol = city_qol.copy()
city_qol["Country"] = city_qol["Country"].map(standardize)

# ── Verification: print unique country names per source ──────────────────────
print("cost_of_living_ts countries:", sorted(cost_of_living_ts["Country"].unique()))
print("\nhappiness_current countries:", sorted(happiness_current["Country name"].unique()))
print("\ncity_qol countries:", sorted(city_qol["Country"].unique()))
