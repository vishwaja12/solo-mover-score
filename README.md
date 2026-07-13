# Solo Mover Score

A data-driven scoring system that ranks **148 global cities** for solo international relocation — quantifying affordability, safety, healthcare access, and social support into a single adjustable score.

Built as a Zerve AI project for a HackerEarth Data Analyst/Data Scientist application.

---

## Problem

People moving abroad alone — students, early-career professionals, anyone starting over without a built-in support system — make relocation decisions from scattered blog posts, forum threads, and gut feel. This project replaces that with a reproducible, data-backed score that lets you compare destinations against what actually matters for solo movers.

It also tests two common assumptions:
- "Cheap places are less safe" — does affordability trade off with safety?
- "Affordable places are more isolating" — is purchasing power correlated with social support?

---

## Datasets

| Dataset | Source | Grain |
|---|---|---|
| Quality of Life Indices by City | Numbeo (via Kaggle) | City-level: Safety, Healthcare, Cost of Living, Purchasing Power, Quality of Life indices |
| World Happiness Report 2026 | Gallup / WHR | Country-level: Happiness score, Social Support, GDP/capita, Healthy life expectancy |
| Cost of Living Index by Country (2020–2024) | Numbeo historical archives | Country-level time series for affordability forecasting |

---

## Pipeline

Scripts run in numbered order. Each script consumes variables from the previous one (as they were designed to run sequentially in Zerve).

| # | Script | What it does |
|---|---|---|
| 01 | `scripts/01_load_inspect.py` | Loads all three raw datasets, prints shape, columns, and sample rows |
| 02 | `scripts/02_clean_filter.py` | Filters cost-of-living to 2020–2024, extracts 2025 happiness data, prepares city QoL data |
| 03 | `scripts/03_standardize_countries.py` | Normalizes country names across all three sources via CamelCase splitting, parenthetical removal, and a canonical spelling map |
| 04 | `scripts/04_eda_correlations.py` | Inner-joins city + happiness data, maps cities to regions, runs Spearman correlations (affordability vs. social support) and Kruskal-Wallis tests (safety/healthcare across 6 regions) |
| 05 | `scripts/05_holt_forecasting.py` | Applies Holt exponential smoothing to forecast cost-of-living for 81 countries through 2025–2026; flags directionally unstable forecasts |
| 06 | `scripts/06_clustering.py` | K-means clustering (silhouette-optimized, k=3) on 148 cities across cost, safety, healthcare, and social support |
| 07 | `scripts/07_solo_mover_score.py` | Computes the weighted Solo Mover Score (0–100) per city; produces three preset rankings |

---

## Key Outputs

### City Archetypes (K-Means, k=3)
| Cluster | Label |
|---|---|
| 0 | Moderate Cost, Moderate Everything |
| 1 | Affluent & High-Functioning |
| 2 | Low-Cost, Lower Safety & Social Support |

### Solo Mover Score
A 0–100 composite built from four normalized sub-scores:

```
Score = w₁ × Affordability + w₂ × Safety + w₃ × Healthcare + w₄ × Social Support
```

Default weights are equal (0.25 each). Two preset profiles are also computed:
- **Safety-heavy** — 0.10 / 0.70 / 0.10 / 0.10
- **Social-heavy** — 0.10 / 0.10 / 0.10 / 0.70

### Affordability Forecast
Holt's exponential smoothing forecasts cost-of-living trends (Rising / Flat / Falling) for 81 countries through 2026. Countries where Holt and naive momentum disagree are flagged as `UNSTABLE`.

---

## Repo Structure

```
solo-mover-score/
├── scripts/
│   ├── 01_load_inspect.py
│   ├── 02_clean_filter.py
│   ├── 03_standardize_countries.py
│   ├── 04_eda_correlations.py
│   ├── 05_holt_forecasting.py
│   ├── 06_clustering.py
│   └── 07_solo_mover_score.py
├── data/               # Place raw data files here (not tracked in git)
├── zerve/              # Zerve canvas and layer config (project metadata)
│   ├── canvas.yaml
│   └── layer.yaml
└── README.md
```

### Data files required (not included in repo)
Place these in `data/` before running:
- `quality_of_life_indices_by_country.csv` — Numbeo city QoL data (via Kaggle)
- `WHR26_Data_Figure_2.1.xlsx` — World Happiness Report 2026
- `livable_cities.csv` — Numbeo livable cities dataset

---

## Dependencies

```
pandas
numpy
scipy
scikit-learn
statsmodels
openpyxl
```

Install with:
```bash
pip install pandas numpy scipy scikit-learn statsmodels openpyxl
```

---

## Built With

- [Zerve](https://zerve.ai) — AI-native data science canvas
- Python (pandas, scikit-learn, statsmodels, scipy)
- Datasets: Numbeo, World Happiness Report / Gallup World Poll
