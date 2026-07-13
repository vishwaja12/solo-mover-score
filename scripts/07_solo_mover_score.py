import pandas as pd
import numpy as np

# ── 1. Pull city_master from upstream ──
df = city_master.copy()

# ── 2. Min-max normalize to 0–100 scale (100 = best) ──
def _minmax(series):
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) * 100

df['PPI_scaled']           = _minmax(df['Purchasing Power Index'])
df['Safety_scaled']        = _minmax(df['Safety Index'])
df['HealthCare_scaled']    = _minmax(df['Health Care Index'])
df['SocialSupport_scaled'] = _minmax(df['Explained by: Social support'])

# ── 3. Archetype column ──
ARCHETYPE_MAP = {
    0: 'Moderate Cost, Moderate Everything',
    1: 'Affluent & High-Functioning',
    2: 'Low-Cost, Lower Safety & Social Support',
}
df['Archetype'] = df['Cluster'].map(ARCHETYPE_MAP)

# ── 4. Forecast_Outlook column ──
AGREE_MAP = {
    'Bangladesh':             'Falling',
    'Bosnia And Herzegovina': 'Flat',
    'Ecuador':                'Falling',
    'Germany':                'Flat',
    'Indonesia':              'Falling',
    'Jordan':                 'Falling',
    'Latvia':                 'Rising',
    'Panama':                 'Falling',
    'Peru':                   'Falling',
    'Sweden':                 'Falling',
    'Thailand':               'Falling',
}
df['Forecast_Outlook'] = df['Country'].map(AGREE_MAP).fillna('Uncertain')

# ── 5. Default Solo_Mover_Score (equal weights) on city_master ──
def compute_score(d, w_afford, w_safety, w_health, w_social):
    return (
        w_afford * d['PPI_scaled'] +
        w_safety * d['Safety_scaled'] +
        w_health * d['HealthCare_scaled'] +
        w_social * d['SocialSupport_scaled']
    ).round(2)

df['Solo_Mover_Score'] = compute_score(df, 0.25, 0.25, 0.25, 0.25)
city_master = df

DISPLAY_COLS = [
    'City', 'Country', 'Region', 'Archetype', 'Forecast_Outlook',
    'Solo_Mover_Score',
    'PPI_scaled', 'Safety_scaled', 'HealthCare_scaled', 'SocialSupport_scaled',
]
_scaled_cols = ['Solo_Mover_Score', 'PPI_scaled', 'Safety_scaled', 'HealthCare_scaled', 'SocialSupport_scaled']

FEAT_COLS = ['City', 'Country', 'Region', 'Archetype', 'Forecast_Outlook',
             'PPI_scaled', 'Safety_scaled', 'HealthCare_scaled', 'SocialSupport_scaled']

# ── 6. Verification: two fresh vectorized computes, no helper functions ──
# Preset A: Safety-heavy  0.10 / 0.70 / 0.10 / 0.10
preset_a = city_master.copy()
preset_a['Score'] = (
    0.10 * preset_a['PPI_scaled'] +
    0.70 * preset_a['Safety_scaled'] +
    0.10 * preset_a['HealthCare_scaled'] +
    0.10 * preset_a['SocialSupport_scaled']
).round(2)
top15_a = (
    preset_a[FEAT_COLS + ['Score']]
    .sort_values('Score', ascending=False)
    .head(15)
    .reset_index(drop=True)
)
top15_a.index += 1

# Preset B: Social-heavy  0.10 / 0.10 / 0.10 / 0.70
preset_b = city_master.copy()
preset_b['Score'] = (
    0.10 * preset_b['PPI_scaled'] +
    0.10 * preset_b['Safety_scaled'] +
    0.10 * preset_b['HealthCare_scaled'] +
    0.70 * preset_b['SocialSupport_scaled']
).round(2)
top15_b = (
    preset_b[FEAT_COLS + ['Score']]
    .sort_values('Score', ascending=False)
    .head(15)
    .reset_index(drop=True)
)
top15_b.index += 1

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

print("=" * 140)
print("PRESET A — Safety-heavy: Afford=0.10 | Safety=0.70 | Health=0.10 | Social=0.10")
print("Score = 0.10*PPI_scaled + 0.70*Safety_scaled + 0.10*HealthCare_scaled + 0.10*SocialSupport_scaled")
print("=" * 140)
print(top15_a.to_string())

print()
print("=" * 140)
print("PRESET B — Social-heavy: Afford=0.10 | Safety=0.10 | Health=0.10 | Social=0.70")
print("Score = 0.10*PPI_scaled + 0.10*Safety_scaled + 0.10*HealthCare_scaled + 0.70*SocialSupport_scaled")
print("=" * 140)
print(top15_b.to_string())

# ── 7. Expose preset tables ──
preset1_table = (
    city_master.assign(Solo_Mover_Score=compute_score(city_master, 0.25, 0.25, 0.25, 0.25))
    [DISPLAY_COLS].sort_values('Solo_Mover_Score', ascending=False).head(15).reset_index(drop=True)
)
preset1_table.index += 1
preset2_table = top15_a.rename(columns={'Score': 'Solo_Mover_Score'})
preset3_table = top15_b.rename(columns={'Score': 'Solo_Mover_Score'})
