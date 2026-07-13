import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLUSTER_COLS = [
    'Cost of Living Index',
    'Purchasing Power Index',
    'Safety Index',
    'Health Care Index',
    'Explained by: Social support'
]

X_raw = city_master[CLUSTER_COLS].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

sil_scores = {}
for k in range(3, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil

best_k = max(sil_scores, key=sil_scores.get)

km_final = KMeans(n_clusters=best_k, n_init=100, random_state=42)
city_master['Cluster'] = km_final.fit_predict(X_scaled)

# cluster centroids in original units
centroids_scaled = km_final.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
cluster_centroids = pd.DataFrame(centroids_orig, columns=CLUSTER_COLS)
cluster_centroids.index.name = 'Cluster'
cnt = city_master.groupby('Cluster').size().rename('City Count')
cluster_centroids = cluster_centroids.join(cnt).round(2)

cluster_assignments = city_master[
    ['City', 'Country', 'Region', 'Cluster'] + CLUSTER_COLS
].sort_values(['Cluster', 'Country', 'City']).reset_index(drop=True)

# ── RAW PRINTS ──────────────────────────────────────────────────────────────

print("=== cluster_centroids ===")
print(cluster_centroids.to_string())

print()
print("=== cluster_assignments['Cluster'].value_counts() ===")
print(cluster_assignments['Cluster'].value_counts().to_string())

for c in sorted(cluster_assignments['Cluster'].unique()):
    print()
    print(f"=== Cluster {c} — City/Country list ===")
    print(cluster_assignments[cluster_assignments['Cluster'] == c][['City', 'Country']].to_string())
