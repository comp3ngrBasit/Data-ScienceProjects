"""Module 3 - Energy usage segmentation.

The UCI dataset is a single household, so households cannot be clustered. The scientifically
justified alternative used here: (a) cluster the household's *daily load profiles* (1,440 days)
to find recurring consumption behaviours, and (b) cluster *national grids* (OPSD countries) by
load-shape features for a regional view. Clusters are named only from their measured statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.config import RANDOM_SEED, SETTINGS
from app.data import ingestion
from app.data import preprocessing as pp
from app.utils.helpers import save_json, save_table

FEATURES = ["daily_kwh", "max_kwh", "min_kwh", "variability_cv", "night_share", "daytime_share",
            "peak_hour_kwh", "load_factor"]


def day_features(hourly: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"v": hourly, "date": hourly.index.normalize(), "hour": hourly.index.hour})
    wide = df.pivot_table(index="date", columns="hour", values="v").dropna()
    n0, n1 = SETTINGS.night_hours
    f = pd.DataFrame(index=wide.index)
    f["daily_kwh"] = wide.sum(axis=1)
    f["max_kwh"] = wide.max(axis=1)
    f["min_kwh"] = wide.min(axis=1)
    f["variability_cv"] = wide.std(axis=1) / wide.mean(axis=1)
    f["night_share"] = wide.loc[:, n0:n1 - 1].sum(axis=1) / f["daily_kwh"]
    f["daytime_share"] = wide.loc[:, 8:17].sum(axis=1) / f["daily_kwh"]
    f["peak_hour_kwh"] = wide.loc[:, 19:22].mean(axis=1)  # household peak window found in Module 2
    f["load_factor"] = wide.mean(axis=1) / wide.max(axis=1)
    f["is_weekend"] = (f.index.dayofweek >= 5).astype(int)
    return f, wide


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    m = labels >= 0
    if len(set(labels[m])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {"silhouette": float(silhouette_score(X[m], labels[m], random_state=RANDOM_SEED)),
            "davies_bouldin": float(davies_bouldin_score(X[m], labels[m])),
            "calinski_harabasz": float(calinski_harabasz_score(X[m], labels[m]))}


def stability(X: np.ndarray, k: int, n_boot: int = 30) -> dict:
    """Bootstrap stability: refit K-Means on resamples, compare to reference labels (ARI) on the resampled points."""
    rng = np.random.default_rng(RANDOM_SEED)
    ref = KMeans(k, n_init=10, random_state=RANDOM_SEED).fit(X)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        km = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
        aris.append(adjusted_rand_score(ref.labels_[idx], km.labels_))
    return {"mean_ari": float(np.mean(aris)), "std_ari": float(np.std(aris)), "n_bootstrap": n_boot}


def name_clusters(prof: pd.DataFrame, weekend_base: float, sizes: pd.Series, weekend_counts: pd.Series) -> dict:
    """Name each cluster from standardised centroid deviations (|z| > 0.6) and weekend over-representation."""
    z = (prof[FEATURES] - prof[FEATURES].mean()) / prof[FEATURES].std(ddof=0)
    rules = [("daily_kwh", "High-consumption", "Low-consumption"), ("night_share", "Night-time intensive", None),
             ("daytime_share", "Daytime intensive", None), ("variability_cv", "High-variability", "Stable"),
             ("min_kwh", "High base-load", None), ("peak_hour_kwh", "Evening-peak heavy", None)]
    names = {}
    for c in prof.index:
        tags = []
        for feat, hi, lo in rules:
            if z.loc[c, feat] > 0.6 and hi:
                tags.append((abs(z.loc[c, feat]), hi))
            elif z.loc[c, feat] < -0.6 and lo:
                tags.append((abs(z.loc[c, feat]), lo))
        p = stats.binomtest(int(weekend_counts[c]), int(sizes[c]), weekend_base, alternative="greater").pvalue
        if p < 0.01 and weekend_counts[c] / sizes[c] > 1.3 * weekend_base:
            tags.append((9, "Weekend-intensive"))
        tags = [t for _, t in sorted(tags, reverse=True)][:3]
        names[c] = " / ".join(tags) if tags else "Typical"
    # make names unique
    seen = {}
    for c, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def household_segmentation() -> dict:
    hh = pp.load_household_hourly()
    hh = hh.loc[~hh["imputed"], "energy_kwh"]  # imputed hours would distort profile shape
    feats, wide = day_features(hh)
    X = StandardScaler().fit_transform(feats[FEATURES])
    pca = PCA(random_state=RANDOM_SEED).fit(X)
    pcs = pca.transform(X)[:, :3]
    selection = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=RANDOM_SEED).fit(X)
        gmm = GaussianMixture(k, covariance_type="full", random_state=RANDOM_SEED, n_init=3).fit(X)
        selection.append({"k": k, "kmeans_inertia": float(km.inertia_), "gmm_bic": float(gmm.bic(X)),
                          **{f"kmeans_{m}": v for m, v in evaluate(X, km.labels_).items()}})
    sel = pd.DataFrame(selection)
    # choose k: best silhouette among k>=3 within 0.02 of the max (favour interpretability over k=2 splits)
    cand = sel[sel["k"] >= 3]
    k = int(cand.loc[cand["kmeans_silhouette"].idxmax(), "k"])
    labels = {"KMeans": KMeans(k, n_init=20, random_state=RANDOM_SEED).fit_predict(X),
              "Hierarchical (Ward)": AgglomerativeClustering(k, linkage="ward").fit_predict(X)}
    gmm_k = int(sel.loc[sel["gmm_bic"].idxmin(), "k"])
    labels["GaussianMixture"] = GaussianMixture(gmm_k, covariance_type="full", random_state=RANDOM_SEED, n_init=3).fit_predict(X)
    nn = NearestNeighbors(n_neighbors=2 * X.shape[1]).fit(X)
    kd = np.sort(nn.kneighbors(X)[0][:, -1])
    eps = float(np.quantile(kd, 0.95))  # k-distance "knee" approximated at the 95th percentile
    labels["DBSCAN"] = DBSCAN(eps=eps, min_samples=2 * X.shape[1]).fit_predict(X)
    comp = []
    for algo, lab in labels.items():
        comp.append({"algorithm": algo, "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
                     "noise_share": float(np.mean(lab == -1)), **evaluate(X, lab),
                     "ari_vs_kmeans": float(adjusted_rand_score(labels["KMeans"], lab))})
    comp = pd.DataFrame(comp)
    stab = stability(X, k)
    feats["cluster"] = labels["KMeans"]
    prof = feats.groupby("cluster")[FEATURES + ["is_weekend"]].mean()
    sizes = feats.groupby("cluster").size()
    names = name_clusters(prof, 2 / 7, sizes, feats.groupby("cluster")["is_weekend"].sum())
    prof["n_days"] = sizes
    prof["weekend_day_share"] = prof.pop("is_weekend")
    prof["name"] = prof.index.map(names)
    prof["months_most_common"] = feats.groupby("cluster").apply(
        lambda d: ", ".join(pd.Series(d.index.month_name().str[:3]).value_counts().index[:3]), include_groups=False)
    feats["cluster_name"] = feats["cluster"].map(names)
    feats[["pc1", "pc2", "pc3"]] = pcs
    for algo, lab in labels.items():
        feats[f"label_{algo}"] = lab
    curves = wide.join(feats["cluster"]).groupby("cluster").mean().T
    curves.columns = [names[c] for c in curves.columns]
    loadings = pd.DataFrame(pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
    save_table(feats.reset_index(names="date"), "segmentation", "household_days")
    save_table(prof.reset_index(), "segmentation", "household_profiles")
    save_table(curves.reset_index(names="hour"), "segmentation", "household_cluster_curves")
    save_table(sel, "segmentation", "household_k_selection")
    save_table(comp, "segmentation", "household_algorithm_comparison")
    save_table(loadings.reset_index(names="feature"), "segmentation", "household_pca_loadings")
    return {"unit_of_analysis": "household-day (24-hour load profile)", "n_days": int(len(feats)), "chosen_k": k,
            "gmm_k_by_bic": gmm_k, "dbscan_eps": eps, "stability": stab, "features": FEATURES,
            "pca_explained_variance": pca.explained_variance_ratio_[:3].tolist(),
            "cluster_names": {int(c): n for c, n in names.items()}}


def grid_segmentation() -> dict:
    cols = ingestion.opsd_columns()
    countries = sorted({c.split("_load_actual")[0] for c in cols if c.endswith("_load_actual_entsoe_transparency")})
    # national series only (GB_UKM duplicates GB_GBN plus Northern Ireland, so GB_GBN is kept)
    countries = [c for c in countries if c.count("_") == 0 or c in ("GB_GBN", "IE_SEM")]
    usecols = ["utc_timestamp"] + [f"{c}_load_actual_entsoe_transparency" for c in countries]
    df = ingestion.load_opsd_raw(usecols=usecols).loc["2016-01-01":"2019-12-31"]
    rows, shapes = [], {}
    for c in countries:
        s = df[f"{c}_load_actual_entsoe_transparency"]
        if s.notna().mean() < 0.9:
            continue
        s = s.interpolate(limit=6).dropna()
        loc = s.tz_convert("Europe/Brussels")  # CET is a common clock for all; shapes compared on one clock
        x = loc / loc.mean()
        hours, dow, month = loc.index.hour, loc.index.dayofweek, loc.index.month
        winter, summer = x[month.isin([12, 1, 2])].mean(), x[month.isin([6, 7, 8])].mean()
        rows.append({"country": c, "mean_load_mw": float(s.mean()), "cv": float(s.std() / s.mean()),
                     "load_factor": float(s.mean() / s.max()), "night_share": float(x[hours < 6].sum() / x.sum()),
                     "weekend_ratio": float(x[dow >= 5].mean() / x[dow < 5].mean()),
                     "winter_summer_ratio": float(winter / summer),
                     "peak_hour": int(x.groupby(hours).mean().idxmax()),
                     "daily_amplitude": float(x.groupby(hours).mean().max() - x.groupby(hours).mean().min())})
        shapes[c] = x.groupby(hours).mean()
    feat = pd.DataFrame(rows).set_index("country")
    fcols = ["cv", "load_factor", "night_share", "weekend_ratio", "winter_summer_ratio", "daily_amplitude"]
    X = StandardScaler().fit_transform(feat[fcols])
    Z = linkage(X, "ward")
    sel = []
    for k in range(2, min(7, len(feat) - 1)):
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
        sel.append({"k": k, **evaluate(X, lab)})
    sel = pd.DataFrame(sel)
    k = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    feat["cluster"] = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    feat["kmeans_cluster"] = KMeans(k, n_init=20, random_state=RANDOM_SEED).fit_predict(X)
    p2 = PCA(2, random_state=RANDOM_SEED).fit_transform(X)
    feat["pc1"], feat["pc2"] = p2[:, 0], p2[:, 1]
    dn = dendrogram(Z, labels=feat.index.tolist(), no_plot=True)
    save_table(feat.reset_index(), "segmentation", "grid_countries")
    save_table(pd.DataFrame(shapes).reset_index(names="hour"), "segmentation", "grid_country_shapes")
    save_table(sel, "segmentation", "grid_k_selection")
    return {"unit_of_analysis": "national grid (OPSD country), 2016-2019", "n_countries": int(len(feat)), "chosen_k": k,
            "features": fcols, "dendrogram_order": dn["ivl"],
            "kmeans_vs_hierarchical_ari": float(adjusted_rand_score(feat["cluster"], feat["kmeans_cluster"]))}


def run() -> dict:
    res = {"household": household_segmentation(), "grid": grid_segmentation()}
    save_json(res, "segmentation", "summary")
    return res
