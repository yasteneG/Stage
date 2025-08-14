#!/usr/bin/env python3
# auto_cluster_csv.py
# --------------------------------------------------------------
# pip install pandas numpy liac-arff scikit-learn hdbscan umap-learn tqdm
# --------------------------------------------------------------
import os, argparse, itertools, warnings, concurrent.futures as cf
from pathlib import Path
from functools import partial
import numpy as np, pandas as pd, arff, umap, hdbscan, matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.cluster        import (KMeans, MiniBatchKMeans, SpectralClustering,
                                    AgglomerativeClustering, AffinityPropagation,
                                    MeanShift, OPTICS, DBSCAN)
from sklearn.mixture        import GaussianMixture
from sklearn.metrics        import (silhouette_score, davies_bouldin_score,
                                    calinski_harabasz_score)
from sklearn.impute         import SimpleImputer
from sklearn.ensemble       import RandomForestRegressor, RandomForestClassifier
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
def load_real_csv(real_csv, runs_arff=None):
    X = pd.read_csv(real_csv)
    X.columns = X.columns.str.strip()
    X.rename(columns={"name": "instance"}, inplace=True, errors="ignore")
    X['instance'] = X['instance'].astype(str)
    if runs_arff and Path(runs_arff).exists():
        raw = arff.load(open(runs_arff, 'r'))
        alg_df = pd.DataFrame(raw["data"],
                              columns=[a[0] for a in raw["attributes"]])
        alg_df.rename(columns={"instance_id": "instance"}, inplace=True)
        num_cols = ["violations", "cost", "duration_ms"]
        alg_df[num_cols] = alg_df[num_cols].apply(pd.to_numeric, errors="coerce")
        alg_metrics = alg_df.groupby("instance")[num_cols].mean().reset_index()
        X = X.merge(alg_metrics, on="instance", how="left")
    X.set_index('instance', inplace=True)
    return X

def clean_and_pca(X, keep_var=0.95):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    X_imp = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X),
                         columns=X.columns, index=X.index)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=keep_var, svd_solver='full', random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    return X_imp, X_pca

# ------------------ grille modèles (compacte) ---------------------------
def model_grid():
    grids = []
    for k in range(2, 16):
        grids.append(("kmeans", KMeans(k, n_init='auto', random_state=0), {'k':k}))
    for k in range(2, 16):
        grids.append(("agglo", AgglomerativeClustering(k, linkage='ward'),
                      {'k':k,'link':'ward'}))
    for k in range(2, 16):
        gm = GaussianMixture(k, covariance_type='full', random_state=0)
        grids.append(("gmm", gm, {'k':k,'cov':'full'}))
    for xi in (0.005, 0.01, 0.02, 0.03, 0.05):
        for ms in (3, 5, 10):
            grids.append(("optics", OPTICS(min_samples=ms, xi=xi),
                          {'xi':xi,'min_samples':ms}))
    for mcs in (3,5,10):
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=2)
        grids.append(("hdbscan", hdb, {'min_cluster_size':mcs,'min_samples':2}))
    return grids

# ------------------ scoring & sélection ----------------------------------
def evaluate(model_tuple, X_pca):
    name, model, params = model_tuple
    try:
        labels = (model.fit_predict(X_pca) if hasattr(model,"fit_predict")
                  else model.fit(X_pca).labels_)
    except Exception:
        return None
    core = labels != -1
    if core.sum() < 2 or len(set(labels[core])) < 2:
        return None
    cov   = core.mean()
    sil   = silhouette_score(X_pca[core], labels[core])
    db    = davies_bouldin_score(X_pca[core], labels[core])
    ch    = calinski_harabasz_score(X_pca[core], labels[core])
    return {'name':name, **params, 'coverage':cov,
            'silhouette':sil,'db':db,'ch':ch,'labels':labels}

def pick_best(results, min_cov=0.60):
    df = pd.DataFrame(r for r in results if r)
    df = df[df.coverage >= min_cov]
    if df.empty:
        return None
    norm = lambda s: (s-s.min())/(s.max()-s.min()+1e-9)
    df['score'] = (norm(df.silhouette) + norm(df.ch)
                   + (1-norm(df.db)) + 2*df.coverage)
    return df.sort_values(['score','coverage'], ascending=False).iloc[0]

# ------------------ importance (intra‑ & inter‑clusters) -----------------
def intra_cluster_features(X_df, labels, top_n):
    rows=[]
    for c in sorted(set(labels)-{-1}):
        mask = labels==c
        if mask.sum()<3: continue
        rf = RandomForestRegressor(200, n_jobs=-1, random_state=0)
        rf.fit(X_df[mask], np.arange(mask.sum()))
        imp = pd.Series(rf.feature_importances_, X_df.columns)
        for f,s in imp.sort_values(ascending=False).head(top_n).items():
            rows.append({'cluster':int(c),'feature':f,'importance':round(s,6)})
    return pd.DataFrame(rows).sort_values(['cluster','importance'])

def inter_cluster_features(X_df, labels, top_n):
    mask = labels != -1
    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        n_jobs=-1, random_state=0)
    rf.fit(X_df[mask], labels[mask])
    imp = pd.Series(rf.feature_importances_, index=X_df.columns)
    topk = imp.sort_values(ascending=False).head(top_n)
    return (topk.reset_index()
                .rename(columns={'index':'feature', 0:'importance'}))

# ------------------ pipeline principal ----------------------------------
def main(real_csv, runs_arff, outdir, top_n, workers):
    os.makedirs(outdir, exist_ok=True)
    X = load_real_csv(real_csv, runs_arff)
    X_imp, X_pca = clean_and_pca(X)
    print(f"[INFO] PCA -> {X_pca.shape[1]} composantes (≥95 % var.)")

    # sauvegardes pour diagnostics
    np.save(Path(outdir)/'X_pca.npy',   X_pca)
    X_imp.to_csv(Path(outdir)/'X_clean.csv')

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(partial(evaluate, X_pca=X_pca), model_grid()))
    best = pick_best(res)
    if best is None:
        print("❌  Aucun clustering valide avec la grille actuelle.")
        return

    print(f"🏆  Meilleure config : {best['name']} | "
          f"coverage {best.coverage:.2%} | sil {best.silhouette:.3f}")

    labels = best.labels
    pd.Series(labels, index=X_imp.index, name='cluster')\
        .to_csv(Path(outdir)/'best_cluster_assignment.csv')

    # ---------- importance INTRA‑cluster ---------------------------
    feats_intra = intra_cluster_features(X_imp, labels, top_n)
    feats_intra.to_csv(Path(outdir)/'best_cluster_features.csv',
                       index=False)

    # ---------- importance INTER‑cluster ---------------------------
    feats_inter = inter_cluster_features(X_imp, labels, top_n)
    feats_inter.to_csv(Path(outdir)/'best_cluster_features_inter.csv',
                       index=False)

    # ---------- log grille (sans labels) ---------------------------
    pd.DataFrame([r for r in res if r]).drop(columns='labels', errors='ignore')\
        .to_csv(Path(outdir)/'all_cluster_scores.csv', index=False)

    # ---------- UMAP visuel ---------------------------------------
    try:
        reducer = umap.UMAP(random_state=0)
        um = reducer.fit_transform(X_pca)
        cmap = plt.cm.get_cmap('tab20', len(set(labels)))
        colors=[cmap(l%20) if l!=-1 else (0.5,0.5,0.5,.3) for l in labels]
        plt.figure(figsize=(7,6))
        plt.scatter(um[:,0],um[:,1],s=8,c=colors,lw=0)
        plt.title(f"UMAP – clusters ({best['name']})")
        plt.tight_layout()
        plt.savefig(Path(outdir)/'best_clusters_umap.png', dpi=300)
        plt.close()
    except Exception as e:
        print("[WARN] UMAP plot skipped:", e)

    print("✔  Sorties écrites dans", outdir)

# ------------------ CLI --------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Exploration automatique de clustering sur real.csv")
    p.add_argument('--real', default='data/real.csv',
                   help='CSV des instances réelles')
    p.add_argument('--runs', default='algorithm_runs.arff',
                   help='ARFF algorithm_runs (optionnel)')
    p.add_argument('--outdir', default='auto_cluster_csv_out')
    p.add_argument('--top', type=int, default=5,
                   help='nb de features à retenir / cluster / importance')
    p.add_argument('--workers', type=int, default=16,
                   help='processus parallèles')
    args = p.parse_args()
    main(args.real, args.runs, args.outdir, args.top, args.workers)
