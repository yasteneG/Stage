#!/usr/bin/env python3
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

# --- Defaults adaptés ---
DEFAULT_REAL = "data/real/real.csv"
DEFAULT_SYN  = "data/synthetic/synthetic_ctgan.csv"
DEFAULT_SOLV_CSV = "outputs/results/solve_ctgan.csv"
DEFAULT_OUT_PNG  = "outputs/figures/umap_ctgan_solved.png"

def build_parser():
    p = argparse.ArgumentParser(
        description="Projette en UMAP les instances réelles et synthétiques, colore par statut résolu/non."
    )
    p.add_argument("--feat", default=None,
                   help="CSV unique de features (mode historique)")
    p.add_argument("--real", default=DEFAULT_REAL,
                   help="CSV des instances réelles")
    p.add_argument("--syn", "--synthetic", default=DEFAULT_SYN,
                   help="CSV des instances synthétiques")
    p.add_argument("--sol_csv", "--sol-csv", default=DEFAULT_SOLV_CSV,
                   help="CSV récap solveur")
    p.add_argument("--out", default=DEFAULT_OUT_PNG,
                   help="Image PNG de sortie")
    return p

def load_features(args):
    if args.feat:
        return pd.read_csv(args.feat)
    real = pd.read_csv(args.real).copy()
    syn  = pd.read_csv(args.syn).copy()
    real["__origin__"] = "real"
    syn["__origin__"]  = "synthetic"
    df = pd.concat([real, syn], ignore_index=True)
    return df

def main():
    parser = build_parser()
    args = parser.parse_args()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger features
    features = load_features(args)

    # Charger résultats solveur
    solve_df = pd.read_csv(args.sol_csv) if pathlib.Path(args.sol_csv).exists() else None

    # Harmoniser colonnes numériques
    X = features.select_dtypes(include="number")
    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    features["umap_x"] = embedding[:,0]
    features["umap_y"] = embedding[:,1]

    if solve_df is not None and "name" in solve_df.columns:
        merged = features.merge(solve_df, on="name", how="left")
        colors = merged["solved"].map({True:"green", False:"red"}).fillna("gray")
    else:
        merged = features
        colors = merged["__origin__"].map({"real":"blue", "synthetic":"orange"})

    plt.figure(figsize=(8,6))
    plt.scatter(merged["umap_x"], merged["umap_y"], c=colors, alpha=0.7)
    plt.title("Projection UMAP des instances (résolues / non résolues)")
    plt.savefig(out_path, dpi=300)
    print(f"Figure sauvegardée dans {out_path}")

if __name__ == "__main__":
    main()
