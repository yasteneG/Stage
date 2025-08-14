#!/usr/bin/env python3
"""
visualize_umap_fixed.py – Réelles fixées, synthétiques projetées

Usage :
    python visualize_umap_fixed.py --real real.csv --synthetic synthetic.csv
    python visualize_umap_fixed.py --real real.csv --synthetic synthetic.csv --out umap_fixed.png
"""

import argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap          # pip install umap-learn

# ------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"'name' column missing in {path}")
    return df

def main(real_path: str, synth_path: str, rng: int = 42,
         outfile: str | None = None):

    # 1) lecture
    df_real  = load_csv(real_path)
    df_syn   = load_csv(synth_path)

    # 2) variables numériques (on entraîne scaler sur RÉEL uniquement)
    num_cols = [c for c in df_real.columns if c not in ("name",)]
    scaler   = StandardScaler().fit(df_real[num_cols])

    X_real_scaled = scaler.transform(df_real[num_cols])
    X_syn_scaled  = scaler.transform(df_syn[num_cols])   # même scaler !

    # 3) UMAP : fit sur le jeu RÉEL, puis transform
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=rng).fit(X_real_scaled)

    emb_real = umap_model.embedding_               # coordonnées figées
    emb_syn  = umap_model.transform(X_syn_scaled)  # projection passive

    # 4) DataFrame combiné pour le tracé
    real_plot = pd.DataFrame(emb_real, columns=["x", "y"])
    real_plot["name"] = df_real["name"]

    syn_plot  = pd.DataFrame(emb_syn,  columns=["x", "y"])

    # 5) tracé
    plt.figure(figsize=(8, 6))

    # a) synthétiques en fond
    plt.scatter(syn_plot.x, syn_plot.y,
                s=20, alpha=0.7, c="#1f77b4",
                label="Synthétiques", zorder=1)

    # b) réelles au-dessus
    plt.scatter(real_plot.x, real_plot.y,
                s=25, alpha=0.9, c="#ff69b4",
                label="Réelles", zorder=2)

    # c) étiquettes des réelles
    for _, r in real_plot.iterrows():
        plt.text(r.x, r.y, r["name"],     # <-- colonne "name"
                 fontsize=7, ha="center", va="center", zorder=3)


    plt.title("UMAP – réelles fixées, synthétiques projetées")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(); plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"Figure sauvegardée → {outfile}")
    else:
        plt.show()

# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real",      default="data/real/real.csv", help="Chemin vers real.csv")
    p.add_argument("--synthetic", default="data/synthetic/synthetic_ctgan.csv", help="Chemin vers synthetic.csv")
    p.add_argument("--out",       default="All_V2",  help="PNG de sortie (optionnel)")
    p.add_argument("--seed",      type=int, default=42, help="Graine UMAP/Scikit-learn")
    args = p.parse_args()

    main(args.real, args.synthetic, args.seed, args.out)
