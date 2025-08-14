#!/usr/bin/env python3
"""
visualize_umap.py – Projection UMAP des instances réelles (avec labels) et synthétiques
"""

import argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap          # pip install umap-learn

def load_and_tag(path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source"] = tag
    return df

def main(real_path: str, synth_path: str, rng: int = 42, outfile: str | None = None):
    # 1) lecture + concaténation
    df_real  = load_and_tag(real_path,  "real")
    df_synth = load_and_tag(synth_path, "synthetic")
    df_all   = pd.concat([df_real, df_synth], ignore_index=True)

    # 2) variables numériques
    X = df_all.drop(columns=["name", "__source"])
    X_scaled = StandardScaler().fit_transform(X)

    # 3) UMAP
    emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=rng).fit_transform(X_scaled)
    df_all["umap_x"], df_all["umap_y"] = emb[:, 0], emb[:, 1]

    # 4) tracé
    plt.figure(figsize=(8, 6))
    m_real = df_all["__source"] == "real"

    # a) synthétiques
    plt.scatter(df_all.loc[~m_real, "umap_x"],
                df_all.loc[~m_real, "umap_y"],
                s=20, alpha=0.7, c="#1f77b4",
                label="Synthétiques", zorder=1)

    # b) réelles
    plt.scatter(df_all.loc[m_real, "umap_x"],
                df_all.loc[m_real, "umap_y"],
                s=25, alpha=0.9, c="#ff69b4",
                label="Réelles", zorder=2)

    # c) labels pour les réelles
    for _, row in df_all[m_real].iterrows():
        label = row["name"]          # valeur de la colonne
        plt.text(row.umap_x, row.umap_y, label,
                fontsize=7, ha="center", va="center", zorder=3)


    plt.title("Projection UMAP – Instances réelles vs. synthétiques")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(); plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"Figure sauvegardée → {outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",      default="data/real/real.csv", help="Chemin vers real.csv")
    parser.add_argument("--synthetic", default="data/synthetic/synthetic_ctgan.csv",    help="Chemin vers synthetic.csv")
    parser.add_argument("--out",       default="Umap",               help="PNG de sortie (optionnel)")
    parser.add_argument("--seed",      type=int, default=42,       help="Graine UMAP")
    args = parser.parse_args()
    main(args.real, args.synthetic, args.seed, args.out)
