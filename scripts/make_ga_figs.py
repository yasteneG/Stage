#!/usr/bin/env python3
# make_ga_figs.py
# Crée :
#  - Fig. 15 : fig15_delta_par_categorie.png (+ CSV agrégé)
#  - Fig. 16 : fig16_diff_structurelle_moyenne.png (si ≥2 stratégies : comparaison par label)
#              et fig16_diff_structurelle_distribution.png (si 1 seule stratégie : histogramme)
#
# Entrées :
#  --ga ga_results_v10_final.csv
#  --comp comparison_summary_average_v10.csv[:LABEL] ...   (1 ou plusieurs fichiers "détaillés")

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- FIG. 15 --------------------------------------
def load_ga_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Nettoyage basique
    for col in ["solve_time_s", "target_time_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["solve_time_s", "target_time_s"])
    # Δ (en secondes)
    df["delta_s"] = df["solve_time_s"] - df["target_time_s"]

    # Catégories (comme dans ton script d’analyse)
    bins = [0, 100, 300, 700, 1500, np.inf]
    labels = ["Très Rapide (0-100s)", "Rapide (100-300s)", "Moyen (300-700s)",
              "Long (700-1500s)", "Très Long (1500s+)"]
    # On catégorise à partir de la cible (temps réel de l’instance d’origine)
    df["time_category"] = pd.cut(df["target_time_s"], bins=bins, labels=labels, right=False)
    return df


def make_fig15(ga_df: pd.DataFrame, outdir: Path):
    # Agrégation Δ moyen + écart-type par catégorie
    grp = ga_df.groupby("time_category")["delta_s"].agg(["mean", "std", "count"]).reset_index()
    grp = grp.sort_values("time_category")
    grp.to_csv(outdir / "fig15_delta_par_categorie.csv", index=False)

    # Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grp, x="time_category", y="mean", errorbar=("sd"))
    plt.axhline(0.0, ls="--", lw=1)
    plt.title(" Δ temps moyen (solveur − cible) par catégorie")
    plt.xlabel("Catégorie (temps cible de l’instance réelle)")
    plt.ylabel("Δ moyen (secondes)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = outdir / "fig15_delta_par_categorie.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] Fig. 15 : {out}")


# ----------------------------- FIG. 16 --------------------------------------
# Le CSV "détaillé" contient des colonnes du type :
#   - "Étudiants (réel)" / "Étudiants (moyenne générée)"
#   - "Cours (réel)" / "Cours (moyenne générée)"
#   - "Limite Classes (moy) (réel)" / "Limite Classes (moy) (moyenne générée)"
#   - "Pénalités (moy) (réel)" / "Pénalités (moy) (moyenne générée)"
# On re-calcule le score "Différence (%)" = moyenne des 4 écarts relatifs (en %).

CANDIDATE_KEYS = [
    # (base, real_suffix, synth_suffix)
    ("Étudiants", " (réel)", " (moyenne générée)"),
    ("Cours", " (réel)", " (moyenne générée)"),
    ("Limite Classes (moy)", " (réel)", " (moyenne générée)"),
    ("Pénalités (moy)", " (réel)", " (moyenne générée)"),
]

def pick_col(df: pd.DataFrame, wanted: str) -> str | None:
    # Recherche tolérante (espaces/accents)
    cols = {c.strip(): c for c in df.columns}
    if wanted in cols:
        return cols[wanted]
    # Fallbacks simples
    for c in df.columns:
        if wanted.replace("é", "e").replace("É", "E") in c.replace("é", "e").replace("É", "E"):
            if "réel" in wanted and "réel" in c:
                return c
            if "moyenne générée" in wanted and "moyenne générée" in c:
                return c
            if (" (réel)" not in wanted) and (" (moyenne générée)" not in wanted) and wanted in c:
                return c
    return None


def compute_structure_diff_pct(df: pd.DataFrame) -> pd.Series:
    diffs = []
    for base, s_real, s_syn in CANDIDATE_KEYS:
        col_r = pick_col(df, base + s_real)
        col_s = pick_col(df, base + s_syn)
        if col_r is None or col_s is None:
            continue
        r = pd.to_numeric(df[col_r], errors="coerce")
        s = pd.to_numeric(df[col_s], errors="coerce")
        d = np.where(r > 0, np.abs((s - r) / r) * 100.0, np.nan)
        diffs.append(pd.Series(d, index=df.index))
    if not diffs:
        raise ValueError("Colonnes attendues introuvables dans le CSV de comparaison détaillé.")
    D = pd.concat(diffs, axis=1)
    return D.mean(axis=1, skipna=True)  # score moyen % par instance


def load_comparison(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["score_diff_pct"] = compute_structure_diff_pct(df)
    df["label"] = label
    # On essaie d’avoir un identifiant d’instance lisible
    if "instance" in df.columns:
        df["Instance"] = df["instance"]
    elif "Instance" not in df.columns:
        df["Instance"] = df.index.astype(str)
    return df[["Instance", "score_diff_pct", "label"]].dropna()


def parse_comp_specs(specs: list[str]) -> list[tuple[Path, str]]:
    out = []
    for s in specs:
        if ":" in s:
            p, lab = s.split(":", 1)
        else:
            p, lab = s, Path(s).stem
        out.append((Path(p), lab))
    return out


def make_fig16(comp_dfs: list[pd.DataFrame], outdir: Path):
    DF = pd.concat(comp_dfs, ignore_index=True)

    # Sauvegarde brute
    DF.to_csv(outdir / "fig16_structure_diff_per_instance.csv", index=False)

    labels = DF["label"].unique().tolist()
    if len(labels) == 1:
        # Un seul scénario : distribution + moyenne annotée
        lab = labels[0]
        mean_val = DF["score_diff_pct"].mean()
        std_val  = DF["score_diff_pct"].std(ddof=1)

        plt.figure(figsize=(10, 6))
        plt.hist(DF["score_diff_pct"].dropna(), bins="auto", density=False)
        plt.axvline(mean_val, ls="--", lw=1)
        plt.title(f" Différence structurelle (%), distribution — {lab}\n"
                  f"Moyenne = {mean_val:.3f} % | Écart-type = {std_val:.3f} %")
        plt.xlabel("Score Différence (%) — plus bas est meilleur")
        plt.ylabel("Nombre d’instances")
        plt.tight_layout()
        out = outdir / "fig16_diff_structurelle_distribution.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] Fig. 16 (distribution) : {out}")
    else:
        # Plusieurs scénarios : barplot des moyennes par label
        grp = DF.groupby("label")["score_diff_pct"].agg(["mean", "std", "count"]).reset_index()
        grp = grp.sort_values("mean")
        grp.to_csv(outdir / "fig16_structure_diff_summary_by_label.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=grp, x="label", y="mean", errorbar=("sd"))
        plt.title("Fig. 16 — Différence structurelle moyenne par stratégie")
        plt.xlabel("Stratégie / Label")
        plt.ylabel("Score Différence moyen (%) — plus bas est meilleur")
        plt.tight_layout()
        out = outdir / "fig16_diff_structurelle_moyenne.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] Fig. 16 (moyennes par label) : {out}")


# ----------------------------- MAIN -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Figs 15 & 16 pour ITC-2019 (GA + comparaison structurelle).")
    ap.add_argument("--ga", default="outputs/ga_instances/ga_results.csv", help="CSV résultats GA (solve_time/target_time).")
    ap.add_argument("--comp", nargs="+", required=True,default="outputs/results/comparison_summary_average.csv"
                    help="Un ou plusieurs CSV de comparaison détaillée : chemin[:LABEL]")
    ap.add_argument("--outdir", default="outputs/figures", help="Dossier de sortie.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Fig. 15
    ga_df = load_ga_csv(Path(args.ga))
    make_fig15(ga_df, outdir)

    # Fig. 16
    comp_specs = parse_comp_specs(args.comp)
    comp_dfs = [load_comparison(p, lab) for p, lab in comp_specs]
    make_fig16(comp_dfs, outdir)

    print("\n[DONE] Figs 15 & 16 générées.")


if __name__ == "__main__":
    main()
