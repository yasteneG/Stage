#!/usr/bin/env python3
"""
Génération d'instances artificielles avec CTGAN + filtrage
selon les contraintes de De Coster (version R).
"""

import os, random, math
import numpy as np, torch, pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata        # conserve malgré le warning

# -------- chemins / paramètres --------------------------------------------
DATA_PATH = "data/real/real.csv"         # ou real_no_erlangen.csv si split
OUT_DIR   = "outputs/ctgan"
MODEL_FN  = "models/ctgan_model.pkl"
SYN_FN    = "data/synthetic/synthetic_ctgan.csv"
META_JSON = "outputs/ctgan/metadata.json"
N_SYN     = 5000
SEED      = 42

os.makedirs(OUT_DIR, exist_ok=True)

# -------- graine globale ---------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------- données réelles --------------------------------------------------
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:                # supprime la colonne texte
    df = df.drop(columns="name")

# -------- métadonnées ------------------------------------------------------
meta = SingleTableMetadata()
meta.detect_from_dataframe(df)

# -------- entraînement CTGAN ----------------------------------------------
ctgan = CTGANSynthesizer(metadata=meta)     # hyper-paramètres par défaut
ctgan.fit(df)

# ---------------------------------------------------------------------------
#  Fonctions utilitaires
# ---------------------------------------------------------------------------
def is_valid(row: pd.Series) -> bool:
    """
    Renvoie True si la ligne 'row' respecte l'ensemble des contraintes.
    Les noms de colonnes doivent correspondre à ceux du CSV d'origine.
    """
    try:
        # Récupération des valeurs (cast en float -> int pour sûreté numérique)
        v = row.astype(float)

        # Division protégée (évite ZeroDivisionError)
        courses_per_teacher = (
            v["courses"] / v["teachers"] if v["teachers"] != 0 else math.inf
        )

        lpc_valid  = 0 <= v["min_lects_per_course"] <= v["max_lects_per_course"]
        cpt_valid  = (
            0 <= v["min_courses_per_teacher"]
            <= courses_per_teacher
            <= v["max_courses_per_teacher"]
        )
        rs_valid   = 0 <= v["min_room_size"] <= v["max_room_size"]
        cnts_valid = (
            v["rooms"] > 0
            and v["n_curricula"] > 0
            and v["lectures"] > 0
            and v["courses"] > 0
        )
        consts_valid = (
            0 <= v["constraints"] <= v["courses"] * v["periods"] * v["days"]
        )
        cpc_valid  = (
            0 <= v["min_courses_per_curriculum"]
            <= v["max_courses_per_curriculum"]
        )

        return (
            lpc_valid
            and cpt_valid
            and rs_valid
            and cnts_valid
            and consts_valid
            and cpc_valid
        )
    except KeyError as e:
        raise KeyError(
            f"Colonne manquante dans la ligne CTGAN : {e}. "
            "Vérifiez le CSV d'entraînement."
        )

def make_unique_names(n: int, prefix: str = "syn_", start: int = 1):
    """Génère des identifiants uniques 'prefix00001', ..."""
    return [f"{prefix}{idx:05d}" for idx in range(start, start + n)]

# ---------------------------------------------------------------------------
#  Boucle : on échantillonne jusqu'à disposer de N_SYN instances valides
# ---------------------------------------------------------------------------
valid_rows   = []          # liste de DataFrames
total_valid  = 0           # compteur global ---------- ICI

while total_valid < N_SYN:             # on teste bien le NB de lignes
    needed     = N_SYN - total_valid   # ce qu'il manque réellement
    batch_size = int(1.2 * max(needed, 500))
    cand       = ctgan.sample(batch_size)

    mask       = cand.apply(is_valid, axis=1)
    new_valid  = cand[mask]

    valid_rows.append(new_valid)
    total_valid += len(new_valid)      # on met à jour le compteur ---- ICI

    print(f"{len(new_valid)} valides trouvées (total = {total_valid})")

synthetic = pd.concat(valid_rows, ignore_index=True).iloc[:N_SYN].copy()

# -------- ajout colonne 'name' --------------------------------------------
synthetic.insert(0, "name", make_unique_names(len(synthetic)))

# -------- sauvegardes ------------------------------------------------------
synthetic.to_csv(SYN_FN, index=False)
ctgan.save(MODEL_FN)
meta.save_to_json(META_JSON)             # facultatif mais recommandé

print("\nCTGAN terminé ✅")
print("  Modèle        :", MODEL_FN)
print("  Métadonnées   :", META_JSON)
print("  Synthétiques  :", SYN_FN, f"({len(synthetic)} lignes)")
