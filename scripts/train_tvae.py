#!/usr/bin/env python3
"""
Génération d'instances artificielles avec **TVAE** + filtrage
selon les contraintes de De Coster.
"""

import os, random, math
import numpy as np, torch, pandas as pd
from sdv.single_table import TVAESynthesizer          # ← changé
from sdv.metadata import SingleTableMetadata

# -------- chemins / paramètres --------------------------------------------
DATA_PATH = "data/real/real_no_erlangen.csv"  # ou real.csv
OUT_DIR   = "outputs/tvae"
MODEL_FN  = "models/tvae_model.pkl"
SYN_FN    = "data/synthetic/synthetic_tvae.csv"
META_JSON = "outputs/tvae/metadata.json"
N_SYN     = 5000
SEED      = 42

os.makedirs(OUT_DIR, exist_ok=True)

# -------- graine globale ---------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------- données réelles --------------------------------------------------
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:            # supprime la colonne texte
    df = df.drop(columns="name")

# -------- métadonnées ------------------------------------------------------
meta = SingleTableMetadata()
meta.detect_from_dataframe(df)

# -------- entraînement TVAE ----------------------------------------------- 
tvae = TVAESynthesizer(metadata=meta)   # ← changé
tvae.fit(df)

# ---------------------------------------------------------------------------
#  Fonctions utilitaires (identiques à la version CTGAN)
# ---------------------------------------------------------------------------
def is_valid(row: pd.Series) -> bool:
    v = row.astype(float)
    cpt = v["courses"] / v["teachers"] if v["teachers"] else math.inf
    return (
        0 <= v["min_lects_per_course"] <= v["max_lects_per_course"] and
        0 <= v["min_courses_per_teacher"] <= cpt <= v["max_courses_per_teacher"] and
        0 <= v["min_room_size"] <= v["max_room_size"] and
        v["rooms"] > 0 and v["n_curricula"] > 0 and v["lectures"] > 0 and v["courses"] > 0 and
        0 <= v["constraints"] <= v["courses"] * v["periods"] * v["days"] and
        0 <= v["min_courses_per_curriculum"] <= v["max_courses_per_curriculum"]
    )

def make_unique_names(n: int, prefix: str = "syn_", start: int = 1):
    return [f"{prefix}{idx:05d}" for idx in range(start, start + n)]

# ---------------------------------------------------------------------------
#  Boucle : échantillonnage jusqu'à N_SYN valides
# ---------------------------------------------------------------------------
valid_rows, total_valid = [], 0
while total_valid < N_SYN:
    need       = N_SYN - total_valid
    batch_size = int(1.2 * max(need, 500))
    cand       = tvae.sample(batch_size)            # ← changé

    mask       = cand.apply(is_valid, axis=1)
    new_valid  = cand[mask]

    valid_rows.append(new_valid)
    total_valid += len(new_valid)
    print(f"{len(new_valid)} valides trouvées (total = {total_valid})")

synthetic = pd.concat(valid_rows, ignore_index=True).iloc[:N_SYN].copy()

# -------- ajout colonne 'name' --------------------------------------------
synthetic.insert(0, "name", make_unique_names(len(synthetic)))

# -------- sauvegardes ------------------------------------------------------
synthetic.to_csv(SYN_FN, index=False)
tvae.save(MODEL_FN)                                     # ← changé
meta.save_to_json(META_JSON)

print("\nTVAE terminé ✅")
print("  Modèle        :", MODEL_FN)
print("  Métadonnées   :", META_JSON)
print("  Synthétiques  :", SYN_FN, f"({len(synthetic)} lignes)")
