#!/usr/bin/env python3
"""
csv2ectt.py – Convertit un ou plusieurs CSV de features (schéma De Coster) en
fichiers .ectt compatibles avec udineGenUtils2.py.

Usage :
    python csv2ectt.py real.csv synthetic.csv --outdir ./muller/data_artificial

Colonnes minimales requises dans le CSV :
    name, courses, lectures, rooms, days, periods,
    teachers, n_curricula, constraints

Toute autre colonne est optionnelle ; des valeurs par défaut intelligentes
sont générées si elles manquent.
"""
import argparse
import math
import pathlib
import subprocess
import sys
from typing import Callable, Dict, Any

import pandas as pd



###############################################################################
# Configuration                                                               #
###############################################################################

GEN_SCRIPT = pathlib.Path("udineGenUtils2.py")  # générateur officiel Udine

DEFAULTS: Dict[str, Any] = {
    # Bornes lectures / cours (≈ De Coster : 5–15 % des leçons totales)
    "min_lects_per_course": lambda r: max(1, math.ceil(0.05 * r["lectures"])),
    "max_lects_per_course": lambda r: max(3, math.ceil(0.15 * r["lectures"])),

    # Répartition cours ↔ enseignants
    "min_courses_per_teacher": 1,
    "max_courses_per_teacher": lambda r: max(1, int(r["courses"] // 3) or 1),

    # Répartition cours ↔ curricula
    "min_courses_per_curriculum": 2,
    "max_courses_per_curriculum": lambda r: max(2, int(r["courses"] // 2) or 2),

    # Capacités salles
    "min_room_size": 20,
    "max_room_size": lambda r: max(150, int(r.get("numStudents", 300))),

    # Contraintes d’indisponibilité
    "constraints": 0,
}

###############################################################################
# Utilitaires                                                                 #
###############################################################################

def ival(row: pd.Series, col: str, default: Any = None) -> int:
    """Force row[col] en entier, sinon retourne default ou DEFAULTS[col]."""
    if col in row and not pd.isna(row[col]):
        return int(round(float(row[col])))

    if default is None:
        default = DEFAULTS[col]
    return int(default(row) if callable(default) else default)


def clamp_params(row: pd.Series) -> Dict[str, int]:
    """Assainit les bornes avant appel Udine (aucune incohérence possible)."""
    nc = ival(row, "courses")

    # -------- curricula --------
    min_cc = min(ival(row, "min_courses_per_curriculum"), nc)
    max_cc = min(max(min_cc, ival(row, "max_courses_per_curriculum")), nc)

    # -------- cours / teacher --------
    min_ct = min(ival(row, "min_courses_per_teacher"), nc)
    max_ct = min(max(min_ct, ival(row, "max_courses_per_teacher")), nc)

    return {
        "n_courses": nc,
        "min_courses_per_curriculum": min_cc,
        "max_courses_per_curriculum": max_cc,
        "min_courses_per_teacher": min_ct,
        "max_courses_per_teacher": max_ct,
    }

###############################################################################
# Construction de la ligne de commande                                       #
###############################################################################

def build_cmd(row: pd.Series, outdir: pathlib.Path) -> list[str]:
    stem = str(row["name"])
    target = outdir / f"{stem}.ectt"
    fix = clamp_params(row)

    # Garantit ≥1 lecture / cours → lectures >= courses
    lectures = max(ival(row, "lectures"), fix["n_courses"])

    # Bornes lectures par cours (≥1 et cohérentes)
    min_lp = max(1, ival(row, "min_lects_per_course"))
    max_lp = max(min_lp, ival(row, "max_lects_per_course"))

    opts = {
        "--name": stem,
        "--out": str(target),
        "--n_courses": fix["n_courses"],
        "--lectures": lectures,
        "--n_rooms": ival(row, "rooms"),
        "--days": ival(row, "days"),
        "--periods": ival(row, "periods"),
        "--teachers": ival(row, "teachers"),
        "--curricula": ival(row, "n_curricula"),
        "--min_courses_per_curriculum": fix["min_courses_per_curriculum"],
        "--max_courses_per_curriculum": fix["max_courses_per_curriculum"],
        "--min_courses_per_teacher": fix["min_courses_per_teacher"],
        "--max_courses_per_teacher": fix["max_courses_per_teacher"],
        "--min_lects_per_course": min_lp,
        "--max_lects_per_course": max_lp,
        "--min_room_size": ival(row, "min_room_size"),
        "--max_room_size": ival(row, "max_room_size"),
        "--constraints": ival(row, "constraints"),
    }

    cmd = [sys.executable, str(GEN_SCRIPT)]
    for k, v in opts.items():
        cmd += [k, str(v)]
    return cmd

###############################################################################
# Traitement CSV → .ectt                                                     #
###############################################################################

def process_csv(csv_path: pathlib.Path, outdir: pathlib.Path):
    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError(f"{csv_path} doit contenir une colonne 'name'.")

    for _, row in df.iterrows():
        cmd = build_cmd(row, outdir)
        subprocess.run(cmd, check=True)
        print("✓", row["name"], flush=True)

###############################################################################
# CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",default ="data/synthetic/synthetic_ctgan.csv", help="CSV sources à convertir")
    ap.add_argument("--outdir", default="data/synthetic/ectt", help="dossier cible .ectt")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for c in args.csv:
        process_csv(pathlib.Path(c), outdir)
