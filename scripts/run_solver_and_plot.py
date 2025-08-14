#!/usr/bin/env python3
import argparse, pathlib, os, csv
import subprocess
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

# --- Defaults adaptés à l'arborescence ---
DEFAULT_ECTT_DIR = "data/synthetic/ectt"
DEFAULT_FEAT_CSV = "data/synthetic/synthetic_ctgan.csv"
DEFAULT_OUT_DIR  = "outputs/results/solver_ctgan"
DEFAULT_OUT_CSV  = "outputs/results/solve_ctgan.csv"
SOLVER_JAR = pathlib.Path("solver/itc2007.jar")

def build_parser():
    p = argparse.ArgumentParser(
        description="Lance le solveur ITC-2007 de Müller sur des instances .ectt"
    )
    p.add_argument("--ectt_dir", "--ectt-dir", default=DEFAULT_ECTT_DIR,
                   help="Dossier contenant les fichiers .ectt")
    p.add_argument("--feat", "--features", default=DEFAULT_FEAT_CSV,
                   help="CSV de features (optionnel)")
    p.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUT_DIR,
                   help="Dossier de sortie")
    p.add_argument("--out_csv", "--out-csv", default=DEFAULT_OUT_CSV,
                   help="CSV récapitulatif des résultats")
    p.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    p.add_argument("--timeout", type=int, default=300, help="Timeout en secondes par instance")
    p.add_argument("-n", "--n", type=int, default=None,
                   help="Nombre max d’instances à traiter")
    p.add_argument("--jobs", type=int, default=8,
                   help="Nombre de jobs en parallèle")
    return p

def run_solver_on_instance(ectt_file, timeout, seed):
    """Appelle le solveur externe de Müller sur une instance"""
    cmd = ["java", "-cp", str(SOLVER_JAR),
       "cb_ctt.algorithms.HybridConstraintSolver",
       str(ectt_file), "--seed", str(seed)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        solved = "success" in result.stdout.lower()
        runtime = extract_runtime(result.stdout)
        return dict(name=ectt_file.name, solved=solved, runtime=runtime)
    except subprocess.TimeoutExpired:
        return dict(name=ectt_file.name, solved=False, runtime=timeout)

def extract_runtime(stdout_text):
    """Extrait un temps de résolution à partir des logs solveur"""
    for line in stdout_text.splitlines():
        if "time" in line.lower() and "ms" in line.lower():
            try:
                return int(line.split()[-2])
            except:
                continue
    return None

def main():
    parser = build_parser()
    args = parser.parse_args()

    ectt_dir = pathlib.Path(args.ectt_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_csv = pathlib.Path(args.out_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ectt_files = sorted(list(ectt_dir.glob("*.ectt")))
    if args.n:
        ectt_files = ectt_files[:args.n]

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for res in tqdm(ex.map(lambda f: run_solver_on_instance(f, args.timeout, args.seed),
                               ectt_files), total=len(ectt_files)):
            results.append(res)

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Résultats sauvegardés dans {out_csv}")

if __name__ == "__main__":
    main()
