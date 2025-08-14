README — Projet ITC Synthetic Instances
1. Introduction
Ce projet s’inscrit dans le cadre d’un stage de recherche portant sur la génération d’instances artificielles pour des problèmes de planification d’emploi du temps universitaire.
Deux compétitions internationales servent de référence :
ITC-2007 : instances en format CSV ou ECTT, problème « course timetabling ».
ITC-2019 : instances en format XML, problème plus riche avec contraintes physiques, distribution et déplacements.
L’objectif est double :
Générer de nouvelles instances réalistes mais inédites, utilisables comme bancs d’essai pour les solveurs.
Étudier l’effet de différentes méthodes (GAN, autoencodeurs, algorithmes génétiques) sur la qualité et la variété des instances produites.

2. Organisation des fichiers

projet_itc/
├── data/
│   ├── real/                        # Instances réelles ITC-2007 (CSV)
│   │   └── real.csv
│   ├── synthetic/
│   │   └── ectt/                    # Instances générées ITC-2007 (ECTT)
│   └── itc2019/                     # Instances réelles ITC-2019 (XML)
│       └── *.xml
├── models/                          # Modèles entraînés (CTGAN, TVAE)
│   ├── ctgan_model.pkl
│   └── tvae_model.pkl
├── outputs/
│   ├── results/                     # Logs & comparaisons CSV
│   └── figures/                     # Figures générées
├── scripts/                         # Tous les scripts Python
│   ├── auto_cluster_csv.py          # Clustering ITC-2007
│   ├── train_ctgan.py               # Génération ITC-2007 (CTGAN)
│   ├── train_tvae.py                # Génération ITC-2007 (TVAE)
│   ├── csv2ectt.py                  # Conversion CSV → ECTT
│   ├── run_solver_and_plot.py       # Solveur Muller (ITC-2007)
│   ├── show_umap_solved.py          # Visualisation UMAP (ITC-2007)
│   ├── ga_itc2019.py                    # Génération ITC-2019 (Algorithme génétique)
│   ├── analyse_ga.py                   # Analyse GA (ITC-2019)
│   └── compare_instances.py         # Comparaison réal vs synth (ITC-2019)
├── solver/
│   ├── itc2007.jar                  # Solveur Muller ITC-2007
│   └── configuration                    
│   │    └── default.cfg
│   ├── dependency/
│   │     └── *.jar
│   └── cpsolver-itc2019-1.0-SNAPSHOT # Solveur ITC-2019
├── requirements.txt                 # Dépendances Python
└── README.md


3. Installation et prérequis
Outils nécessaires
Python 3.10.11
Java 11+
Git (optionnel)


Installation Python :
pip install -r requirements.txt



4. Pipeline ITC-2007
4.1 Clustering des instances réelles

python scripts/auto_cluster_csv.py --real data/real/real.csv --outdir outputs/results/clusters.csv --top 5
Sortie : clusters.csv avec la catégorie assignée à chaque instance.


4.2 Génération via CTGAN

python scripts/train_ctgan.py


4.3 Génération via TVAE
python scripts/train_tvae.py 


4.4 Conversion vers ECTT
python scripts/csv2ectt.py --csv data/synthetic/synthetic_ctgan.csv  --outdir data/synthetic/ectt/


4.5 Solveur Muller ITC-2007
python scripts/run_solver_and_plot.py --ectt data/synthetic/ectt --feat data/synthetic/synthetic_ctgan.csv  --output_dir outputs/results/solver_ctgan --out_csv outputs/results/solve_ctgan.csv --timeout 300 

4.6 Visualisation UMAP
python scripts/show_umap_solved.py --real data/real/real.csv  --syn  data/synthetic/synthetic_ctgan.csv  --sol_csv outputs/results/solve_ctgan.csv --out  outputs/figures/umap_ctgan_solved.png




5. Pipeline ITC-2019 (Algorithme génétique)
5.1 Génération avec ga_itc2019.py

python scripts/ga_itc2019.py

Sorties :
outputs/ga_instances/<instance>/*.xml
outputs/ga_instances/ga_results.csv


5.2 Analyse des résultats GA
python scripts/analyse_ga.py



5.3 Comparaison réal vs synth
python scripts/compare_instances.py


6. Résultats et figures

Les scripts produisent automatiquement :

Pour ITC-2007 :
Graphiques UMAP (instances résolues vs non résolues)
Statistiques solveur Muller

Pour ITC-2019 :
evolution_comparative_par_categorie.png (fitness par génération)
fig15_delta_par_categorie.png (écarts de temps cible par catégorie)
fig16_diff_structurelle_distribution.png (différences structurelles moyennes)



