import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_results(csv_filename: str):
    """
    Analyse les résultats, calcule des statistiques agrégées par catégorie,
    génère un graphique comparatif et sauvegarde le résumé dans un fichier texte.
    """
    # --- 1. Lecture et Nettoyage des Données ---
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{csv_filename}' n'a pas été trouvé. Veuillez vérifier le nom du fichier.")
        return

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['solve_time_s'], inplace=True)
    
    # S'assurer que delta_s est bien numérique pour les calculs
    df['delta_s'] = pd.to_numeric(df['solve_time_s'] - df['target_time_s'], errors='coerce')


    if df.empty:
        print("Le fichier CSV est vide ou ne contient aucune génération réussie.")
        return

    # --- 2. Catégorisation des Instances ---
    real_instances = df[['base_instance', 'target_time_s']].drop_duplicates()
    
    bins = [0, 100, 300, 700, 1500, np.inf]
    labels = ["Très Rapide (0-100s)", "Rapide (100-300s)", "Moyen (300-700s)", 
              "Long (700-1500s)", "Très Long (1500s+)"]
              
    real_instances['time_category'] = pd.cut(real_instances['target_time_s'], bins=bins, labels=labels, right=False)
    df = pd.merge(df, real_instances[['base_instance', 'time_category']], on='base_instance')

    # --- 3. Analyse Comparative et Préparation de la Sortie ---
    
    # On va stocker toute la sortie texte dans une liste de chaînes de caractères
    output_lines = []
    
    output_lines.append("="*70)
    output_lines.append("ANALYSE COMPARATIVE DES RÉSULTATS DE L'ALGORITHME GÉNÉTIQUE")
    output_lines.append("="*70)

    category_target_times = real_instances.groupby('time_category')['target_time_s'].mean()
    children_mean_times_per_gen = df.groupby(['time_category', 'generation'])['solve_time_s'].mean()

    for category in labels:
        if category in category_target_times.index and category in children_mean_times_per_gen.index.get_level_values('time_category'):
            
            # Filtrer le dataframe pour la catégorie actuelle
            df_category = df[df['time_category'] == category]
            
            # Calculs des nouvelles statistiques
            target_time = category_target_times[category]
            avg_child_time_overall = df_category['solve_time_s'].mean()
            avg_delta_overall = df_category['delta_s'].mean()

            output_lines.append(f"\n--- Catégorie: {category} ---")
            output_lines.append(f"  - Temps Cible Moyen (Instances Réelles) : {target_time:.2f}s")
            output_lines.append(f"  - Temps Moyen (Tous les Enfants Générés): {avg_child_time_overall:.2f}s")
            output_lines.append(f"  - Delta Moyen (Toutes Générations)      : {avg_delta_overall:+.2f}s")
            output_lines.append("\n  Evolution détaillée par génération :")
            
            comparison_df = pd.DataFrame(children_mean_times_per_gen.loc[category])
            comparison_df.rename(columns={'solve_time_s': 'temps_moyen_enfant'}, inplace=True)
            comparison_df['delta_vs_cible_catégorie'] = comparison_df['temps_moyen_enfant'] - target_time
            output_lines.append(comparison_df.to_string(float_format="%.2f"))
            output_lines.append("-"*(len(category) + 8))

    # Affichage dans la console
    for line in output_lines:
        print(line)

    # Sauvegarde dans un fichier texte
    summary_filename = "outputs/results/analyse_summary.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

    # --- 4. Génération du Graphique Comparatif ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 9))
    
    palette = sns.color_palette("viridis", n_colors=len(labels))
    category_colors = dict(zip(labels, palette))

    sns.lineplot(data=df, x='generation', y='solve_time_s', hue='time_category', 
                 hue_order=labels, marker='o', errorbar=None, palette=palette, ax=ax, legend='full')

    for category in category_target_times.index:
        if category in category_colors: # S'assurer que la catégorie a une couleur
            ax.axhline(y=category_target_times[category], color=category_colors[category], 
                        linestyle='--', label=f'Cible - {category}')
    
    handles, legend_labels = ax.get_legend_handles_labels()
    # Corriger la légende pour n'afficher que les lignes principales
    main_handles = [h for h, l in zip(handles, legend_labels) if not l.startswith('Cible')]
    main_labels = [l for l in legend_labels if not l.startswith('Cible')]
    ax.legend(main_handles, main_labels, title="Catégorie (Temps initial)")

    plt.title("Évolution Comparative du Temps de Résolution Moyen par Catégorie", fontsize=16, weight='bold')
    plt.xlabel("Génération", fontsize=12)
    plt.ylabel("Temps de Résolution Moyen (s)", fontsize=12)
    plt.xticks(ticks=sorted(df['generation'].unique()))
    plt.tight_layout()
    
    output_image_filename = "outputs/figures/evolution_comparative_par_categorie.png"
    plt.savefig(output_image_filename)
    
    print("\n" + "="*70)
    print(f"📊 Graphique comparatif sauvegardé : '{output_image_filename}'")
    print(f"📄 Résumé textuel sauvegardé      : '{summary_filename}'")
    print("="*70)


if __name__ == '__main__':
    # ================== PARAMÈTRE UTILISATEUR ==================
    CSV_TO_ANALYZE = "outputs/ga_instances/ga_results.csv"
    # ==========================================================
    
    analyze_results(CSV_TO_ANALYZE)