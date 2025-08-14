import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
from pathlib import Path

def get_element_counts(tree: ET.ElementTree) -> dict:
    """Niveau 1: Compte les éléments clés de l'instance."""
    return {
        "Salles": len(tree.findall('.//room')),
        "Cours": len(tree.findall('.//course')),
        "Étudiants": len(tree.findall('.//student')),
        "Contraintes": len(tree.findall('.//distribution'))
    }

def get_numeric_distributions(tree: ET.ElementTree) -> dict:
    """Niveau 2: Calcule les statistiques sur les paramètres numériques clés."""
    capacities = [int(r.get('capacity', 0)) for r in tree.findall('.//room')]
    limits = [int(c.get('limit', 0)) for c in tree.findall('.//class')]
    penalties = [int(p.get('penalty', 0)) for p in tree.findall('.//*[@penalty]')]
    
    return {
        "Capacité Salles (moy)": np.mean(capacities) if capacities else 0,
        "Limite Classes (moy)": np.mean(limits) if limits else 0,
        "Pénalités (moy)": np.mean(penalties) if penalties else 0
    }

def batch_compare_folders_average(real_folder: Path, synthetic_folder: Path):
    """
    Compare en lot les instances réelles avec la moyenne de leurs versions générées
    et produit un résumé simplifié ainsi qu'un export CSV détaillé.
    """
    all_results_detailed = []

    if not real_folder.exists() or not synthetic_folder.exists():
        print(f"Erreur : Un des dossiers spécifiés n'existe pas.")
        return

    for real_path in sorted(real_folder.glob("*.xml")):
        instance_name = real_path.stem
        synthetic_instance_folder = synthetic_folder / instance_name
        
        if not synthetic_instance_folder.exists(): continue
        generated_files = list(synthetic_instance_folder.glob("*.xml"))
        if not generated_files: continue

        try:
            real_tree = ET.parse(real_path)
            real_counts = get_element_counts(real_tree)
            real_dist = get_numeric_distributions(real_tree)

            all_gen_counts = []
            all_gen_dists = []
            for gen_path in generated_files:
                try:
                    gen_tree = ET.parse(gen_path)
                    all_gen_counts.append(get_element_counts(gen_tree))
                    all_gen_dists.append(get_numeric_distributions(gen_tree))
                except ET.ParseError: continue

            if not all_gen_counts: continue

            avg_gen_counts = pd.DataFrame(all_gen_counts).mean().to_dict()
            avg_gen_dists = pd.DataFrame(all_gen_dists).mean().to_dict()
            
            # --- Remplissage de la ligne de résultat détaillée (CETTE PARTIE EST CORRIGÉE) ---
            detailed_row = {"instance": instance_name}
            for key in real_counts:
                detailed_row[f"{key} (réel)"] = real_counts[key]
                detailed_row[f"{key} (moyenne générée)"] = avg_gen_counts.get(key, 0)
            
            for key in real_dist:
                detailed_row[f"{key} (réel)"] = real_dist[key]
                detailed_row[f"{key} (moyenne générée)"] = avg_gen_dists.get(key, 0)
                
            all_results_detailed.append(detailed_row)

        except ET.ParseError as e:
            continue

    # --- Préparation et Affichage Final ---
    if not all_results_detailed:
        print("\nAucune comparaison n'a pu être effectuée.")
        return
        
    # Créer le DataFrame détaillé à partir des résultats collectés
    summary_df_detailed = pd.DataFrame(all_results_detailed).set_index("instance")
    
    # --- Création du résumé simple pour la console ---
    summary_simple_list = []
    for instance_name, row in summary_df_detailed.iterrows():
        diff_students_pct = abs((row["Étudiants (moyenne générée)"] - row["Étudiants (réel)"]) / row["Étudiants (réel)"]) * 100 if row["Étudiants (réel)"] else 0
        diff_courses_pct = abs((row["Cours (moyenne générée)"] - row["Cours (réel)"]) / row["Cours (réel)"]) * 100 if row["Cours (réel)"] else 0
        diff_limits_pct = abs((row["Limite Classes (moy) (moyenne générée)"] - row["Limite Classes (moy) (réel)"]) / row["Limite Classes (moy) (réel)"]) * 100 if row["Limite Classes (moy) (réel)"] else 0
        diff_penalties_pct = abs((row["Pénalités (moy) (moyenne générée)"] - row["Pénalités (moy) (réel)"]) / row["Pénalités (moy) (réel)"]) * 100 if row["Pénalités (moy) (réel)"] else 0
            
        similarity_score = np.mean([diff_students_pct, diff_courses_pct, diff_limits_pct, diff_penalties_pct])
            
        changes = {
            "Nombre d'étudiants": diff_students_pct, "Nombre de cours": diff_courses_pct,
            "Limite des classes": diff_limits_pct, "Pénalités": diff_penalties_pct
        }
        main_change = max(changes, key=changes.get)

        summary_simple_list.append({
            "Instance": instance_name,
            "Score Différence (%)": similarity_score,
            "Changement Principal": main_change
        })
    
    summary_simple_df = pd.DataFrame(summary_simple_list).set_index("Instance")

    print("="*80)
    print("                 SCORE DE SIMILARITÉ (RÉEL vs. MOYENNE DES GÉNÉRATIONS)")
    print("="*80)
    print("Le 'Score de Différence' est la variation structurelle moyenne. Un score bas est meilleur.")
    print("\n" + summary_simple_df.to_string(float_format="%.2f"))
    
    # --- Sauvegarde du Fichier CSV DÉTAILLÉ (CORRIGÉ) ---
    output_filename = "outputs/results/comparison_summary_average.csv"
    summary_df_detailed.to_csv(output_filename)
    
    print("\n" + "="*80)
    print(f"📊 Un résumé détaillé et complet a été sauvegardé dans : '{output_filename}'")
    print("="*80)

if __name__ == '__main__':
    # ================== PARAMÈTRES UTILISATEUR ==================
    REAL_INSTANCES_FOLDER = "data/itc2019"
    SYNTHETIC_INSTANCES_FOLDER = "outputs/ga_instances" # Mettez ici le dossier de résultats à analyser
    # ==========================================================
    
    batch_compare_folders_average(REAL_INSTANCES_FOLDER, SYNTHETIC_INSTANCES_FOLDER)