import os
import pathlib
import shutil
import subprocess
import time
import argparse
import random
import pandas as pd
from xml.etree import ElementTree as ET
from multiprocessing import Pool, cpu_count

# ═════════════════════ Configuration Globale ══════════════════════
ROOT = pathlib.Path(__file__).resolve().parent.parent   # .../projet_itc
SOLVER_JAR_2019 = ROOT / "solver/cpsolver-itc2019-1.0-SNAPSHOT.jar"
DEPS_DIR        = ROOT / "solver/dependency"   # peut être vide si shaded-jar
CFG_DEFAULT     = ROOT / "solver/configuration/default.cfg"
XML_DIR         = ROOT / "data/itc2019"                # réelles 2019 (entrées)
OUT_DIR         = ROOT / "outputs/ga_instances"        # sorties GA (instances synth)

INIT_CFG_PATH   = OUT_DIR / "init_only.cfg"
RESULTS_CSV     = OUT_DIR / "ga_results.csv"
# ══════════════════════ Fonctions de Mutation (inchangées) ════════════════════════

def mutate_room_capacity(tree: ET.ElementTree, amount_percent: float = 0.05):
    rooms = tree.findall('.//room[@capacity]')
    if not rooms: return False
    room = random.choice(rooms)
    current_capacity = int(room.get('capacity'))
    delta = max(1, int(current_capacity * amount_percent))
    new_capacity = max(1, current_capacity + random.randint(-delta, delta))
    room.set('capacity', str(new_capacity))
    return True

def mutate_class_limit(tree: ET.ElementTree, amount_percent: float = 0.05):
    classes = tree.findall('.//class[@limit]')
    if not classes: return False
    chosen_class = random.choice(classes)
    current_limit = int(chosen_class.get('limit'))
    delta = max(1, int(current_limit * amount_percent))
    new_limit = max(1, current_limit + random.randint(-delta, delta))
    chosen_class.set('limit', str(new_limit))
    return True
    
def mutate_time_penalty(tree: ET.ElementTree, max_penalty: int = 40):
    times = tree.findall('.//time[@penalty]')
    if not times: return False
    chosen_time = random.choice(times)
    new_penalty = random.randint(0, max_penalty)
    chosen_time.set('penalty', str(new_penalty))
    return True

def mutate_distribution_penalty(tree: ET.ElementTree, max_penalty: int = 40):
    distributions = tree.findall('.//distribution[@penalty]')
    if not distributions: return False
    dist = random.choice(distributions)
    new_penalty = str(random.randint(0, max_penalty))
    dist.set('penalty', new_penalty)
    return True

def mutate_travel_time(tree: ET.ElementTree):
    travels = tree.findall('.//travel[@value]')
    if not travels: return False
    travel = random.choice(travels)
    new_value = max(0, int(travel.get('value')) + random.choice([-1, 1]))
    travel.set('value', str(new_value))
    return True

def delete_random_student(tree: ET.ElementTree):
    students_element = tree.find('students')
    if students_element is None: return False
    students = students_element.findall('student')
    if not students: return False
    student_to_delete = random.choice(students)
    students_element.remove(student_to_delete)
    return True

def delete_random_course(tree: ET.ElementTree):
    courses_element = tree.find('courses')
    students_element = tree.find('students')
    if courses_element is None or students_element is None: return False
    courses = courses_element.findall('course')
    if not courses: return False
    course_to_delete = random.choice(courses)
    course_id = course_to_delete.get('id')
    if course_id is None: return False
    courses_element.remove(course_to_delete)
    for student in students_element.findall('student'):
        for course_enrollment in student.findall(f"./course[@id='{course_id}']"):
            student.remove(course_enrollment)
    return True
    
MUTATION_OPERATORS = {
    "room_capacity": mutate_room_capacity, "class_limit": mutate_class_limit,
    "time_penalty": mutate_time_penalty, "distribution_penalty": mutate_distribution_penalty,
    "travel_time": mutate_travel_time, "delete_student": delete_random_student,
    "delete_course": delete_random_course
}

def apply_constructive_mutation(tree: ET.ElementTree, probabilities: dict):
    mutations_applied = []
    while not mutations_applied:
        for op_name, probability in probabilities.items():
            if random.random() < probability:
                if MUTATION_OPERATORS[op_name](tree):
                    mutations_applied.append(op_name)
    return mutations_applied

def apply_deleter_mutation(tree: ET.ElementTree, probabilities: dict):
    deleter_names = list(probabilities.keys())
    weights = list(probabilities.values())
    chosen_deleter = random.choices(deleter_names, weights=weights, k=1)[0]
    MUTATION_OPERATORS[chosen_deleter](tree)
    return [chosen_deleter]

def setup_solver_config():
    if INIT_CFG_PATH.exists(): return
    if not CFG_DEFAULT.exists(): raise FileNotFoundError(CFG_DEFAULT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CFG_DEFAULT, INIT_CFG_PATH)
    with INIT_CFG_PATH.open("a") as f:
        f.write("\nTermination.StopWhenComplete=true\n")

def solve_time(xml_path: pathlib.Path, timeout: int, java_mem: str) -> float:
    cp = f"{JAR_PATH}:{os.path.join(DEPS_DIR, '*')}"
    cmd = ["java", f"-Xmx{java_mem}", "-cp", cp,
           "org.cpsolver.coursett.Test",
           str(INIT_CFG_PATH), str(xml_path), str(xml_path.parent)]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
        return time.perf_counter() - t0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return float("inf")

def calculate_initial_time(args):
    xml_path, timeout, java_mem = args
    instance_name = xml_path.stem
    print(f"  Calcul pour {instance_name}...")
    target_time = solve_time(xml_path, timeout, java_mem)
    if target_time == float("inf"):
        print(f"  -> {instance_name} a dépassé le timeout. Ignoré.")
        return None
    print(f"  -> {instance_name} résolu en {target_time:.2f}s.")
    return (instance_name, xml_path, target_time)

def process_instance(args):
    instance_name, original_xml_path, target_time, num_generations, n_children, timeout, java_mem, constructive_probs, deleter_probs = args
    
    print(f"[START] Traitement de {instance_name} (Temps cible: {target_time:.2f}s)")
    
    instance_out_dir = OUT_DIR / instance_name
    instance_out_dir.mkdir(parents=True, exist_ok=True)

    results_log = []
    
    current_parent_path = original_xml_path
    current_parent_time = target_time

    for gen in range(num_generations):
        try:
            if not current_parent_path.exists() or current_parent_path.stat().st_size == 0:
                print(f"  G{gen+1}: ERREUR - Le fichier parent est vide ou n'existe pas. Arrêt.")
                break 
            
            tree = ET.parse(current_parent_path)
        except (ET.ParseError, TimeoutError) as e:
            print(f"  G{gen+1}: ERREUR CRITIQUE de lecture du parent: {e}. Arrêt.")
            break

        children_of_generation = []
        
        for i in range(n_children):
            child_tree = ET.ElementTree(tree.getroot())
            
            if current_parent_time > target_time:
                mutations_applied = apply_deleter_mutation(child_tree, deleter_probs)
            else:
                mutations_applied = apply_constructive_mutation(child_tree, constructive_probs)
            
            child_path = instance_out_dir / f"{instance_name}_gen_{gen + 1}_child_{i + 1}.xml"
            child_tree.write(child_path, encoding="UTF-8", xml_declaration=True)

            child_time = solve_time(child_path, timeout, java_mem)
            child_fitness = float("inf") if child_time == float("inf") else abs(child_time - target_time)
            
            children_of_generation.append({
                "child_path": child_path, "child_time": child_time, 
                "child_fitness": child_fitness, "mutations": ", ".join(mutations_applied)
            })

        valid_children = [c for c in children_of_generation if c['child_fitness'] != float('inf')]

        best_child_of_gen = None
        if valid_children:
            best_child_of_gen = min(valid_children, key=lambda x: x['child_fitness'])
        
        # --- NETTOYAGE INTELLIGENT DES FICHIERS ---
        for child_data in children_of_generation:
            # On ne conserve que le fichier du meilleur enfant, on supprime les autres
            if child_data != best_child_of_gen:
                try:
                    os.remove(child_data['child_path'])
                except OSError:
                    pass # Ignorer si le fichier n'existe pas
        
        # Log des résultats (uniquement le meilleur enfant est maintenant pertinent)
        if best_child_of_gen:
            results_log.append({
                "base_instance": instance_name, "generation": gen + 1,
                "mutations_applied": best_child_of_gen["mutations"],
                "solve_time_s": best_child_of_gen["child_time"], "target_time_s": target_time,
                "fitness_score": best_child_of_gen["child_fitness"]
            })
            
            current_parent_path = best_child_of_gen['child_path']
            current_parent_time = best_child_of_gen['child_time']
        else:
            print(f"  G{gen+1}: Tous les enfants ont échoué. Conservation du parent.")

    print(f"[DONE] Traitement de {instance_name} terminé.")
    return (instance_name, "OK", results_log)

def main():
    # ================== PARAMÈTRES UTILISATEUR ==================
    # Paramètres réduits pour générer moins de fichiers
    NUM_GENERATIONS = 3
    N_CHILDREN_PER_GENERATION = 3
    TIMEOUT = 2000
    JAVA_MEM = "8g"
    
    #CONSTRUCTIVE_PROBS = {
     #   "distribution_penalty": 0.40, "class_limit": 0.30,
     #   "room_capacity": 0.20, "time_penalty": 0.15, "travel_time": 0.10,
    #}
    
    #DELETER_PROBS = { "delete_student": 0.80, "delete_course":  0.20 }


    # Priorité aux mutations les plus impactantes et destructrices
    #CONSTRUCTIVE_PROBS = {
     #   "time_penalty": 0.40,          # Augmenté (très impactant)
      #  "class_limit": 0.20,
       # "distribution_penalty": 0.20,
       # "room_capacity": 0.10,
       # "travel_time": 0.10,
    #}

    #DELETER_PROBS = { 
    #    "delete_student": 0.60, 
    #    "delete_course":  0.40          # Augmenté (très destructeur)
    #}


    # Priorité aux mutations sur les salles et les temps de trajet
    CONSTRUCTIVE_PROBS = {
        "room_capacity": 0.50,         # Fortement augmenté
        "travel_time": 0.30,           # Fortement augmenté
        "class_limit": 0.10,
        "distribution_penalty": 0.05,  # Réduit
        "time_penalty": 0.05,          # Réduit
    }

    # On garde la même logique de suppression
    DELETER_PROBS = { "delete_student": 0.80, "delete_course":  0.20 }
        
    NUM_WORKERS = max(1, cpu_count() - 2)
    
    #global ROOT, JAR_PATH, DEPS_DIR, CFG_DEFAULT, XML_DIR, OUT_DIR, INIT_CFG_PATH
    #ROOT = pathlib.Path(__file__).resolve().parent
    #JAR_PATH = ROOT / "cpsolver-itc2019-master/target/cpsolver-itc2019-1.0-SNAPSHOT.jar"
    #DEPS_DIR = ROOT / "cpsolver-itc2019-master/target/dependency"
    #CFG_DEFAULT = ROOT / "cpsolver-itc2019-master/configuration/default.cfg"
    #XML_DIR = ROOT / "data"
    #OUT_DIR = ROOT / "ga_final_instances_v10_Contraintes_Phy"
    #INIT_CFG_PATH = OUT_DIR / "init_only.cfg"
    # ==========================================================

    if OUT_DIR.exists():
        print(f"Nettoyage : Suppression de l'ancien dossier de résultats '{OUT_DIR}'...")
        shutil.rmtree(OUT_DIR)
    
    print("Démarrage de l'algorithme génétique V10 (Gestion Espace Disque)...")
    setup_solver_config()
    
    print(f"\nÉtape 1: Calcul des temps initiaux sur {NUM_WORKERS} processus...")
    original_instances_paths = sorted(list(XML_DIR.glob("*.xml")))
    initial_calc_args = [(path, TIMEOUT * 2, JAVA_MEM) for path in original_instances_paths]
    
    ga_tasks = []
    with Pool(processes=NUM_WORKERS) as pool:
        initial_results = pool.map(calculate_initial_time, initial_calc_args)
    
    for result in initial_results:
        if result:
            instance_name, xml_path, target_time = result
            ga_tasks.append((instance_name, xml_path, target_time, NUM_GENERATIONS, N_CHILDREN_PER_GENERATION, TIMEOUT, JAVA_MEM, CONSTRUCTIVE_PROBS, DELETER_PROBS))
    
    if not ga_tasks:
        print("\nFATAL: Aucune instance n'a pu être résolue.")
        return

    print(f"\nÉtape 2: Lancement du GA sur {len(ga_tasks)} instances valides...")
    all_results = []
    with Pool(processes=NUM_WORKERS) as pool:
        process_outputs = pool.map(process_instance, ga_tasks)

    for instance_name, status, results in process_outputs:
        if status == "OK": all_results.extend(results)
        else: print(f"Erreur pour {instance_name}: {status}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        output_csv_path = OUT_DIR / "ga_results.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nRésultats complets sauvegardés dans : {output_csv_path}")
    else:
        print("\nAucun résultat n'a été généré.")
        
    print("Algorithme terminé.")

if __name__ == '__main__':
    main()