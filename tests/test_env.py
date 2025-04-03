import sys
import os
import numpy as np
import time
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv

def test_env_basics():
    """Test des fonctionnalités de base de l'environnement"""
    print("\n=== Test des fonctionnalités de base ===")

    try:
        # Création de l'environnement
        env = TacticiensEnv(opponent_type='random')
        print("Environnement créé avec succès")

        # Test du reset
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Nombre de mouvements valides: {len(env.valid_moves)}")
        print(f"Info initial: {info}")

        # Test d'une action aléatoire
        if len(env.valid_moves) > 0:
            action = np.random.randint(0, len(env.valid_moves))
            print(f"Action choisie: {action}")
            print(f"Mouvement correspondant: {env.valid_moves[action]}")

            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Récompense: {reward}")
            print(f"Épisode terminé: {terminated}")
            print(f"Épisode tronqué: {truncated}")
            print(f"Info: {info}")
        else:
            print("Aucun mouvement valide disponible")

        # Fermer l'environnement
        env.close()
        return True

    except Exception as e:
        print(f"Erreur lors du test de base: {e}")
        return False

def test_full_episode():
    """Test d'un épisode complet"""
    print("\n=== Test d'un épisode complet ===")

    try:
        # Création et initialisation de l'environnement
        env = TacticiensEnv(opponent_type='random')
        obs, info = env.reset()

        # Variables de suivi
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        max_steps = 100  # Limite pour éviter les boucles infinies

        # Boucle principale de l'épisode
        start_time = time.time()
        while not (terminated or truncated) and steps < max_steps:
            # Choisir une action aléatoire parmi les mouvements valides
            if len(env.valid_moves) > 0:
                action = np.random.randint(0, len(env.valid_moves))

                # Exécuter l'action
                obs, reward, terminated, truncated, info = env.step(action)

                # Mettre à jour les statistiques
                total_reward += reward
                steps += 1

                # Afficher les informations de l'étape
                print(f"Étape {steps}, Action: {action}, Récompense: {reward}, Total: {total_reward}")
                print(f"Mouvements valides: {len(env.valid_moves)}")

                # Afficher le plateau tous les 5 mouvements pour ne pas surcharger la sortie
                if steps % 5 == 0:
                    env.render()
            else:
                print("Aucun mouvement valide disponible")
                break

        # Calculer la durée de l'épisode
        duration = time.time() - start_time

        # Afficher le résumé de l'épisode
        print("\n=== Résumé de l'épisode ===")
        print(f"Durée: {duration:.2f} secondes")
        print(f"Nombre d'étapes: {steps}")
        print(f"Récompense totale: {total_reward}")
        print(f"Terminé: {terminated}")
        print(f"Tronqué: {truncated}")

        # Fermer l'environnement
        env.close()
        return steps, total_reward

    except Exception as e:
        print(f"Erreur lors du test d'épisode complet: {e}")
        return 0, 0

def visualize_observation(obs):
    """Visualise l'observation sous forme de représentation textuelle du plateau"""
    if obs is None:
        return

    # Créer une représentation du plateau
    board = [[' ' for _ in range(5)] for _ in range(5)]

    # Remplir le plateau avec les pièces
    for y in range(5):
        for x in range(5):
            # Vérifier les pièces bleues (canaux 0-3)
            for i in range(4):
                if obs[y, x, i] == 1:
                    board[y][x] = f"B{i+1}"

            # Vérifier les pièces orange (canaux 4-7)
            for i in range(4):
                if obs[y, x, i+4] == 1:
                    board[y][x] = f"O{i+1}"

    # Afficher le plateau
    print("\nÉtat du plateau:")
    print("  " + " ".join([f"{i}" for i in range(5)]))
    for y in range(5):
        print(f"{y} " + " ".join([f"{board[y][x]:2}" for x in range(5)]))

if __name__ == "__main__":
    print(f"=== Tests de l'environnement Tacticiens - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Test des fonctionnalités de base
    basic_test_success = test_env_basics()

    # Test d'un épisode complet si le test de base a réussi
    if basic_test_success:
        steps, total_reward = test_full_episode()

        print("\n=== Résultat final ===")
        print(f"Tests terminés avec succès. Épisode de {steps} étapes et récompense totale de {total_reward}.")
    else:
        print("\n=== Résultat final ===")
        print("Les tests de base ont échoué. Correction nécessaire avant de poursuivre.")