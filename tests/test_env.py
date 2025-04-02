import sys
import os
import numpy as np
import time

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv

def test_env_basics():
    """Test des fonctionnalités de base de l'environnement"""
    env = TacticiensEnv(opponent_type='random')

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

        obs, reward, done, truncated, info = env.step(action)
        print(f"Récompense: {reward}")
        print(f"Épisode terminé: {done}")
        print(f"Info: {info}")
    else:
        print("Aucun mouvement valide disponible")

    # Fermer l'environnement
    env.close()
    return True

def test_full_episode():
    """Test d'un épisode complet"""
    env = TacticiensEnv(opponent_type='random')
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:  # Limite de 100 étapes pour éviter les boucles infinies
        # Choisir une action aléatoire parmi les mouvements valides
        if len(env.valid_moves) > 0:
            action = np.random.randint(0, len(env.valid_moves))
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            print(f"Étape {steps}, Récompense: {reward}, Total: {total_reward}")

            # Afficher le plateau
            env.render()
        else:
            print("Aucun mouvement valide disponible")
            break

    print(f"Épisode terminé après {steps} étapes avec une récompense totale de {total_reward}")

    # Fermer l'environnement
    env.close()
    return steps, total_reward

if __name__ == "__main__":
    print("Test des fonctionnalités de base de l'environnement...")
    test_env_basics()

    print("\nTest d'un épisode complet...")
    steps, total_reward = test_full_episode()