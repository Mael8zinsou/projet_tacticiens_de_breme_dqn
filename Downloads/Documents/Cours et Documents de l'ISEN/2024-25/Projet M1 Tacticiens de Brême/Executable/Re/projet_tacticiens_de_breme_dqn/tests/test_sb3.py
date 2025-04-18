import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv

def test_agent(model_path, num_episodes=5, render=True):
    """
    Teste un agent DQN entraîné avec Stable Baselines 3.

    Args:
        model_path (str): Chemin vers le modèle SB3 (.zip)
        num_episodes (int): Nombre d'épisodes de test
        render (bool): Afficher l'environnement pendant le test
    """
    try:
        # Charger le modèle SB3
        model = DQN.load(model_path)
        logging.info(f"Modèle chargé depuis {model_path}")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle: {e}")
        return None, 0

    # Créer l'environnement
    env = TacticiensEnv(opponent_type='random')

    # Variables pour suivre les performances
    scores = []
    win_count = 0
    episode_lengths = []
    total_steps = 0

    # Jouer plusieurs épisodes
    for e in range(num_episodes):
        # Réinitialiser l'environnement
        state, _ = env.reset()
        terminated = False
        truncated = False
        score = 0
        steps = 0

        logging.info(f"\nÉpisode {e+1}/{num_episodes}")

        # Jouer un épisode
        while not (terminated or truncated):
            # Prédire l'action avec le modèle SB3
            action, _ = model.predict(state, deterministic=True)

            # Exécuter l'action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Afficher l'état du jeu
            if render:
                env.render()
                time.sleep(0.5)  # Pause pour mieux visualiser

            # Mettre à jour l'état
            state = next_state

            # Accumuler le score et les étapes
            score += reward
            steps += 1
            total_steps += 1

            # Afficher les informations sur l'étape
            logging.info(f"Étape {steps}, Action: {action}, Récompense: {reward}, Score total: {score}")

        # Vérifier le résultat de l'épisode
        if info.get('win', False):
            win_count += 1
            logging.info("✅ L'agent a gagné!")
        else:
            logging.info("❌ L'agent a perdu ou l'épisode est terminé.")

        # Enregistrer les statistiques
        scores.append(score)
        episode_lengths.append(steps)

        logging.info(f"Score final: {score}")
        logging.info(f"Nombre d'étapes: {steps}")
        logging.info("-" * 50)

    # Calculer les statistiques
    avg_score = np.mean(scores)
    avg_length = np.mean(episode_lengths)
    win_rate = win_count / num_episodes

    # Afficher les statistiques finales
    logging.info("\nRésultats des tests:")
    logging.info(f"Score moyen: {avg_score:.2f}")
    logging.info(f"Taux de victoire: {win_rate:.2%}")
    logging.info(f"Longueur moyenne des épisodes: {avg_length:.2f}")
    logging.info(f"Nombre total d'étapes: {total_steps}")

    # Créer les visualisations
    plt.figure(figsize=(15, 5))

    # Score par épisode
    plt.subplot(1, 3, 1)
    plt.plot(scores, 'b-')
    plt.title('Score par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.grid(True)

    # Longueur des épisodes
    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths, 'r-')
    plt.title('Longueur des épisodes')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')
    plt.grid(True)

    # Distribution des scores
    plt.subplot(1, 3, 3)
    plt.hist(scores, bins=min(len(scores), 10), color='g')
    plt.title('Distribution des scores')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.grid(True)

    plt.tight_layout()

    # Sauvegarder le graphique
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/test_results.png")
    logging.info(f"Graphique sauvegardé dans {results_dir}/test_results.png")

    plt.show()

    # Fermer l'environnement
    env.close()

    return scores, win_rate

if __name__ == "__main__":
    # Vérifier si un modèle a été spécifié
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # model_path = "./models/sb3_dqn/dqn_final_model.zip"  # Chemin par défaut pour le modèle SB3
        model_path = "./best_models/best_model_5000_-5.00.zip"

    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        logging.error(f"Le modèle {model_path} n'existe pas.")
        logging.error("Veuillez d'abord entraîner un modèle ou spécifier un chemin valide.")
        sys.exit(1)

    # Tester l'agent
    logging.info(f"Test de l'agent avec le modèle {model_path}")
    try:
        scores, win_rate = test_agent(model_path, num_episodes=5, render=True)
        if scores is not None:
            logging.info("Test terminé avec succès!")
    except Exception as e:
        logging.error(f"Erreur lors du test: {e}")
        sys.exit(1)