import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv
from train.train_dqn import DQNAgent

def test_agent(model_path, num_episodes=10, render=True):
    # Créer l'environnement
    env = TacticiensEnv(opponent_type='random')

    # Obtenir la forme de l'état et la taille de l'espace d'action
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # Créer l'agent DQN
    agent = DQNAgent(state_shape, action_size)

    # Charger les poids du modèle
    agent.load(model_path)

    # Désactiver l'exploration
    agent.epsilon = 0.0

    # Variables pour suivre les performances
    scores = []
    win_count = 0

    # Jouer plusieurs épisodes
    for e in range(num_episodes):
        # Réinitialiser l'environnement
        state = env.reset()
        done = False
        score = 0
        steps = 0

        print(f"Épisode {e+1}/{num_episodes}")

        # Jouer un épisode
        while not done:
            # Choisir une action
            action = agent.act(state, env.valid_moves)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)

            # Afficher l'état du jeu
            if render:
                env.render()
                time.sleep(0.5)  # Pause pour mieux visualiser

            # Mettre à jour l'état
            state = next_state

            # Accumuler le score
            score += reward
            steps += 1

            # Afficher les informations sur l'étape
            print(f"Étape {steps}, Action: {action}, Récompense: {reward}, Score total: {score}")

            # Vérifier si l'agent a gagné
            if done and 'win' in info and info['win']:
                win_count += 1
                print("L'agent a gagné!")
            elif done:
                print("L'agent a perdu ou l'épisode est terminé.")

        # Enregistrer le score
        scores.append(score)

        print(f"Score final: {score}")
        print("-" * 50)

    # Afficher les statistiques
    print(f"Score moyen sur {num_episodes} épisodes: {np.mean(scores):.2f}")
    print(f"Taux de victoire: {win_count/num_episodes:.2%}")

    # Tracer la courbe des scores
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Score par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

    # Fermer l'environnement
    env.close()

    return scores, win_count/num_episodes

if __name__ == "__main__":
    # Vérifier si un modèle a été spécifié
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/dqn_agent_final.h5"

    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Le modèle {model_path} n'existe pas.")
        print("Veuillez d'abord entraîner un modèle ou spécifier un chemin valide.")
        sys.exit(1)

    # Tester l'agent
    print(f"Test de l'agent avec le modèle {model_path}")
    scores, win_rate = test_agent(model_path, num_episodes=5, render=True)