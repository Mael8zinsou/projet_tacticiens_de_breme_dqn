import sys
import os
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_env.tacticiens_env import TacticiensEnv

class DQNAgent:
    """Agent d'apprentissage par renforcement utilisant l'algorithme Deep Q-Network"""

    def __init__(self, state_shape, action_size, config=None):
        """
        Initialise l'agent DQN avec les paramètres spécifiés

        Args:
            state_shape: Forme de l'état (observation)
            action_size: Taille de l'espace d'action
            config: Dictionnaire de configuration (optionnel)
        """
        self.state_shape = state_shape
        self.action_size = action_size

        # Paramètres de configuration avec valeurs par défaut
        config = config or {}
        self.memory_size = config.get('memory_size', 10000)
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)

        # Initialisation de la mémoire et des modèles
        self.memory = deque(maxlen=self.memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Statistiques d'entraînement
        self.train_count = 0
        self.avg_loss = 0

    def _build_model(self):
        """Construit le réseau de neurones pour l'approximation de la fonction Q"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copie les poids du modèle principal vers le modèle cible"""
        self.target_model.set_weights(self.model.get_weights())
        logging.info("Modèle cible mis à jour")

    def remember(self, state, action, reward, next_state, done):
        """Stocke l'expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        """
        Choisit une action selon la politique epsilon-greedy

        Args:
            state: État actuel
            valid_moves: Liste des mouvements valides

        Returns:
            int: Index de l'action choisie
        """
        if not valid_moves:
            return 0
        
        print(f"Valid moves: {valid_moves}")

        # Exploration: choisir une action aléatoire
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, len(valid_moves))

        # Exploitation: prédire les valeurs Q
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        # Filtrer les actions valides
        valid_q_values = [(i, q_values[i]) for i in range(len(q_values)) if i < len(valid_moves)]

        # Si aucune action valide n'a de valeur Q, choisir aléatoirement
        if not valid_q_values:
            return np.random.randint(0, len(valid_moves))

        # Choisir l'action avec la plus grande valeur Q
        return max(valid_q_values, key=lambda x: x[1])[0]

    def replay(self, batch_size):
        """
        Entraîne le modèle sur un mini-batch d'expériences

        Args:
            batch_size: Taille du mini-batch
        """
        if len(self.memory) < batch_size:
            return

        start_time = time.time()

        # Échantillonner un mini-batch de la mémoire
        minibatch = random.sample(self.memory, batch_size)

        # Préparer les données d'entraînement
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Prédire les valeurs Q actuelles et futures
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Mettre à jour les valeurs cibles pour les actions choisies
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Entraîner le modèle
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        # Mettre à jour les statistiques
        self.train_count += 1
        self.avg_loss = history.history['loss'][0]

        # Réduire epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        end_time = time.time()
        logging.info(f"Entraînement batch #{self.train_count}: loss={self.avg_loss:.4f}, temps={end_time-start_time:.2f}s, epsilon={self.epsilon:.4f}")

    def load(self, name):
        """Charge les poids du modèle depuis un fichier"""
        try:
            self.model.load_weights(name)
            self.update_target_model()
            logging.info(f"Modèle chargé depuis {name}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {e}")

    def save(self, name):
        """Sauvegarde les poids du modèle dans un fichier"""
        try:
            self.model.save_weights(name)
            logging.info(f"Modèle sauvegardé dans {name}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du modèle: {e}")

def train_dqn(episodes=100, batch_size=32, update_target_every=10, save_freq=10):
    """
    Entraîne un agent DQN sur l'environnement Tacticiens

    Args:
        episodes: Nombre d'épisodes d'entraînement
        batch_size: Taille du mini-batch pour l'entraînement
        update_target_every: Fréquence de mise à jour du modèle cible
        save_freq: Fréquence de sauvegarde du modèle

    Returns:
        tuple: (agent, scores)
    """
    # Créer l'environnement
    env = TacticiensEnv(opponent_type='random')

    # Obtenir la forme de l'état et la taille de l'espace d'action
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # Créer l'agent DQN
    agent = DQNAgent(state_shape, action_size)

    # Variables pour suivre les performances
    scores = []
    epsilons = []
    wins = []
    episode_lengths = []

    # Créer un dossier pour sauvegarder les modèles
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    # Entraînement
    total_start_time = time.time()
    logging.info(f"Début de l'entraînement pour {episodes} épisodes")

    for e in range(episodes):
        episode_start_time = time.time()

        # Réinitialiser l'environnement
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        # Jouer un épisode
        while not done and steps < 1000:  # Limite de 1000 étapes par épisode
            # Choisir une action
            action = agent.act(state, env.valid_moves)

            # Exécuter l'action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)

            # Mettre à jour l'état
            state = next_state

            # Accumuler le score et incrémenter les étapes
            score += reward
            steps += 1

            # Entraîner le modèle
            if len(agent.memory) >= batch_size and steps % 4 == 0:
                agent.replay(batch_size)

        # Mettre à jour le modèle cible périodiquement
        if e % update_target_every == 0:
            agent.update_target_model()

        # Enregistrer les statistiques
        scores.append(score)
        epsilons.append(agent.epsilon)
        wins.append(1 if info.get('win', False) else 0)
        episode_lengths.append(steps)

        # Calculer les statistiques
        episode_time = time.time() - episode_start_time
        win_rate = sum(wins[-100:]) / min(len(wins), 100)
        avg_score = np.mean(scores[-100:])
        avg_length = np.mean(episode_lengths[-100:])

        # Afficher les informations sur l'épisode
        logging.info(f"Episode {e+1}/{episodes}: score={score:.2f}, steps={steps}, win={info.get('win', False)}, time={episode_time:.2f}s")

        # Afficher les statistiques périodiquement
        if (e+1) % 10 == 0:
            logging.info(f"Statistiques (100 derniers épisodes): win_rate={win_rate:.2f}, avg_score={avg_score:.2f}, avg_length={avg_length:.2f}")

        # Sauvegarder le modèle périodiquement
        if (e+1) % save_freq == 0:
            agent.save(f"{models_dir}/dqn_agent_episode_{e+1}.h5")

    # Sauvegarder le modèle final
    agent.save(f"{models_dir}/dqn_agent_final.h5")

    # Calculer le temps total d'entraînement
    total_time = time.time() - total_start_time
    logging.info(f"Entraînement terminé en {total_time:.2f} secondes")

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.title('Score par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')

    plt.subplot(2, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Epsilon')

    plt.subplot(2, 2, 3)
    plt.plot(wins)
    plt.title('Victoires par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Victoire (1=oui, 0=non)')

    plt.subplot(2, 2, 4)
    plt.plot(episode_lengths)
    plt.title('Longueur des épisodes')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')

    plt.tight_layout()
    plt.savefig(f"{models_dir}/learning_curves.png")

    # Fermer l'environnement
    env.close()

    return agent, scores

if __name__ == "__main__":
    # Définir les paramètres d'entraînement
    episodes = 100
    batch_size = 32
    update_target_every = 10
    save_freq = 10

    # Entraîner l'agent
    start_time = time.time()
    agent, scores = train_dqn(episodes, batch_size, update_target_every, save_freq)
    end_time = time.time()

    # Afficher les résultats
    training_time = end_time - start_time
    print(f"Entraînement terminé en {training_time:.2f} secondes")
    print(f"Score moyen sur les 100 derniers épisodes: {np.mean(scores[-100:]):.2f}")