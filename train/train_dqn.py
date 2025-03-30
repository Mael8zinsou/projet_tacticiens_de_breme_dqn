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

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv

# Classe pour l'agent DQN
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # facteur d'actualisation
        self.epsilon = 1.0   # taux d'exploration initial
        self.epsilon_min = 0.01  # taux d'exploration minimal
        self.epsilon_decay = 0.995  # taux de décroissance de l'exploration
        self.learning_rate = 0.001  # taux d'apprentissage
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Réseau de neurones pour l'approximation de la fonction Q
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copier les poids du modèle principal vers le modèle cible
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Stocker l'expérience dans la mémoire
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        # Choisir une action selon la politique epsilon-greedy
        if np.random.rand() <= self.epsilon:
            # Exploration: choisir une action aléatoire parmi les mouvements valides
            return np.random.randint(0, len(valid_moves)) if valid_moves else 0

        # Exploitation: prédire les valeurs Q pour toutes les actions
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        # Filtrer les actions valides
        valid_q_values = [(i, q_values[i]) for i in range(len(q_values)) if i < len(valid_moves)]

        # Si aucune action valide n'a de valeur Q, choisir aléatoirement
        if not valid_q_values:
            return np.random.randint(0, len(valid_moves)) if valid_moves else 0

        # Sinon, choisir l'action avec la plus grande valeur Q parmi les actions valides
        return max(valid_q_values, key=lambda x: x[1])[0]

    def replay(self, batch_size ):
        
        # Entraîner le modèle sur un mini-batch d'expériences
        if len(self.memory) < batch_size:
            return
        
        
        print("entrainement de l'agent")
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            start_time = time.time()  # Démarrer le timer
            target = reward
            if not done:
                # Utiliser le modèle cible pour calculer la valeur Q future
                target = reward + self.gamma * np.amax(self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])

            # Mettre à jour la valeur Q pour l'action choisie
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target

            # Entraîner le modèle
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
            
            end_time = time.time()  # Arrêter le timer
            print(f"Temps d'entraînement pour ce mini-batch : {end_time - start_time:.4f} secondes")

        # Réduire epsilon pour diminuer l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_dqn(episodes=10, batch_size=32, update_target_every=10):
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

    # Créer un dossier pour sauvegarder les modèles
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Entraînement
    for e in range(episodes):
        # Réinitialiser l'environnement
        state = env.reset()
        done = False
        score = 0
        
        # Afficher le début de l'épisode
        print(f"Début de l'épisode {e+1}/{episodes}...")

        # Jouer un épisode
        while not done:
            # Choisir une action
            action = agent.act(state, env.valid_moves)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)

            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)

            # Mettre à jour l'état
            state = next_state

            # Accumuler le score
            score += reward

            
            # Entraîner le modèle tout les 2 episodes
            if e % 2 == 0 and e != 0:
                agent.replay(batch_size)
            
            print(f"Action: {action}, Reward: {reward}, Done: {done} , episode: {e}")

        # Mettre à jour le modèle cible périodiquement
        if e % update_target_every == 0:
            agent.update_target_model()

        # Enregistrer le score et epsilon
        scores.append(score)
        epsilons.append(agent.epsilon)

        # Afficher les informations sur l'épisode
        print(f"Episode: {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

        # Sauvegarder le modèle périodiquement
        if (e+1) % 100 == 0:
            agent.save(f"{models_dir}/dqn_agent_episode_{e+1}.h5")

    # Sauvegarder le modèle final
    agent.save(f"{models_dir}/dqn_agent_final.h5")

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(f"{models_dir}/learning_curves.png")
    plt.show()

    # Fermer l'environnement
    env.close()

    return agent, scores

if __name__ == "__main__":
    # Définir les paramètres d'entraînement
    episodes = 10
    batch_size = 32
    update_target_every = 10

    # Entraîner l'agent
    print(f"Début de l'entraînement pour {episodes} épisodes...")
    start_time = time.time()

    agent, scores = train_dqn(episodes, batch_size, update_target_every)

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Entraînement terminé en {training_time:.2f} secondes")
    print(f"Score moyen sur les 100 derniers épisodes: {np.mean(scores[-100:]):.2f}")