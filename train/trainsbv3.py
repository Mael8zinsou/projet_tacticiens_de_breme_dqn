import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
import sys
from stable_baselines3.common.env_checker import check_env
import time
import tensorflow as tf
import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
import csv
import random

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tacticiens_env import TacticiensEnv

# Vérification de l'environnement
print("PyTorch GPU Available: ", torch.cuda.is_available())


class TrainingInfoCallback(BaseCallback):
    """
    Callback personnalisé pour afficher des informations pendant l'entraînement
    et enregistrer les métriques dans TensorBoard.
    """
    def __init__(self, verbose=1):
        super(TrainingInfoCallback, self).__init__(verbose)
        self.episode_rewards = 0
        self.episode_lengths = []
        self.episode_wins = []  # Liste pour stocker les victoires (1 pour victoire, 0 sinon)
        self.episode_turns = []  # Liste pour stocker le nombre de coups par partie
        self.episode_counter = 0  # Compteur d'épisodes pour les moyennes toutes les 10 parties
        self.global_turns = []  # Stocke les métriques globales des tours
        self.global_wins = []  # Stocke les métriques globales des victoires
        self.global_counter = 0  # Compteur global pour afficher toutes les 100 parties
        self.real_episode_counter = 0  # Compteur d'épisodes réel pour les moyennes globales

    def _on_step(self) -> bool:
        # Démarrer le timer pour l'entraînement sur un minibatch
        self.train_start_time = time.time()

        # SB3 effectue l'entraînement automatiquement ici
        # Arrêter le timer après l'entraînement
        train_time = time.time() - self.train_start_time


        if self.num_timesteps % 100 == 0:
            print(f"Step: {self.num_timesteps}, Episode: {self.episode_counter}, Global Counter: {self.global_counter}")
            print(f"Reward: {self.episode_rewards:.2f}")
            print(f"Temps pris pour l'entraînement sur un minibatch : {train_time:.4f} secondes")

        self.episode_rewards += self.locals["rewards"][0]
        for info in self.locals["infos"]:
            if "turns" in info:
                turns = info["turns"]
                print(f"Nombre de coups dans la partie : {turns} , victoire détectée")
                self.episode_turns.append(turns)

                is_win = 1 if info.get("win", False) else 0
                self.episode_wins.append(is_win)
                
                #enregistrement 
                # Déterminer le gagnant
                winner = "agent_dqn" if is_win == 1 else "minimax"

                # Stocker les informations de la partie dans le dictionnaire
                partie_key = f"partie_{self.real_episode_counter + 1}"
                self.parties[partie_key] = {
                    "nombre_de_coups": turns,
                    "vainqueur": winner
                }

                self.episode_counter += 1
                self.global_counter += 1
                self.real_episode_counter += 1

                if self.episode_counter == 1:
                    avg_turns = self.episode_turns if self.episode_turns else 0
                    win_rate = self.episode_wins if self.episode_wins else 0

                    self.global_turns.append(avg_turns)
                    self.global_wins.append(win_rate)

                    print("Épisode : ", self.real_episode_counter)
                    print(f"Après 1 partie :")
                    print(f"  Nombre de coups : {avg_turns:.2f}")
                    print(f"  victoire : {win_rate:.2f}")
                    print(f"Récompense de la partie : {self.episode_rewards:.2f}")

                    self.episode_turns = []
                    self.episode_wins = []
                    self.episode_counter = 0
                    self.episode_rewards = 0

                if self.global_counter % 5 == 0:
                    global_avg_turns = np.mean(self.global_turns) if self.global_turns else 0
                    global_win_rate = np.mean(self.global_wins) if self.global_wins else 0

                    print(f"Après 5 parties :")
                    print(f"  Moyenne globale des coups : {global_avg_turns:.2f}")
                    print(f"  Taux de victoire global : {global_win_rate:.2f}")

        return True

    def _on_training_end(self) -> None:
        print("Entraînement terminé ! . sauvegarde des résultats")
        self.save_to_csv()
    
    def save_to_csv(self):
        """Sauvegarder les données des parties dans un fichier CSV."""
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Écrire l'en-tête
            writer.writerow(["Partie", "Nombre de Coups", "Vainqueur"])
            # Écrire les données
            for partie, data in self.parties.items():
                writer.writerow([partie, data["nombre_de_coups"], data["vainqueur"]])
        print(f"Données des parties sauvegardées dans {self.csv_path}")

class MaskedDQNPolicy(DQNPolicy):
    def __init__(self, *args, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, **kwargs):
        super(MaskedDQNPolicy, self).__init__(*args, **kwargs)
        self.epsilon = epsilon  # Valeur initiale d'epsilon
        self.epsilon_min = epsilon_min  # Valeur minimale d'epsilon
        self.epsilon_decay = epsilon_decay  # Facteur de décroissance d'epsilon

    def forward(self, obs, deterministic=False):
        # Obtenir les Q-values pour toutes les actions
        print("Appel de la méthode `forward` dans MaskedDQNPolicy")
        q_values = self.q_net(obs)
        print(self._last_obs_info["valid_actions"])

        # Appliquer le masque d'actions
        if "valid_actions" in self._last_obs_info:
            valid_actions = self._last_obs_info["valid_actions"]
            mask = torch.full_like(q_values, float('-inf'))  # Initialiser le masque avec -inf
            mask[:, valid_actions] = 0  # Garder les Q-values des actions valides
            q_values += mask
        print(q_values)

        # Stratégie epsilon-greedy
        if not deterministic and torch.rand(1).item() < self.epsilon:
            # Exploration : choisir une action aléatoire parmi les actions valides
            actions = torch.tensor([torch.choice(valid_actions) for valid_actions in self._last_obs_info["valid_actions"]])
        else:
            # Exploitation : choisir l'action avec la plus grande Q-value
            actions = q_values.argmax(dim=1)
            print(f"actions : {actions}")

        # Réduire epsilon (décroissance)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print("je suis dans le masked dqn policy")
        return actions, q_values


def train_sb3_dqn(episodes=100000, save_path="./models/sb3_dqn"):
    # Nom du modèle
    model_name = "dqn_masked_final_model"
    model_path = os.path.join(save_path, model_name + ".zip")
    

    # Vérifier si le modèle existe déjà
    if os.path.exists(model_path):
        print(f"Modèle existant trouvé : {model_path}. Chargement du modèle...")
        env = Monitor(TacticiensEnv(opponent_type='minimax'))
        model = DQN.load(model_path, env=env)
    else:
        print(f"Aucun modèle existant trouvé. Création d'un nouveau modèle : {model_name}")
        # Créer l'environnement
        env = Monitor(TacticiensEnv(opponent_type='minimax'))

        # Définir les arguments pour la politique
        policy_kwargs = dict(
            net_arch=[256, 128, 64],  # Trois couches
            epsilon=1.0,  # Valeur initiale d'epsilon
            epsilon_min=0.05,  # Valeur minimale d'epsilon
            epsilon_decay=0.999  # Facteur de décroissance d'epsilon
        )

        # Créer le modèle DQN avec la politique masquée
        model = DQN(
            MaskedDQNPolicy,  # Utiliser la politique masquée avec epsilon-greedy
            env,
            learning_rate=0.001,
            buffer_size=100000,
            learning_starts=100,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            train_freq=5,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",  # Répertoire pour les logs TensorBoard
            policy_kwargs=policy_kwargs,  # Passer les arguments de la politique
            device="cuda" if torch.cuda.is_available() else "cpu",  # Utiliser le GPU si disponible
        )
        
    print(f"Utilisation de la politique : {model.policy}")

    # Créer un répertoire pour sauvegarder les modèles
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Ajouter un callback pour sauvegarder le modèle périodiquement
    callback_list = CallbackList([
        CheckpointCallback(save_freq=1000, save_path=save_path, name_prefix="dqn_masked"),
        TrainingInfoCallback(verbose=1,)
    ])

    # Entraîner le modèle
    model.learn(
        total_timesteps=episodes,
        callback=callback_list,
        log_interval=100,
        tb_log_name="DQN_Tacticiens"
    )

    # Sauvegarder le modèle final
    model.save(model_path)
    print(f"Modèle sauvegardé sous : {model_path}")

    # Fermer l'environnement
    env.close()


if __name__ == "__main__":
    # Définir les paramètres d'entraînement
    episodes = 100000
    save_path = "./models/sb3_dqn"

    # Entraîner l'agent
    print(f"Starting training for {episodes} timesteps...")
    train_sb3_dqn(episodes, save_path)
    print("Training completed!")
