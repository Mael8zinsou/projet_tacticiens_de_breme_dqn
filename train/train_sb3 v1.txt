import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np


import sys

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_env.tacticiens_env import TacticiensEnv

from stable_baselines3.common.env_checker import check_env

env = TacticiensEnv(opponent_type='random')
check_env(env, warn=True)

class TrainingInfoCallback(BaseCallback):
    """
    Callback personnalisé pour afficher des informations pendant l'entraînement.
    """
    def __init__(self, verbose=1):
        super(TrainingInfoCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []  # Liste pour stocker les victoires (1 pour victoire, 0 sinon)

    def _on_step(self) -> bool:
        #print("self.locals keys:", self.locals.keys())
        # Collecter les récompenses et longueurs des épisodes
        #print(self.locals["infos"])
        for info in self.locals["infos"]:
            if  "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                is_win = 1 if info.get('win', False) else 0
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_wins.append(is_win)
                #print(f"Episode terminé: récompense = {reward}, longueur = {length}, victoire = {is_win}")

        # Afficher des informations toutes les 1000 étapes
        if self.n_calls % 1000 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            avg_moves = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            wins_last_100 = sum(self.episode_wins[-100:])  # Nombre de victoires sur les 100 derniers épisodes
            print(f"Step: {self.num_timesteps}")
            print(f"  Score moyen (100 derniers épisodes): {avg_reward:.2f}")
            print(f"  Nombre moyen de coups joués (100 derniers épisodes): {avg_moves:.2f}")
            print(f"  Nombre de victoires (100 derniers épisodes): {wins_last_100}")
        return True

def train_sb3_dqn(episodes=100000, save_path="./models/sb3_dqn"):
    # Create the environment
    env = Monitor(TacticiensEnv(opponent_type='random'))

    # Create the DQN model
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        buffer_size=100000, 
        learning_starts=1000, 
        batch_size=32, 
        gamma=0.99, 
        target_update_interval=500, 
        train_freq=10, 
        verbose=1
    )

    # Create a directory to save models
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Add a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_path, name_prefix="dqn_model")
    training_info_callback = TrainingInfoCallback(verbose=1)

    # Train the model
    model.learn(total_timesteps=episodes, callback=[checkpoint_callback, training_info_callback],log_interval=1000)

    # Save the final model
    model.save(os.path.join(save_path, "dqn_final_model"))

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Define training parameters
    episodes = 100000
    save_path = "./models/sb3_dqn"

    # Train the agent
    print(f"Starting training for {episodes} timesteps...")
    train_sb3_dqn(episodes, save_path)
    print("Training completed!")
