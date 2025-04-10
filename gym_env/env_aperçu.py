import gym
from gym import spaces
import numpy as np

class TacticiensEnv(gym.Env):
    def __init__(self):
        super(TacticiensEnv, self).__init__()
        # Définir l'espace d'observation (par exemple, un tableau 5x5x8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 5, 8), dtype=np.int8)
        # Définir l'espace d'action (par exemple, 100 actions possibles)
        self.action_space = spaces.Discrete(100)
        # Initialiser l'état du jeu
        self.state = None

    def reset(self):
        # Réinitialiser l'état du jeu
        self.state = np.zeros((5, 5, 8), dtype=np.int8)
        return self.state

    def step(self, action):
        # Exécuter une action et mettre à jour l'état
        reward = 0
        done = False
        info = {}
        # Exemple simplifié : mettre à jour l'état et calculer la récompense
        self.state = np.random.randint(0, 2, (5, 5, 8))
        reward = 1 if action == 0 else -1
        done = True if action == 99 else False
        return self.state, reward, done, info

    def render(self, mode='human'):
        # Afficher l'état du jeu
        print(self.state)