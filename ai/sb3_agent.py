import numpy as np
import os
import random
from stable_baselines3 import DQN

class DQNAgent:
    """
    Agent DQN pour le jeu Les Tacticiens de Brême.
    Cette implémentation utilise un modèle pré-entraîné avec Stable Baselines 3.
    """

    def __init__(self, color, game=None):
        """
        Initialise l'agent DQN.

        Args:
            color (str): Couleur de l'agent ("blue" ou "orange")
            game: Instance du jeu (optionnel)
        """
        self.color = color
        self.game = game
        self.type = "DQN"

        # Configuration du modèle
        self.model = None
        self.model_path = "./models/sb3_dqn/dqn_final_model.zip"  # Chemin du modèle SB3
        self.input_shape = (5, 5, 8)
        self.action_size = 100

        # Paramètres pour l'exploration
        self.epsilon = 0.1

        # Charger le modèle s'il existe
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Attention: Modèle non trouvé à {self.model_path}")
            print("Un modèle SB3 est requis pour le fonctionnement de l'agent")

    def load_model(self):
        """Charge un modèle pré-entraîné de Stable Baselines 3"""
        try:
            self.model = DQN.load(self.model_path)
            print(f"Modèle SB3 DQN chargé depuis {self.model_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle SB3: {e}")
            self.model = None

    def encode_state(self):
        """
        Encode l'état du jeu en un tableau numpy pour l'observation.
        Returns:
            numpy.ndarray: État encodé de forme (5, 5, 8)
        """
        state = np.zeros(self.input_shape, dtype=np.int8)

        for pawn in self.game.pawns:
            if 0 <= pawn.x < 5 and 0 <= pawn.y < 5:
                piece_type = pawn.type - 1
                color_offset = 0 if pawn.color == 'blue' else 4
                state[pawn.y, pawn.x, piece_type + color_offset] = 1

        return state

    def playsmart(self):
        """
        Utilise le modèle SB3 DQN pour choisir le meilleur mouvement.
        Returns:
            list: Mouvement choisi [color, piece_type, x, y] ou None
        """
        if self.model is None:
            print("Modèle SB3 DQN non disponible. Utilisation d'un mouvement aléatoire.")
            return self.playrandom(self.game.all_next_moves(self.color))

        # Obtenir tous les mouvements valides
        valid_moves = self.game.all_next_moves(self.color)

        if not valid_moves:
            print("Aucun mouvement valide disponible")
            return None

        # Exploration aléatoire (epsilon-greedy)
        if random.random() < self.epsilon:
            chosen_move = self.playrandom(valid_moves)
            print(f"Mouvement aléatoire (exploration): {chosen_move}")
            return chosen_move

        # Obtenir l'état actuel du jeu
        state = self.encode_state()

        try:
            # Utiliser le modèle SB3 pour prédire l'action
            action, _ = self.model.predict(state, deterministic=True)

            # S'assurer que l'action est dans la plage valide
            action_idx = min(int(action), len(valid_moves) - 1)

            # Retourner le mouvement correspondant
            chosen_move = valid_moves[action_idx]
            print(f"Mouvement choisi par SB3 DQN: {chosen_move}")

            return chosen_move

        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return self.playrandom(valid_moves)

    def playrandom(self, moves):
        """
        Choisit un mouvement aléatoire parmi les mouvements valides.
        Args:
            moves (list): Liste des mouvements valides
        Returns:
            list: Mouvement aléatoire ou None
        """
        if not moves:
            return None
        return moves[np.random.randint(0, len(moves))]