import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import os
import random

class DQNAgent:
    """
    Agent DQN pour le jeu Les Tacticiens de Brême.
    Cette implémentation se concentre sur l'utilisation d'un modèle pré-entraîné.
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
        self.type = "DQN"  # Type d'IA pour la compatibilité avec le code existant

        # Configuration du modèle
        self.model = None
        self.model_path = "./models/dqn_agent_final.h5"
        self.input_shape = (5, 5, 8)  # Forme de l'entrée du modèle
        self.action_size = 100  # Nombre maximal d'actions possibles

        # Paramètres pour l'exploration (utile si on veut continuer l'entraînement)
        self.epsilon = 0.1  # Taux d'exploration (10% de mouvements aléatoires)

        # Charger le modèle s'il existe
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Attention: Modèle non trouvé à {self.model_path}")
            self.create_model()  # Créer un modèle vide si aucun n'est trouvé

    def create_model(self):
        """Crée un nouveau modèle DQN"""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        print("Nouveau modèle DQN créé (non entraîné)")

    def load_model(self):
        """Charge un modèle pré-entraîné"""
        try:
            # Créer l'architecture du modèle
            self.create_model()

            # Charger les poids
            self.model.load_weights(self.model_path)
            print(f"Modèle DQN chargé depuis {self.model_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.create_model()

    def save_model(self, path=None):
        """Sauvegarde le modèle"""
        if self.model is not None:
            save_path = path or self.model_path
            self.model.save_weights(save_path)
            print(f"Modèle sauvegardé à {save_path}")

    def encode_state(self):
        """
        Encode l'état du jeu en un tableau numpy pour l'observation.

        Returns:
            numpy.ndarray: État encodé de forme (5, 5, 8)
        """
        # Initialiser un tableau 5x5x8 rempli de zéros
        # (5x5 pour le plateau, 8 pour les 4 types de pièces x 2 couleurs)
        state = np.zeros(self.input_shape, dtype=np.int8)

        # Remplir le tableau avec les positions des pièces
        for pawn in self.game.pawns:
            if 0 <= pawn.x < 5 and 0 <= pawn.y < 5:  # Vérifier que la pièce est sur le plateau
                piece_type = pawn.type - 1  # Types de 1 à 4 -> index 0 à 3
                color_offset = 0 if pawn.color == 'blue' else 4  # Bleu: 0-3, Orange: 4-7
                state[pawn.y, pawn.x, piece_type + color_offset] = 1

        return state

    def playsmart(self):
        """
        Utilise le modèle DQN pour choisir le meilleur mouvement.

        Returns:
            list: Mouvement choisi [color, piece_type, x, y] ou None
        """
        if self.model is None:
            print("Modèle DQN non disponible. Utilisation d'un mouvement aléatoire.")
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

        # Prédire les valeurs Q pour toutes les actions
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        # Filtrer les actions valides
        valid_actions = min(len(valid_moves), self.action_size)
        valid_q_values = [(i, q_values[i]) for i in range(valid_actions)]

        # Choisir l'action avec la plus grande valeur Q parmi les actions valides
        best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]

        # Retourner le mouvement correspondant
        chosen_move = valid_moves[best_action_idx]
        print(f"Mouvement choisi par DQN: {chosen_move}, Q-value: {q_values[best_action_idx]:.4f}")

        return chosen_move

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