
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import os

class DQNAgent:
    def __init__(self, color, game=None):
        self.color = color
        self.game = game
        self.type = "DQN"  # Type d'IA pour la compatibilité avec le code existant

        # Charger le modèle s'il existe
        self.model = None
        self.model_path = "./models/dqn_agent_final.h5"

        if os.path.exists(self.model_path):
            self.load_model()

    def load_model(self):
        # Créer le modèle
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(5, 5, 8)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(100, activation='linear'))  # 100 actions possibles max

        # Charger les poids
        self.model.load_weights(self.model_path)

    def encode_state(self):
        """Encode l'état du jeu en un tableau numpy pour l'observation"""
        # Initialiser un tableau 5x5x8 rempli de zéros
        # (5x5 pour le plateau, 8 pour les 4 types de pièces x 2 couleurs)
        state = np.zeros((5, 5, 8), dtype=np.int8)

        # Remplir le tableau avec les positions des pièces
        for pawn in self.game.pawns:
            if pawn.x >= 0 and pawn.y >= 0:  # Vérifier que la pièce est sur le plateau
                piece_type = pawn.type - 1  # Types de 1 à 4 -> index 0 à 3
                color_offset = 0 if pawn.color == 'blue' else 4  # Bleu: 0-3, Orange: 4-7
                state[pawn.y, pawn.x, piece_type + color_offset] = 1

        return state

    def playsmart(self):
        """Utilise le modèle DQN pour choisir le meilleur mouvement"""
        if self.model is None:
            print("Modèle DQN non chargé. Utilisation d'un mouvement aléatoire.")
            return self.playrandom(self.game.all_next_moves(self.color))

        # Obtenir l'état actuel du jeu
        state = self.encode_state()

        # Obtenir tous les mouvements valides
        valid_moves = self.game.all_next_moves(self.color)

        if not valid_moves:
            return None

        # Prédire les valeurs Q pour toutes les actions
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        # Filtrer les actions valides
        valid_q_values = [(i, q_values[i]) for i in range(len(q_values)) if i < len(valid_moves)]

        # Choisir l'action avec la plus grande valeur Q parmi les actions valides
        best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]

        # Retourner le mouvement correspondant
        return valid_moves[best_action_idx]

    def playrandom(self, moves):
        """Choisit un mouvement aléatoire parmi les mouvements valides"""
        if not moves:
            return None
        return moves[np.random.randint(0, len(moves))]