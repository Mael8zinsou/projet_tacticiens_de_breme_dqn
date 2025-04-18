import numpy as np
import os
import random
from stable_baselines3 import DQN
from typing import Optional, List

class DQN_sb3_Agent:
    """
    Agent DQN optimisé pour Les Tacticiens de Brême.
    Compatible à la fois avec le moteur de jeu original et l'environnement Gym.
    """

    def __init__(self, color: str, game=None, model_path: str = "./models/dqn_phase2.zip"):
        """
        Args:
            color: "blue" ou "orange"
            game: Instance du jeu (optionnelle)
            model_path: Chemin vers le modèle .zip entraîné
        """
        self.color = color
        self.game = game
        self.type = "D"  # D pour DQN
        self.model_path = model_path
        self.epsilon = 0.1  # Taux d'exploration
        
        # Chargement immédiat du modèle
        self.model = self._load_model()

    def _load_model(self) -> Optional[DQN]:
        """Charge le modèle Stable Baselines 3 avec vérification d'erreurs."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle DQN introuvable à {self.model_path}")
        
        try:
            model = DQN.load(self.model_path)
            print(f"Modèle DQN chargé depuis {self.model_path}")
            return model
        except Exception as e:
            print(f"Erreur de chargement : {e}")
            return None

    def encode_state(self) -> np.ndarray:
        """
        Encode l'état du jeu dans le format attendu par le modèle.
        Version compatible avec les deux représentations (moteur de jeu et Gym).
        """
        if self.game is None:
            raise ValueError("Instance de jeu non fournie")
        
        # Initialisation avec la même structure que l'environnement Gym
        state = np.zeros((5, 5, 5), dtype=np.int8)
        
        for pawn in self.game.pawns:
            if 0 <= pawn.x < 5 and 0 <= pawn.y < 5:
                # Channel 0: propriétaire de la base
                if state[pawn.y, pawn.x, 0] == 0:
                    state[pawn.y, pawn.x, 0] = 1 if pawn.color == "blue" else 2
                
                # Channels 1-4: présence des pions
                state[pawn.y, pawn.x, pawn.type] = 1
                
        return state

    def playsmart(self) -> Optional[List]:
        """
        Sélectionne un mouvement optimal avec stratégie epsilon-greedy.
        Retourne [color, piece_type, x, y] ou None si aucun mouvement valide.
        """
        valid_moves = self.game.all_next_moves(self.color)
        if not valid_moves:
            return None

        # Exploration aléatoire
        if random.random() < self.epsilon:
            return self._random_move(valid_moves)

        # Prédiction du modèle
        try:
            state = self.encode_state()
            action, _ = self.model.predict(state, deterministic=True)
            
            # Conversion de l'action en mouvement valide
            return self._action_to_move(action, valid_moves)
        except Exception as e:
            print(f"Erreur de prédiction : {e}")
            return self._random_move(valid_moves)

    def _random_move(self, moves: List) -> List:
        """Sélection aléatoire parmi les mouvements valides."""
        return random.choice(moves) if moves else None

    def _action_to_move(self, action: int, valid_moves: List) -> Optional[List]:
        """
        Convertit l'action du modèle (0-99) en mouvement valide.
        Gère les actions potentiellement invalides.
        """
        if not valid_moves:
            return None
            
        # Méthode 1: Sélection directe si l'action est valide
        if 0 <= action < len(valid_moves):
            return valid_moves[action]
            
        # Méthode 2: Fallback intelligent (recherche du mouvement le plus proche)
        pawn_type = (action // 25) + 1
        x = (action % 25) % 5
        y = (action % 25) // 5
        
        # Trouve le mouvement valide le plus proche
        for move in valid_moves:
            if move[1] == pawn_type and move[2] == x and move[3] == y:
                return move
                
        return self._random_move(valid_moves)

    # Alias pour compatibilité
    playrandom = _random_move