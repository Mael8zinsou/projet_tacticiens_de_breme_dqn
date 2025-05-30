import gym
from gym import spaces
import numpy as np
import os
from datetime import datetime
import csv

# Importations des modules du jeu
from game.game import Game
from game.grid import Grid
from game.pawn import Pawn
from game.env_var import *
from data.data_manager import DataManager
from ai.Minimax import Minimax
from ai.dummyAI import Dummyai
import copy

class TacticiensEnv(gym.Env):
    """Environnement OpenAI Gym pour le jeu 'Les Tacticiens de Brême'"""

    def __init__(self, opponent_type='random', player_color='blue'):
        super().__init__()
        # Initialisation du data manager
        self.data_manager = DataManager(False)

        # Initialisation du jeu avec le data_manager
        ai_types = (1, 1) if opponent_type == 'minimax' else (2, 2)
        self.game = Game(self.data_manager, manual_mode=False, use_ai=True, ai_types=ai_types)

        # Ajouter l'attribut initializing manquant
        self.game.initializing = False

        # Initialisation des attributs spécifiques à l'environnement
        self.player_color = player_color
        self.opponent_type = opponent_type
        self.opponent_color = 'orange' if player_color == 'blue' else 'blue'
        self.last_move_result = [False, False]  # [success, win]
        self.turn_counter = 0

        # Initialisation du fichier de log pour les mouvements
        self._init_move_log()

        # Définition de l'espace d'observation
        # Représentation du plateau 5x5 avec 8 canaux (4 types de pièces x 2 couleurs)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 5, 8), dtype=np.int8)

        # Définition de l'espace d'action
        # Action discrète représentant l'index du mouvement dans la liste des mouvements possibles
        self.action_space = spaces.Discrete(100)  # Nombre maximal estimé de mouvements possibles

        # Stockage des mouvements valides pour l'état actuel
        self.valid_moves = []

        # Initialisation des IA
        if self.opponent_type == 'minimax':
            self.opponent_ai = Minimax(self.opponent_color, self.game)
        else:
            self.opponent_ai = Dummyai(self.opponent_color)

    def _init_move_log(self):
        """Initialise le fichier de log pour les mouvements"""
        path = "./CSV/"
        if not os.path.exists(path):
            os.makedirs(path)

        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.move_log_filename = f"{path}game_moves_{self.time}.csv"

        with open(self.move_log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Color", "Pawn", "X", "Y", "Turn"])  # Write header row

    def reset(self):
        """Réinitialise l'environnement et retourne l'observation initiale"""
        # Réinitialiser le data_manager
        self.data_manager = DataManager(False)

        # Réinitialiser le jeu
        ai_types = (1, 1) if self.opponent_type == 'minimax' else (2, 2)
        self.game = Game(self.data_manager, manual_mode=False, use_ai=True, ai_types=ai_types)

        # Ajouter l'attribut initializing manquant
        self.game.initializing = False

        # Réinitialiser les attributs
        self.last_move_result = [False, False]
        self.turn_counter = 0

        # Réinitialiser le fichier de log
        self._init_move_log()

        # Réinitialiser les IA
        if self.opponent_type == 'minimax':
            self.opponent_ai = Minimax(self.opponent_color, self.game)
        else:
            self.opponent_ai = Dummyai(self.opponent_color)

        # Obtenir les mouvements valides
        self.valid_moves = self.game.all_next_moves(self.player_color)

        return self._get_observation()

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action (int): Index du mouvement à effectuer

        Returns:
            observation (object): Nouvel état du jeu
            reward (float): Récompense obtenue
            done (bool): Indique si l'épisode est terminé
            info (dict): Informations supplémentaires
        """
        # Mettre à jour la liste des mouvements valides si nécessaire
        if not self.valid_moves:
            self.valid_moves = self.game.all_next_moves(self.player_color)

        # Convertir l'action en mouvement
        move = self._action_to_move(action)
        if not move:
            return self._get_observation(), -1, False, {'valid_moves': len(self.valid_moves)}

        # Exécuter le mouvement du joueur
        color, piece_type, x, y = move

        # Trouver le pion à déplacer
        for pawn in self.game.pawns:
            if pawn.type == piece_type and pawn.color == color:
                # Vérifier si le pion doit jouer (en cas de retraite)
                if pawns_must_play[color] == [] or pawn in pawns_must_play[color]:
                    ispawnmoved = pawn.move(x, y, self.game.grid, self.game.pawns, self.game)
                    if pawns_must_play[color] and pawn in pawns_must_play[color]:
                        pawns_must_play[color].remove(pawn)
                else:
                    ispawnmoved = [False, False]
                break

        success, win = ispawnmoved
        self.last_move_result = [success, win]

        # Si le mouvement est valide, enregistrer dans le fichier de log et mettre à jour l'historique
        if success:
            with open(self.move_log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([color, piece_type, x, y, self.turn_counter])

            # Mettre à jour l'historique du pion
            self.data_manager.update_pawn_history(color, piece_type, (x, y), self.turn_counter)

            # Incrémenter le compteur de tours
            self.turn_counter += 1

        # Si le joueur a gagné, terminer l'épisode
        if win:
            # Enregistrer les données de la partie
            final_stack = self.game.grid.getfinalstack(x, y)
            ai = [
                {"type": "DQN", "depth": None, "color": "BLUE" if self.player_color == "blue" else "ORANGE"},
                {"type": self.opponent_ai.type, "depth": getattr(self.opponent_ai, 'base_depth', None),
                 "color": "ORANGE" if self.player_color == "blue" else "BLUE"}
            ]
            self.data_manager.write(ai, self.player_color, self.turn_counter, self.game.num_retreat, final_stack)

            return self._get_observation(), 100, True, {'win': True, 'valid_moves': len(self.valid_moves)}

        # Si le mouvement a échoué, pénaliser le joueur
        if not success:
            return self._get_observation(), -1, False, {'invalid_move': True, 'valid_moves': len(self.valid_moves)}

        # Vérifier si le jeu est en retraite
        with open(self.move_log_filename, mode='r', newline='') as file:
            last_move = list(csv.reader(file))[-1]

        if self.game.isretraite(last_move):
            self.game.num_retreat += 1

        # Faire jouer l'adversaire
        opponent_move = self._play_opponent_move()
        if opponent_move:
            opponent_color, opponent_piece_type, opponent_x, opponent_y = opponent_move

            # Trouver le pion adverse à déplacer
            for pawn in self.game.pawns:
                if pawn.type == opponent_piece_type and pawn.color == opponent_color:
                    # Vérifier si le pion doit jouer (en cas de retraite)
                    if pawns_must_play[opponent_color] == [] or pawn in pawns_must_play[opponent_color]:
                        opponent_ispawnmoved = pawn.move(opponent_x, opponent_y, self.game.grid, self.game.pawns, self.game)
                        if pawns_must_play[opponent_color] and pawn in pawns_must_play[opponent_color]:
                            pawns_must_play[opponent_color].remove(pawn)
                    else:
                        opponent_ispawnmoved = [False, False]
                    break

            opponent_success, opponent_win = opponent_ispawnmoved

            # Si le mouvement de l'adversaire est valide, enregistrer dans le fichier de log et mettre à jour l'historique
            if opponent_success:
                with open(self.move_log_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([opponent_color, opponent_piece_type, opponent_x, opponent_y, self.turn_counter])

                # Mettre à jour l'historique du pion
                self.data_manager.update_pawn_history(opponent_color, opponent_piece_type, (opponent_x, opponent_y), self.turn_counter)

                # Incrémenter le compteur de tours
                self.turn_counter += 1

            # Si l'adversaire a gagné, terminer l'épisode
            if opponent_win:
                # Enregistrer les données de la partie
                final_stack = self.game.grid.getfinalstack(opponent_x, opponent_y)
                ai = [
                    {"type": "DQN", "depth": None, "color": "BLUE" if self.player_color == "blue" else "ORANGE"},
                    {"type": self.opponent_ai.type, "depth": getattr(self.opponent_ai, 'base_depth', None),
                     "color": "ORANGE" if self.player_color == "blue" else "BLUE"}
                ]
                self.data_manager.write(ai, self.opponent_color, self.turn_counter, self.game.num_retreat, final_stack)

                return self._get_observation(), -100, True, {'opponent_win': True, 'valid_moves': len(self.valid_moves)}

        # Vérifier si le jeu est en retraite après le mouvement de l'adversaire
        with open(self.move_log_filename, mode='r', newline='') as file:
            last_move = list(csv.reader(file))[-1]

        if self.game.isretraite(last_move):
            self.game.num_retreat += 1

        # Mettre à jour la liste des mouvements valides
        self.valid_moves = self.game.all_next_moves(self.player_color)

        # Calculer la récompense
        reward = self._calculate_reward(success, win)

        # Vérifier si l'épisode est terminé
        done = win or (opponent_move and opponent_win) or len(self.valid_moves) == 0 or self.game.grid.isbroken

        return self._get_observation(), reward, done, {'valid_moves': len(self.valid_moves)}

    def _get_observation(self):
        """Convertit l'état du jeu en observation pour l'agent"""
        return self._encode_state()

    def _action_to_move(self, action):
        """Convertit un index d'action en mouvement valide"""
        if action < len(self.valid_moves):
            return self.valid_moves[action]
        else:
            # Action invalide, retourner None
            return None

    def _play_opponent_move(self):
        """Fait jouer l'adversaire selon le type spécifié"""
        if self.opponent_type == 'minimax':
            # Utiliser l'IA Minimax
            move = self.opponent_ai.playsmart()
            if move and move[1] != -1:  # Vérifier que le mouvement est valide
                return move
            return None
        else:
            # Utiliser l'IA aléatoire
            opponent_moves = self.game.all_next_moves(self.opponent_color)
            if not opponent_moves:
                return None
            return self.opponent_ai.playrandom(opponent_moves)

    def _calculate_reward(self, success, win):
        """Calcule la récompense en fonction du résultat du mouvement"""
        if win:
            return 100  # Victoire
        elif success:
            # Récompense basée sur l'évaluation du plateau
            eval_score = self.game.evaluateClassic(self.player_color)
            # Normaliser le score pour éviter des valeurs trop grandes
            return min(max(eval_score / 1000, -10), 10)  # Limiter entre -10 et 10
        else:
            return -1  # Mouvement invalide

    def _encode_state(self):
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

    def render(self, mode='human'):
        """Affiche l'état actuel du jeu"""
        if mode == 'human':
            self.game.grid.display()
        return self._get_observation()

    def close(self):
        """Nettoie les ressources"""
        # Supprimer le fichier de log si nécessaire
        if hasattr(self, 'move_log_filename') and os.path.exists(self.move_log_filename):
            os.remove(self.move_log_filename)