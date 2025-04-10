import gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
from datetime import datetime
import csv
import copy

# Importations des modules du jeu
from game.game import Game
from game.grid import Grid
from game.pawn import Pawn
from game.env_var import *
from data.data_manager import DataManager
from ai.Minimax import Minimax
from ai.dummyAI import Dummyai

class TacticiensEnv(Env):
    """
    Environnement OpenAI Gym pour le jeu 'Les Tacticiens de Brême'.
    Permet d'entraîner des agents d'apprentissage par renforcement sur ce jeu de plateau.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, opponent_type='random', player_color='blue'):
        """
        Initialise l'environnement du jeu.

        Args:
            opponent_type (str): Type d'adversaire ('random' ou 'minimax')
            player_color (str): Couleur du joueur ('blue' ou 'orange')
        """
        super().__init__()

        # Configuration de l'environnement
        self.player_color = player_color
        self.opponent_color = 'orange' if player_color == 'blue' else 'blue'
        self.opponent_type = opponent_type

        # Initialisation du jeu
        self.data_manager = DataManager(False)
        ai_types = (1, 1) if opponent_type == 'minimax' else (2, 2)
        self.game = Game(self.data_manager, manual_mode=False, use_ai=True, ai_types=ai_types)
        self.game.initializing = False  # Attribut nécessaire

        # État du jeu
        self.turn_counter = 0
        self.last_move_result = [False, False]  # [success, win]

        # Attributs pour le suivi des mouvements et récompenses améliorées
        self.last_move_coords = (-1, -1)
        self.last_move_piece_type = None
        self.last_move_color = None
        self.previous_opponent_pawns_count = 0

        # Espaces d'observation et d'action
        self.observation_space = Box(low=0, high=1, shape=(5, 5, 8), dtype=np.int8)
        self.action_space = Discrete(100)  # Nombre maximal estimé de mouvements possibles

        # Stockage des mouvements valides
        self.valid_moves = []

        # Initialisation de l'adversaire
        self._init_opponent()

        # Initialisation du fichier de log
        self._init_move_log()

        # Compter les pions adverses au début
        self.previous_opponent_pawns_count = sum(1 for pawn in self.game.pawns if pawn.color == self.opponent_color)

    def _init_opponent(self):
        """Initialise l'IA adversaire selon le type spécifié"""
        if self.opponent_type == 'minimax':
            self.opponent_ai = Minimax(self.opponent_color, self.game)
        else:
            self.opponent_ai = Dummyai(self.opponent_color)

    def _init_move_log(self):
        """Initialise le fichier de log pour les mouvements"""
        path = "./CSV/"
        os.makedirs(path, exist_ok=True)

        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.move_log_filename = f"{path}game_moves_{self.time}.csv"

        with open(self.move_log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Color", "Pawn", "X", "Y", "Turn"])

    def reset(self, *, seed=None, options=None):
        """
        Réinitialise l'environnement et retourne l'observation initiale.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        # Réinitialiser le jeu
        self.data_manager = DataManager(False)
        ai_types = (1, 1) if self.opponent_type == 'minimax' else (2, 2)
        self.game = Game(self.data_manager, manual_mode=False, use_ai=True, ai_types=ai_types)
        self.game.initializing = False

        # Réinitialiser l'état
        self.turn_counter = 0
        self.last_move_result = [False, False]
        self.last_move_coords = (-1, -1)
        self.last_move_piece_type = None
        self.last_move_color = None

        # Réinitialiser le fichier de log
        self._init_move_log()

        # Réinitialiser l'adversaire
        self._init_opponent()

        # Obtenir les mouvements valides
        self.valid_moves = self.game.all_next_moves(self.player_color)

        # Compter les pions adverses au début
        self.previous_opponent_pawns_count = sum(1 for pawn in self.game.pawns if pawn.color == self.opponent_color)

        return self._encode_state(), {}

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action (int): Index du mouvement à effectuer

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Mettre à jour la liste des mouvements valides si nécessaire
        if not self.valid_moves:
            self.valid_moves = self.game.all_next_moves(self.player_color)

        # Vérifier si l'action est valide
        if action >= len(self.valid_moves) or not self.valid_moves:
            return self._encode_state(), -5, True, False, {'valid_moves': 0}

        # Mettre à jour le compteur de pions adverses avant le mouvement
        self.previous_opponent_pawns_count = sum(1 for pawn in self.game.pawns if pawn.color == self.opponent_color)

        # Exécuter le mouvement du joueur
        player_success, player_win = self._execute_player_move(action)

        # Si le joueur a gagné, terminer l'épisode
        if player_win:
            return self._encode_state(), 10, True, False, {'win': True, 'valid_moves': len(self.valid_moves)}

        # Si le mouvement a échoué, pénaliser le joueur
        if not player_success:
            return self._encode_state(), -5, True, False, {'invalid_move': True, 'valid_moves': len(self.valid_moves)}

        # Vérifier les retraites
        self._check_retreat()

        # Faire jouer l'adversaire
        opponent_success, opponent_win = self._execute_opponent_move()

        # Si l'adversaire a gagné, terminer l'épisode
        if opponent_win:
            return self._encode_state(), -10, True, False, {'opponent_win': True, 'valid_moves': len(self.valid_moves)}

        # Vérifier les retraites après le mouvement de l'adversaire
        self._check_retreat()

        # Mettre à jour la liste des mouvements valides
        self.valid_moves = self.game.all_next_moves(self.player_color)

        # Calculer la récompense
        reward = self._calculate_reward(player_success, player_win)

        # Vérifier si l'épisode est terminé
        done = (player_win or opponent_win or
                len(self.valid_moves) == 0 or
                self.game.grid.isbroken)

        return self._encode_state(), reward, done, False, {'valid_moves': len(self.valid_moves)}

    def _execute_player_move(self, action):
        """
        Exécute le mouvement du joueur.

        Args:
            action (int): Index du mouvement à effectuer

        Returns:
            tuple: (success, win)
        """
        move = self.valid_moves[action]
        color, piece_type, x, y = move

        # Stocker les coordonnées du mouvement avant de l'exécuter
        self.last_move_coords = (x, y)
        self.last_move_piece_type = piece_type
        self.last_move_color = color

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
            self._log_move(color, piece_type, x, y)
            self.data_manager.update_pawn_history(color, piece_type, (x, y), self.turn_counter)
            self.turn_counter += 1

        # Si le joueur a gagné, enregistrer les données de la partie
        if win:
            self._record_game_result(color, x, y)

        return success, win

    def _execute_opponent_move(self):
        """
        Fait jouer l'adversaire.

        Returns:
            tuple: (success, win)
        """
        opponent_move = self._get_opponent_move()
        if not opponent_move:
            return False, False

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
            self._log_move(opponent_color, opponent_piece_type, opponent_x, opponent_y)
            self.data_manager.update_pawn_history(opponent_color, opponent_piece_type, (opponent_x, opponent_y), self.turn_counter)
            self.turn_counter += 1

        # Si l'adversaire a gagné, enregistrer les données de la partie
        if opponent_win:
            self._record_game_result(opponent_color, opponent_x, opponent_y)

        return opponent_success, opponent_win

    def _get_opponent_move(self):
        """
        Obtient le mouvement de l'adversaire selon le type spécifié.

        Returns:
            list: Mouvement de l'adversaire [color, piece_type, x, y] ou None
        """
        if self.opponent_type == 'minimax':
            move = self.opponent_ai.playsmart()
            if move and move[1] != -1:  # Vérifier que le mouvement est valide
                return move
            return None
        else:
            opponent_moves = self.game.all_next_moves(self.opponent_color)
            if not opponent_moves:
                return None
            return self.opponent_ai.playrandom(opponent_moves)

    def _log_move(self, color, piece_type, x, y):
        """
        Enregistre un mouvement dans le fichier de log.

        Args:
            color (str): Couleur du joueur
            piece_type (int): Type de pièce
            x (int): Coordonnée x
            y (int): Coordonnée y
        """
        with open(self.move_log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([color, piece_type, x, y, self.turn_counter])

    def _check_retreat(self):
        """Vérifie si le jeu est en retraite après un mouvement"""
        with open(self.move_log_filename, mode='r', newline='') as file:
            moves = list(csv.reader(file))
            if len(moves) > 1:
                last_move = moves[-1]
                if self.game.isretraite(last_move):
                    self.game.num_retreat += 1

    def _record_game_result(self, winner_color, x, y):
        """
        Enregistre les résultats de la partie.

        Args:
            winner_color (str): Couleur du gagnant
            x (int): Coordonnée x de la pile gagnante
            y (int): Coordonnée y de la pile gagnante
        """
        final_stack = self.game.grid.getfinalstack(x, y)
        ai = [
            {"type": "DQN", "depth": None, "color": "BLUE" if self.player_color == "blue" else "ORANGE"},
            {"type": self.opponent_ai.type, "depth": getattr(self.opponent_ai, 'base_depth', None),
             "color": "ORANGE" if self.player_color == "blue" else "BLUE"}
        ]
        self.data_manager.write(ai, winner_color, self.turn_counter, self.game.num_retreat, final_stack)

    def _calculate_reward(self, success, win):
        """
        Calcule une récompense détaillée basée sur plusieurs facteurs stratégiques.

        Args:
            success (bool): Indique si le mouvement a réussi
            win (bool): Indique si le joueur a gagné

        Returns:
            float: Récompense
        """
        if win:
            return 10  # Victoire
        elif success:
            # Récompense de base basée sur l'évaluation du plateau
            eval_score = self.game.evaluateClassic(self.player_color)
            base_reward = min(max(eval_score / 800, -5), 5)

            # Extraire les informations du dernier mouvement
            x, y = self.last_move_coords
            piece_type = self.last_move_piece_type

            # 1. Bonus pour les mouvements vers le centre
            center_bonus = 0.2 if (1 <= x <= 3 and 1 <= y <= 3) else 0

            # 2. Bonus pour les mouvements vers les objectifs (coins pour type 1)
            objective_bonus = 0
            if piece_type == 1:  # Si c'est un pion de type 1
                corners = [(0,0), (0,4), (4,0), (4,4)]
                if (x, y) in corners:
                    objective_bonus = 0.5
                else:
                    # Récompense la progression vers les coins
                    min_distance = min(abs(x-cx) + abs(y-cy) for cx, cy in corners)
                    objective_bonus = 0.1 * (5 - min_distance)

            # 3. Bonus pour les captures
            current_opponent_pawns_count = sum(1 for pawn in self.game.pawns if pawn.color == self.opponent_color)
            capture_bonus = 0.3 * (self.previous_opponent_pawns_count - current_opponent_pawns_count)

            # 4. Bonus pour la formation de piles
            stack_size = len(self.game.grid.grid[y][x])
            stack_bonus = 0.1 * (stack_size - 1)

            # Calculer la récompense totale
            total_reward = base_reward + center_bonus + objective_bonus + capture_bonus + stack_bonus

            return total_reward
        else:
            return -5  # Mouvement invalide

    def _encode_state(self):
        """
        Encode l'état du jeu en un tableau numpy pour l'observation.

        Returns:
            numpy.ndarray: État encodé de forme (5, 5, 8)
        """
        # Initialiser un tableau 5x5x8 rempli de zéros
        # (5x5 pour le plateau, 8 pour les 4 types de pièces x 2 couleurs)
        state = np.zeros((5, 5, 8), dtype=np.int8)

        # Remplir le tableau avec les positions des pièces
        for pawn in self.game.pawns:
            if 0 <= pawn.x < 5 and 0 <= pawn.y < 5:  # Vérifier que la pièce est sur le plateau
                piece_type = pawn.type - 1  # Types de 1 à 4 -> index 0 à 3
                color_offset = 0 if pawn.color == 'blue' else 4  # Bleu: 0-3, Orange: 4-7
                state[pawn.y, pawn.x, piece_type + color_offset] = 1

        return state

    def render(self, mode='human'):
        """
        Affiche l'état actuel du jeu.

        Args:
            mode (str): Mode de rendu ('human' ou 'rgb_array')

        Returns:
            numpy.ndarray: État encodé
        """
        if mode == 'human':
            self.game.grid.display()

        return self._encode_state()

    def close(self):
        """Nettoie les ressources"""
        # Supprimer le fichier de log si nécessaire
        if hasattr(self, 'move_log_filename') and os.path.exists(self.move_log_filename):
            os.remove(self.move_log_filename)