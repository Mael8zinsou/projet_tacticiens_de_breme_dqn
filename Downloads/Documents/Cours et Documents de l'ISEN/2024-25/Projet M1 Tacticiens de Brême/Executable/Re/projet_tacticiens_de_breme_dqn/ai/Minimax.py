import copy
import random

class Minimax:
    """
    Implémentation de l'algorithme Minimax avec élagage alpha-bêta pour le jeu Les Tacticiens de Brême.
    """

    def __init__(self, color, game):
        """
        Initialise l'IA Minimax.

        Args:
            color (str): Couleur de l'IA ("blue" ou "orange")
            game: Instance du jeu
        """
        self.type = "M"
        self.color = color
        self.game = game
        self.base_depth = self.set_base_depth_by_color(color)
        self.moves_scores = {}
        self.ispawnmoved = [False, False]

    def set_base_depth_by_color(self, color):
        """
        Définit la profondeur de recherche en fonction de la couleur.

        Args:
            color (str): Couleur de l'IA

        Returns:
            int: Profondeur de recherche
        """
        return 4  # Même profondeur pour les deux couleurs

    def minimax(self, game, depth, max_depth, is_maximizing, alpha=float('-inf'), beta=float('inf'), move=None):
        """
        Implémentation récursive de l'algorithme Minimax avec élagage alpha-bêta.

        Args:
            game: Instance du jeu
            depth (int): Profondeur actuelle
            max_depth (int): Profondeur maximale
            is_maximizing (bool): True si c'est le tour du joueur maximisant
            alpha (float): Valeur alpha pour l'élagage
            beta (float): Valeur bêta pour l'élagage
            move: Dernier mouvement effectué

        Returns:
            float: Score d'évaluation
        """
        # Condition d'arrêt : profondeur maximale atteinte ou victoire
        if depth == max_depth or self.ispawnmoved[1]:
            return game.evaluateClassic(self.color)

        if is_maximizing:
            # Tour du joueur maximisant (l'IA)
            max_eval = float('-inf')
            for next_move in game.all_next_moves(self.color):
                # Simulation du mouvement
                simulated_game = copy.deepcopy(game)
                color, piece_type, x, y = next_move
                self.ispawnmoved = simulated_game.simulate_move(color, piece_type, x, y)

                # Évaluation récursive
                eval = self.minimax(simulated_game, depth + 1, max_depth, False, alpha, beta, next_move)

                # Enregistrement des scores à la racine
                if depth == 0:
                    self.moves_scores[tuple(next_move)] = eval

                # Mise à jour de l'évaluation maximale et d'alpha
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                # Élagage alpha-bêta
                if beta <= alpha:
                    break

            return max_eval
        else:
            # Tour de l'adversaire (joueur minimisant)
            min_eval = float('inf')
            opponent_color = "orange" if self.color == "blue" else "blue"

            for next_move in game.all_next_moves(opponent_color):
                # Simulation du mouvement
                simulated_game = copy.deepcopy(game)
                color, piece_type, x, y = next_move
                simulated_game.simulate_move(color, piece_type, x, y)

                # Évaluation récursive
                eval = self.minimax(simulated_game, depth + 1, max_depth, True, alpha, beta, next_move)

                # Mise à jour de l'évaluation minimale et de bêta
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # Élagage alpha-bêta
                if beta <= alpha:
                    break

            return min_eval

    def choose_best_move(self):
        """
        Choisit le meilleur mouvement en fonction des scores calculés.

        Returns:
            list: Meilleur mouvement [color, piece_type, x, y]
        """
        if self.moves_scores:
            # Trouver le score maximum
            max_score = max(self.moves_scores.values())

            # Collecter tous les mouvements ayant ce score
            best_moves = [move for move, score in self.moves_scores.items() if score == max_score]

            # Choisir aléatoirement parmi les meilleurs mouvements
            best_move = random.choice(best_moves)
            print(f"Meilleur mouvement choisi (score: {max_score}): {best_move}")
            return list(best_move)
        else:
            print("Aucun mouvement valide trouvé, retour au mouvement par défaut.")
            all_moves = self.game.all_next_moves(self.color)

            if all_moves:
                return random.choice(all_moves)  # Mouvement aléatoire
            else:
                return [self.color, -1, -1, -1]  # Aucun mouvement possible

    def playsmart(self):
        """
        Exécute l'algorithme Minimax et retourne le meilleur mouvement.

        Returns:
            list: Meilleur mouvement [color, piece_type, x, y]
        """
        self.moves_scores = {}
        self.minimax(self.game, 0, self.base_depth, True)
        return self.choose_best_move()