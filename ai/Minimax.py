import copy
import random
import time

class Minimax:
    """
    Implémentation de l'algorithme Minimax avec élagage alpha-bêta pour l'IA du jeu.
    """

    def __init__(self, color, game):
        """
        Initialise l'IA Minimax.

        Args:
            color (str): Couleur de l'IA ("blue" ou "orange")
            game: Instance du jeu
        """
        self.type = "M"  # Type d'IA (M pour Minimax)
        self.color = color
        self.game = game
        self.base_depth = self.set_base_depth_by_color(color)
        self.moves_scores = {}  # Dictionnaire pour stocker les scores des mouvements
        self.ispawnmoved = [False, False]
        self.opponent_color = "orange" if color == "blue" else "blue"

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
            game: État du jeu à évaluer
            depth: Profondeur actuelle dans l'arbre de recherche
            max_depth: Profondeur maximale de recherche
            is_maximizing: True si c'est le tour du joueur maximisant
            alpha: Valeur alpha pour l'élagage
            beta: Valeur bêta pour l'élagage
            move: Dernier mouvement effectué

        Returns:
            float: Score de l'état du jeu
        """
        # Condition d'arrêt : profondeur maximale atteinte ou victoire
        if depth == max_depth or self.ispawnmoved[1]:
            return game.evaluateClassic(self.color)

        if is_maximizing:
            return self._maximize(game, depth, max_depth, alpha, beta)
        else:
            return self._minimize(game, depth, max_depth, alpha, beta)

    def _maximize(self, game, depth, max_depth, alpha, beta):
        """
        Phase de maximisation de l'algorithme Minimax.
        """
        max_eval = float('-inf')

        # Parcourir tous les mouvements possibles pour le joueur actuel
        for next_move in game.all_next_moves(self.color):
            # Simuler le mouvement
            simulated_game = copy.deepcopy(game)
            color, piece_type, x, y = next_move
            self.ispawnmoved = simulated_game.simulate_move(color, piece_type, x, y)

            # Évaluer récursivement
            eval_score = self.minimax(simulated_game, depth + 1, max_depth, False, alpha, beta)

            # Enregistrer le score à la racine de l'arbre
            if depth == 0:
                self.moves_scores[tuple(next_move)] = eval_score

            # Mettre à jour le score maximal et alpha
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)

            # Élagage alpha-bêta
            if beta <= alpha:
                break

        return max_eval

    def _minimize(self, game, depth, max_depth, alpha, beta):
        """
        Phase de minimisation de l'algorithme Minimax.
        """
        min_eval = float('inf')

        # Parcourir tous les mouvements possibles pour l'adversaire
        for next_move in game.all_next_moves(self.opponent_color):
            # Simuler le mouvement
            simulated_game = copy.deepcopy(game)
            color, piece_type, x, y = next_move
            simulated_game.simulate_move(color, piece_type, x, y)

            # Évaluer récursivement
            eval_score = self.minimax(simulated_game, depth + 1, max_depth, True, alpha, beta)

            # Mettre à jour le score minimal et bêta
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)

            # Élagage alpha-bêta
            if beta <= alpha:
                break

        return min_eval

    def choose_best_move(self):
        """
        Choisit le meilleur mouvement en fonction des scores calculés.

        Returns:
            list: Le meilleur mouvement à jouer
        """
        if not self.moves_scores:
            return self._handle_no_moves()

        # Trouver le score maximal
        max_score = max(self.moves_scores.values())

        # Collecter tous les mouvements ayant ce score
        best_moves = [move for move, score in self.moves_scores.items() if score == max_score]

        # Choisir aléatoirement parmi les meilleurs mouvements
        best_move = random.choice(best_moves)
        print(f"Best move chosen randomly from {len(best_moves)} top scoring moves: {best_move}")

        return list(best_move)

    def _handle_no_moves(self):
        """
        Gère le cas où aucun mouvement n'a été évalué.
        """
        print("No valid moves found, returning default move.")
        all_moves = self.game.all_next_moves(self.color)

        if all_moves:
            return random.choice(all_moves)
        else:
            return [self.color, -1, -1, -1]  # Mouvement spécial indiquant l'impossibilité de jouer

    def playsmart(self):
        """
        Calcule et retourne le meilleur mouvement à jouer.

        Returns:
            list: Le meilleur mouvement à jouer
        """
        start_time = time.time()
        self.moves_scores = {}

        # Lancer l'algorithme Minimax
        self.minimax(self.game, 0, self.base_depth, True)

        # Choisir le meilleur mouvement
        best_move = self.choose_best_move()

        end_time = time.time()
        print(f"Minimax calculation took {end_time - start_time:.2f} seconds")

        return best_move