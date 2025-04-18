import random
from game.env_var import *

class Dummyai:
    """
    Une IA simple qui joue des coups aléatoires.
    """

    def __init__(self, color):
        """
        Initialise l'IA avec une couleur.

        Args:
            color (str): La couleur de l'IA ("blue" ou "orange")
        """
        self.color = color
        self.type = "R"  # R pour Random

    def playrandom(self, nextmouvs):
        """
        Choisit un mouvement aléatoire parmi les mouvements possibles.
        Si des pions doivent jouer en priorité (règle de retraite),
        le mouvement est choisi parmi ceux-ci.

        Args:
            nextmouvs (list): Liste des mouvements possibles

        Returns:
            list: Un mouvement aléatoire ou None si aucun mouvement n'est disponible
        """
        # Vérifier si des mouvements sont disponibles
        if not nextmouvs:
            print("Aucun mouvement disponible")
            return None

        # Si des pions doivent jouer en priorité
        if pawns_must_play[self.color]:
            # Filtrer les mouvements pour ne garder que ceux des pions qui doivent jouer
            priority_moves = []
            pawn_types_must_play = [pawn.type for pawn in pawns_must_play[self.color]]

            for move in nextmouvs:
                if move[1] in pawn_types_must_play:
                    priority_moves.append(move)

            # Si des mouvements prioritaires sont disponibles, en choisir un aléatoirement
            if priority_moves:
                chosen_move = random.choice(priority_moves)
                print(f"Mouvement prioritaire choisi: {chosen_move}")
                return chosen_move

        # Sinon, choisir un mouvement aléatoire parmi tous les mouvements possibles
        chosen_move = random.choice(nextmouvs)
        print(f"Mouvement aléatoire choisi: {chosen_move}")
        return chosen_move