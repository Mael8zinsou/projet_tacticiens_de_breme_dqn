from game.env_var import *

class Mouvement:
    """
    Classe gérant les règles de déplacement des pions.
    Vérifie si un mouvement est légal selon le type de mouvement du pion.
    """

    @staticmethod
    def legit_mouv(pawn, self, x, y, grid):
        """
        Vérifie si un mouvement est légal pour un pion donné.

        Args:
            pawn: Le pion à déplacer
            self: Référence au pion (même que pawn, redondant)
            x: Coordonnée x de destination
            y: Coordonnée y de destination
            grid: La grille de jeu

        Returns:
            bool: True si le mouvement est légal, False sinon
        """
        # Vérifier que la destination est dans la grille et que le pion peut être empilé
        grid_size = len(grid.grid)
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False

        if not (grid.grid[y][x][-1] > pawn.type or grid.grid[y][x][-1] == 0):
            return False

        # Calculer la distance entre la position actuelle et la destination
        dx, dy = distance(pawn, x, y)

        # Vérifier selon le type de mouvement
        if pawn.mouvement == "L":
            return Mouvement._check_L_movement(dx, dy)
        elif pawn.mouvement == "+":
            return Mouvement._check_plus_movement(pawn, x, y, dx, dy, grid)
        elif pawn.mouvement == "X":
            return Mouvement._check_X_movement(pawn, x, y, dx, dy, grid)
        elif pawn.mouvement == "*":
            return Mouvement._check_star_movement(pawn, x, y, dx, dy, grid)

        return False

    @staticmethod
    def _check_L_movement(dx, dy):
        """
        Vérifie si un mouvement en L est légal.
        """
        return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

    @staticmethod
    def _check_plus_movement(pawn, x, y, dx, dy, grid):
        """
        Vérifie si un mouvement en + est légal.
        """
        # Le mouvement doit être horizontal ou vertical
        if not ((dx > 0 and dy == 0) or (dx == 0 and dy > 0)):
            return False

        # Vérifier qu'il n'y a pas d'obstacle sur le chemin
        if dx == 0:  # Mouvement vertical
            start_y = min(pawn.y, y) + 1
            end_y = max(pawn.y, y)
            for i in range(start_y, end_y):
                if grid.grid[i][x][-1] != 0:
                    return False
        else:  # Mouvement horizontal
            start_x = min(pawn.x, x) + 1
            end_x = max(pawn.x, x)
            for i in range(start_x, end_x):
                if grid.grid[y][i][-1] != 0:
                    return False

        return True

    @staticmethod
    def _check_X_movement(pawn, x, y, dx, dy, grid):
        """
        Vérifie si un mouvement en X est légal.
        """
        # Le mouvement doit être diagonal
        if dx != dy:
            return False

        # Déterminer la direction du mouvement
        x_step = 1 if x > pawn.x else -1
        y_step = 1 if y > pawn.y else -1

        # Vérifier qu'il n'y a pas d'obstacle sur le chemin
        current_x = pawn.x + x_step
        current_y = pawn.y + y_step

        while current_x != x and current_y != y:
            if grid.grid[current_y][current_x][-1] != 0 and pawn.type >= grid.grid[current_y][current_x][-1]:
                return False
            current_x += x_step
            current_y += y_step

        return True

    @staticmethod
    def _check_star_movement(pawn, x, y, dx, dy, grid):
        """
        Vérifie si un mouvement en * est légal.
        """
        # Le mouvement doit être horizontal, vertical ou diagonal
        if not ((dx > 0 and dy == 0) or (dx == 0 and dy > 0) or (dx == dy)):
            return False

        # Vérifier selon le type de mouvement
        if dx == 0 or dy == 0:  # Mouvement horizontal ou vertical
            return Mouvement._check_plus_movement(pawn, x, y, dx, dy, grid)
        else:  # Mouvement diagonal
            # Déterminer la direction du mouvement
            x_step = 1 if x > pawn.x else -1
            y_step = 1 if y > pawn.y else -1

            # Vérifier qu'il n'y a pas de pion de type 1 sur le chemin
            current_x = pawn.x + x_step
            current_y = pawn.y + y_step

            while current_x != x and current_y != y:
                if grid.grid[current_y][current_x][-1] == 1:
                    return False
                current_x += x_step
                current_y += y_step

            return True