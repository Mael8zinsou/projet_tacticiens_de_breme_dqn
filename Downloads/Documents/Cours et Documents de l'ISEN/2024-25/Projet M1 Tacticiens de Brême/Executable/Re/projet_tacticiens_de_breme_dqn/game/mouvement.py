from game.env_var import *

class Mouvement:
    @staticmethod
    def legit_mouv(pawn, x, y, grid):
        """
        Vérifie si un mouvement est légal.

        Args:
            pawn: Le pion à déplacer
            x, y: Coordonnées de destination
            grid: La grille de jeu

        Returns:
            bool: True si le mouvement est légal, False sinon
        """
        # 1. Vérifications de base
        if not (0 <= x < 5 and 0 <= y < 5):
            return False

        # Vérifier si la case de destination est vide ou contient un pion plus fort
        target_value = grid.grid[y][x][-1] if len(grid.grid[y][x]) > 0 else 0
        if not (target_value == 0 or target_value > pawn.type):
            return False

        # Calculer les distances
        dx = abs(pawn.x - x)
        dy = abs(pawn.y - y)

        # 2. Vérification selon le type de mouvement
        if pawn.mouvement == "L":
            return Mouvement._check_L_movement(dx, dy)
        elif pawn.mouvement == "+":
            return Mouvement._check_plus_movement(pawn, x, y, grid, dx, dy)
        elif pawn.mouvement == "X":
            return Mouvement._check_diagonal_movement(pawn, x, y, grid, dx, dy)
        elif pawn.mouvement == "*":
            return Mouvement._check_star_movement(pawn, x, y, grid, dx, dy)

        return False

    @staticmethod
    def _check_L_movement(dx, dy):
        """Vérifie le mouvement en L"""
        return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

    @staticmethod
    def _check_plus_movement(pawn, x, y, grid, dx, dy):
        """Vérifie le mouvement en +"""
        # Mouvement horizontal ou vertical uniquement
        if not ((dx > 0 and dy == 0) or (dx == 0 and dy > 0)):
            return False

        # Vérifier le chemin
        if dx > 0:  # Mouvement horizontal
            start_x = min(pawn.x, x) + 1
            end_x = max(pawn.x, x)
            return all(len(grid.grid[y][i]) == 0 for i in range(start_x, end_x))
        else:  # Mouvement vertical
            start_y = min(pawn.y, y) + 1
            end_y = max(pawn.y, y)
            return all(len(grid.grid[i][x]) == 0 for i in range(start_y, end_y))

    @staticmethod
    def _check_diagonal_movement(pawn, x, y, grid, dx, dy):
        """Vérifie le mouvement en diagonale"""
        if dx != dy:
            return False

        # Déterminer la direction
        step_x = 1 if x > pawn.x else -1
        step_y = 1 if y > pawn.y else -1

        # Vérifier le chemin
        curr_x, curr_y = pawn.x + step_x, pawn.y + step_y
        while curr_x != x and curr_y != y:
            if len(grid.grid[curr_y][curr_x]) > 0:
                return False
            curr_x += step_x
            curr_y += step_y

        return True

    @staticmethod
    def _check_star_movement(pawn, x, y, grid, dx, dy):
        """Vérifie le mouvement en étoile"""
        # Peut se déplacer en + ou en X
        if dx == dy:  # Mouvement diagonal
            return Mouvement._check_diagonal_movement(pawn, x, y, grid, dx, dy)
        else:  # Mouvement horizontal ou vertical
            return Mouvement._check_plus_movement(pawn, x, y, grid, dx, dy)