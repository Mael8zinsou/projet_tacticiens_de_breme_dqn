
from game.env_var import *

class Mouvement:
    """
    Classe gérant les règles de déplacement des pions.
    Vérifie si un mouvement est légal selon le type de mouvement du pion.
    """

    @staticmethod
    def legit_mouv(pawn, pawn2, x, y, grid):
        """Vérifie si un mouvement est légal"""
        # Vérifier que la destination est dans la grille
        if x < 0 or x >= len(grid.grid) or y < 0 or y >= len(grid.grid):
            return False

        # Vérifier que le pion ne reste pas sur place
        if x == pawn.x and y == pawn.y:
            return False

        # Vérifier que le pion peut se poser sur la destination
        if len(grid.grid[y][x]) > 0 and grid.grid[y][x][0] != 0 and grid.grid[y][x][0] <= pawn.type:
            return False

        # Calculer la distance
        dx = abs(pawn.x - x)
        dy = abs(pawn.y - y)

        # Vérifier le type de mouvement
        if pawn.mouvement == "L":
            # Mouvement en L: 2 cases dans une direction, 1 dans l'autre
            return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

        elif pawn.mouvement == "+":
            # Mouvement en +: horizontal ou vertical
            if (dx > 0 and dy == 0) or (dx == 0 and dy > 0):
                # Vérifier que le chemin est libre
                if dy == 0:  # Horizontal
                    start_x = min(pawn.x, x) + 1
                    end_x = max(pawn.x, x)
                    for i in range(start_x, end_x):
                        if grid.grid[y][i][0] != 0:
                            return False
                else:  # Vertical
                    start_y = min(pawn.y, y) + 1
                    end_y = max(pawn.y, y)
                    for i in range(start_y, end_y):
                        if grid.grid[i][x][0] != 0:
                            return False
                return True
            return False

        elif pawn.mouvement == "X":
            # Mouvement en X: diagonal
            if dx == dy and dx > 0:
                # Vérifier que le chemin est libre
                x_step = 1 if x > pawn.x else -1
                y_step = 1 if y > pawn.y else -1
                cx, cy = pawn.x + x_step, pawn.y + y_step

                while cx != x and cy != y:
                    if grid.grid[cy][cx][0] != 0:
                        return False
                    cx += x_step
                    cy += y_step
                return True
            return False

        elif pawn.mouvement == "*":
            # Mouvement en *: horizontal, vertical ou diagonal
            if (dx > 0 and dy == 0) or (dx == 0 and dy > 0) or (dx == dy and dx > 0):
                # Vérifier que le chemin est libre
                if dy == 0:  # Horizontal
                    start_x = min(pawn.x, x) + 1
                    end_x = max(pawn.x, x)
                    for i in range(start_x, end_x):
                        if grid.grid[y][i][0] != 0:
                            return False
                elif dx == 0:  # Vertical
                    start_y = min(pawn.y, y) + 1
                    end_y = max(pawn.y, y)
                    for i in range(start_y, end_y):
                        if grid.grid[i][x][0] != 0:
                            return False
                else:  # Diagonal
                    x_step = 1 if x > pawn.x else -1
                    y_step = 1 if y > pawn.y else -1
                    cx, cy = pawn.x + x_step, pawn.y + y_step

                    while cx != x and cy != y:
                        if grid.grid[cy][cx][0] != 0:
                            return False
                        cx += x_step
                        cy += y_step
                return True
            return False

        return False
