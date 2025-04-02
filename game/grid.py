from game.env_var import *
import numpy as np
from colorama import Back


class Grid:
    def __init__(self, size, pawns):
        """
        Initialise la grille de jeu.

        Args:
            size: Taille de la grille (nombre de lignes/colonnes)
            pawns: Liste des pions présents sur le plateau
        """
        # Initialisation de la grille avec des cases vides
        self.size = size
        self.grid = np.zeros((size, size), dtype=object)
        for i in range(size):
            for j in range(size):
                self.grid[i][j] = np.array([0])

        # Référence à tous les pions
        self.all_pawns = pawns

        # Placement des pions sur la grille
        for pawn in pawns:
            if pawn.x >= 0 and pawn.y >= 0:  # Vérifier que le pion est sur le plateau
                self.grid[pawn.y][pawn.x] = np.array([pawn.type])

        # Variables pour la vérification de l'intégrité
        self.error_rounds = []  # Remplace 'aaa' par un nom plus descriptif
        self.isbroken = False

    def __getitem__(self, key):
        """
        Permet d'accéder à la grille avec la notation grid[i].
        """
        return self.grid[key]

    def display(self):
        """
        Affiche la grille de jeu avec les pions colorés.
        Actuellement désactivé (pass) pour éviter l'affichage pendant l'entraînement.
        """
        # La fonction est désactivée (pass) mais le code est conservé pour référence
        pass

        # Code d'affichage commenté
        '''
        print("----")
        row = ""
        for i in range(self.size):
            print("|", end="")
            for j in range(self.size):
                if self.grid[i][j][0] != 0:
                    for pawn in self.grid[i][j]:
                        for pawntype in self.all_pawns:
                            if pawntype.type == pawn and pawntype.x == j and pawntype.y == i:
                                if pawntype.color == "blue":
                                    row += Back.BLUE + str(pawntype.type)
                                elif pawntype.color == "orange":
                                    row += Back.RED + str(pawntype.type)
                else:
                    row += Back.BLACK + "0 "
            print(row + Back.BLACK + "|")
            row = ""
        print("----")
        '''

    def checkgrid(self, round):
        """
        Vérifie l'intégrité de la grille.
        Chaque type de pion (1-4) doit apparaître exactement 2 fois sur la grille.

        Args:
            round: Le tour actuel (pour enregistrer quand une erreur se produit)
        """
        # Compteur pour chaque type de pion (1-4)
        counter = [0, 0, 0, 0]

        # Compter les occurrences de chaque type de pion
        for pawn_type in range(1, 5):  # Types de pions 1, 2, 3, 4
            for j in range(self.size):
                for i in range(self.size):
                    if pawn_type in self.grid[j][i]:
                        counter[pawn_type - 1] += 1

        # Vérifier si un type de pion apparaît plus de 2 fois
        for pawn_type, count in enumerate(counter, 1):
            if count > 2:
                self.error_rounds.append(round)
                self.isbroken = True
                print(f"Error: Pawn type {pawn_type} appears {count} times (should be 2)")
                break

        # Vérifier si tous les types de pions apparaissent exactement 2 fois
        if any(count != 2 for count in counter) and self.error_rounds:
            print(f"Grid integrity error detected at round {self.error_rounds[0]}")
            self.isbroken = True

    def getfinalstack(self, x, y):
        """
        Retourne la pile finale de pions à une position donnée.

        Args:
            x: Coordonnée x
            y: Coordonnée y

        Returns:
            Liste des pions à cette position, triée par type (décroissant)
        """
        final_stack = []

        # Collecter tous les pions à cette position
        for pawn in self.all_pawns:
            if pawn.x == x and pawn.y == y:
                pawn_info = {
                    "color": pawn.color,
                    "type": pawn.type,
                    "pos": (pawn.x, pawn.y),
                    "mouvement": pawn.mouvement
                }
                final_stack.append(pawn_info)

        # Trier la pile par type de pion (ordre décroissant)
        final_stack.sort(key=lambda x: x["type"], reverse=True)

        return final_stack