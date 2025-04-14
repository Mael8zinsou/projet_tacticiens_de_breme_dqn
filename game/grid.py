from game.env_var import *
import numpy as np
from colorama import Back


class Grid:
    """
    Classe représentant la grille de jeu.
    """

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
        self.error_rounds = []  # Liste des tours où une erreur a été détectée  
        self.isbroken = False   # Indique si la grille est dans un état invalide  

    def __getitem__(self, key):  
        """  
        Permet d'accéder à la grille avec la notation grid[i].  
        
        Args:  
            key: Indice de ligne  
            
        Returns:  
            La ligne correspondante de la grille  
        """  
        return self.grid[key]  

    def display(self):  
        """  
        Affiche la grille de jeu avec les pions colorés.  
        """  
        # commentez ce bloc pour désactiver l'affichage  
        print("----")  
        for i in range(self.size):  
            print("|", end="")  
            row = ""  
            for j in range(self.size):  
                if self.grid[i][j][0] != 0:  
                    for pawn_value in self.grid[i][j]:  
                        for pawn in self.all_pawns:  
                            if pawn.type == pawn_value and pawn.x == j and pawn.y == i:  
                                if pawn.color == "blue":  
                                    row += Back.BLUE + str(pawn.type)  
                                elif pawn.color == "orange":  
                                    row += Back.RED + str(pawn.type)  
                else:  
                    row += Back.BLACK + "0 "  
            print(row + Back.BLACK + "|")  
        print("----")  
        # Pour l'instant, l'affichage est activé  
        # pass  

    def checkgrid(self, round):  
        """  
        Vérifie l'intégrité de la grille.  
        Chaque type de pion (1-4) doit apparaître exactement 2 fois sur la grille.  

        Args:  
            round: Le tour actuel (pour enregistrer quand une erreur se produit)  
            
        Returns:  
            bool: True si la grille est valide, False sinon  
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
                return False  

        # Vérifier si tous les types de pions apparaissent exactement 2 fois  
        if any(count != 2 for count in counter):  
            if not self.error_rounds:  # Ajouter le tour actuel seulement s'il n'y a pas déjà d'erreurs  
                self.error_rounds.append(round)  
            self.isbroken = True  
            print(f"Grid integrity error detected at round {self.error_rounds[0]}")  
            return False  
            
        return True  

    def getfinalstack(self, x, y):  
        """  
        Retourne la pile finale de pions à une position donnée.  

        Args:  
            x: Coordonnée x  
            y: Coordonnée y  

        Returns:  
            list: Liste des pions à cette position, triée par type (décroissant)  
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