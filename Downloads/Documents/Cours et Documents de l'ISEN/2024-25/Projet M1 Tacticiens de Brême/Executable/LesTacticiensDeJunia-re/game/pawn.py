from game.env_var import *
from game.mouvement import Mouvement
import numpy as np
import copy


class Pawn:
    """
    Classe représentant un pion dans le jeu.
    """

    def __init__(self, x, y, type, mouvement, color):  
        """  
        Initialise un pion.  
        
        Args:  
            x: Coordonnée x du pion  
            y: Coordonnée y du pion  
            type: Type du pion (1-4)  
            mouvement: Type de mouvement du pion  
            color: Couleur du pion ("blue" ou "orange")  
        """  
        self.x = x  
        self.y = y  
        if type < 5:  
            self.type = type  
        self.mouvement = mouvement  
        self.color = color  

    def move(self, x, y, grid, allpawns, game, simulate=False):  
        """  
        Déplace le pion vers une nouvelle position.  
        
        Args:  
            x: Coordonnée x de destination  
            y: Coordonnée y de destination  
            grid: La grille de jeu  
            allpawns: Liste de tous les pions  
            game: Instance du jeu  
            simulate: Si True, simule le mouvement sans l'effectuer réellement  
            
        Returns:  
            list: [mouvement_réussi, victoire] où mouvement_réussi est un booléen indiquant  
                si le mouvement a été effectué, et victoire est un booléen indiquant  
                si le mouvement a conduit à une victoire  
        """  
    
        # Avant tout, vérifier qu'on ne duplique pas un pion
        for pawn in allpawns:
            if pawn != self and pawn.x == x and pawn.y == y and pawn.type == self.type:
                print(f"ERREUR : Duplication de {self.color} {self.type} @ ({x},{y})")
                return [False, False]
        
        # Si la pile a déjà 4 pions, on ne peut plus la déplacer (fix pour minimax profondeur 3)  
        if len(grid[self.y][self.x]) == 4:  
            return np.array(grid[self.y][self.x])  
        
        # Vérifier si le mouvement est hors de la grille  
        if x < 0 or x >= len(grid.grid) or y < 0 or y >= len(grid.grid) or (self.x == x and self.y == y):  
            return [False, False]  
        
        # Si le jeu est en phase d'initialisation, les règles sont différentes  
        if not game.initializing:  
            # Vérifier si le mouvement est légitime  
            if Mouvement.legit_mouv(self, x, y, grid):  
                # Calculer la pile de pions  
                grid.grid[y][x] = self.stack(x, y, grid, allpawns)  
                self.x = x  
                self.y = y  
                
                if not simulate:  
                    grid.display()  

                # Vérifier si le jeu est gagné (pile de 4)  
                if len(grid.grid[y][x]) == 4:  
                    return [True, True]  
                else:  
                    return [True, False]  
            else:  
                if not simulate:  
                    print("Cant move there")  
                    print("pawn moves", self.mouvement)  
                return [False, False]  
        else:  
            # Règles pour la phase d'initialisation  
            if (y == 0 and self.color == "blue") or (y == 4 and self.color == "orange"):  
                grid.grid[y][x] = np.array([self.type])  
                self.x = x  
                self.y = y  
                if not simulate:  
                    grid.display()  
                return [True, False]  
            else:  
                if not simulate:  
                    print("Cant move there")  
                    print("pawn moves", self.mouvement)  
                return [False, False]  

    def display(self):  
        """  
        Affiche les informations du pion.  
        """  
        print(f"-Pawn {self.color} {self.type} -  x: {self.x} y: {self.y} mouvement: {self.mouvement}")  

    def stack(self, x, y, grid, allpawns):  
        """  
        Gère l'empilement des pions lors d'un déplacement.  
        
        Args:  
            x: Coordonnée x de destination  
            y: Coordonnée y de destination  
            grid: La grille de jeu  
            allpawns: Liste de tous les pions  
            
        Returns:  
            numpy.array: La nouvelle pile de pions à la position de destination  
        """  
        # Cas 1: Déplacement partiel d'une pile  
        if len(grid[self.y][self.x]) > 1 and self.type != grid[self.y][self.x][0]:  
            return self._move_partial_stack(x, y, grid, allpawns)  
        
        # Cas 2: Déplacement d'une pile entière et fusion avec une autre pile  
        elif len(grid[self.y][self.x]) > 1 and grid[y][x][-1] > self.type and self.type == grid[self.y][self.x][0]:  
            return self._move_and_merge_stack(x, y, grid, allpawns)  
        
        # Cas 3: Déplacement d'une pile entière  
        elif len(grid[self.y][self.x]) > 1 and self.type == grid[self.y][self.x][0]:  
            return self._move_full_stack(x, y, grid, allpawns)  
        
        # Cas 4: Déplacement d'un pion unique vers une case vide  
        elif np.array_equal(grid[y][x], np.array([0])) and len(grid[self.y][self.x]) <= 1:  
            grid[self.y][self.x] = np.array([0])  
            return np.array([self.type])  
        
        # Cas 5: Déplacement d'un pion unique et fusion avec une pile  
        elif grid[y][x][-1] > self.type:  
            grid[self.y][self.x] = np.array([0])  
            return np.append(grid[y][x], self.type)  
        
        else:  
            print("Cant move there definitely don't")  
            return 0  

    def _move_partial_stack(self, x, y, grid, allpawns):  
        """  
        Déplace une partie d'une pile de pions.  
        """  
        # Trouver l'index du pion dans la pile  
        tmpid = 0  
        for id in range(len(grid[self.y][self.x])):  
            if grid[self.y][self.x][id] == self.type:  
                tmpid = id  
        
        # Séparer la pile en deux parties  
        stayingarr = grid[self.y][self.x].copy()  
        
        if grid[y][x][-1] != 0:  
            goingarr = grid[y][x]  
        else:  
            goingarr = np.array([], dtype=int)  

        # Déplacer les pions de la pile d'origine vers la pile de destination  
        for i in range(tmpid, len(grid[self.y][self.x])):  
            if i == tmpid:  
                goingarr = np.append(goingarr, stayingarr[i])  
                stayingarr = np.delete(stayingarr, i)  
            else:  
                goingarr = np.append(goingarr, stayingarr[i-1])  
                stayingarr = np.delete(stayingarr, i-1)  
        
        grid[self.y][self.x] = stayingarr  
        grid[y][x] = goingarr  
        
        # Mettre à jour les positions des pions  
        pawnstomove = []  
        pawnstostay = []  
        tmpxy = [self.x, self.y]  
        
        for pawn in allpawns:  
            if pawn.type in goingarr and pawn.x == self.x and pawn.y == self.y:  
                pawnstomove.append(pawn)  
            if pawn.type in stayingarr and pawn.x == self.x and pawn.y == self.y:  
                pawnstostay.append(pawn)  
        
        for pawn in pawnstomove:  
            pawn.x = x  
            pawn.y = y  
        
        for pawn in pawnstostay:  
            pawn.x = tmpxy[0]  
            pawn.y = tmpxy[1]  
        
        return goingarr  

    def _move_and_merge_stack(self, x, y, grid, allpawns):  
        """  
        Déplace une pile entière et la fusionne avec une autre pile.  
        """  
        pawns = []  
        currentstack = grid[self.y][self.x]  
        stack = grid[y][x]  
        
        grid[self.y][self.x] = np.array([0])  
        for pawn in currentstack:  
            stack = np.append(stack, pawn)  
        
        for pawn in allpawns:  
            if np.isin(pawn.type, currentstack) and pawn.x == self.x and pawn.y == self.y:  
                pawns.append(pawn)  
        
        for pawn in pawns:  
            pawn.x = x  
            pawn.y = y  
        
        return stack  

    def _move_full_stack(self, x, y, grid, allpawns):  
        """  
        Déplace une pile entière.  
        """  
        tmparr = grid[self.y][self.x]  
        pawns = []  
        
        grid[self.y][self.x] = np.array([0])  
        
        for pawn in allpawns:  
            if np.isin(pawn.type, tmparr) and pawn.x == self.x and pawn.y == self.y:  
                pawns.append(pawn)  
        
        for pawn in pawns:  
            pawn.x = x  
            pawn.y = y  

        return tmparr