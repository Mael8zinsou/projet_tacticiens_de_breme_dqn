from game.env_var import *
from game.mouvement import Mouvement
import numpy as np
import copy


class Pawn:

    def __init__(self, x, y, type, mouvement, color):
        self.x = x
        self.y = y
        if type < 5:
            self.type = type
        self.mouvement = mouvement
        self.color = color

    def move(self, x, y, grid, allpawns, game, simulate = False):
        #fix depth 3 minmax
        if len(grid[self.y][self.x]) == 4:
            return np.array(grid[self.y][self.x])
        
        #Check if the mouvement is out of grid
        if x < 0 or x > len(grid.grid) or y < 0 or y > len(grid.grid) or (self.x == x and self.y == y):
            return [False, False]
        
        # if the game is initializing, we can't play as usual
        if not game.initializing:
            #Check if the mouvement is legit
            if Mouvement.legit_mouv(self, self, x, y, grid):
                # Calculate the stack of pawns (and if a pawn is mouved to an empty cell, the stack is the pawn itself)
                grid.grid[y][x] = self.stack(x, y, grid, allpawns)
                self.x = x
                self.y = y
                if not simulate:
                    grid.display()

                # Check if the game is won
                if len(grid.grid[y][x]) == 4:
                    if not simulate:
                        #print("moving", self.mouvement)
                        pass
                    return [True, True]
                # Check if the pawn moved successfully
                else:
                    if not simulate:
                        #print("moving", self.mouvement)
                        pass
                    return [True, False]
                
            # If the mouvement is not legit
            else:
                if not simulate:
                    print("Cant move there")
                    print("pawn moves", self.mouvement)
                return [False, False]
        else:
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
        print("-Pawn", self.color, self.type, "-  x:", self.x, "y:", self.y, "mouvement:", self.mouvement)

    def stack(self, x, y, grid, allpawns):
        # Move partially the stack of pawns
        if len(grid[self.y][self.x])>1 and self.type != grid[self.y][self.x][0]:
            tmpid=0
            for id in range(len(grid[self.y][self.x])):
                if grid[self.y][self.x][id] == self.type:
                    tmpid = id
            stayingarr = grid[self.y][self.x].copy()
            
            if grid[y][x][-1] != 0:
                goingarr = grid[y][x]
            else:
                goingarr = np.array([], dtype=int)

            for i in range(tmpid,len(grid[self.y][self.x])):
                if i == tmpid:
                    goingarr = np.append(goingarr, stayingarr[i])
                    stayingarr = np.delete(stayingarr, i)
                else:
                    goingarr = np.append(goingarr, stayingarr[i-1])
                    stayingarr = np.delete(stayingarr, i-1)
            
            grid[self.y][self.x] = stayingarr
            grid[y][x] = goingarr
            
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
        
        # Move the stack of pawns and merge the stack
        elif len(grid[self.y][self.x])>1 and grid[y][x][-1] > self.type and self.type == grid[self.y][self.x][0]:
            pawns=[]
            currentstack = grid[self.y][self.x]
            stack = grid[y][x]
            
            grid[self.y][self.x] = np.array([0])
            for pawn in currentstack:
                stack = np.append(stack, pawn)
                
            for pawn in allpawns:
                if  np.isin(pawn.type, currentstack) and pawn.x == self.x and pawn.y == self.y:
                    pawns.append(pawn)
                    
            for pawn in pawns:
                pawn.x = x
                pawn.y = y
                
            
            return stack
        
        # Move the stack of pawns
        elif len(grid[self.y][self.x])>1 and self.type == grid[self.y][self.x][0]:
            tmparr = grid[self.y][self.x]
            pawns=[]
            grid[self.y][self.x] = np.array([0])
            for pawn in allpawns:
                if  np.isin(pawn.type, tmparr) and pawn.x == self.x and pawn.y == self.y:
                    pawns.append(pawn)
            for pawn in pawns:
                pawn.x = x
                pawn.y = y

            return tmparr
        
        # Move a single pawn to an empty cell
        elif np.array_equal(grid[y][x], np.array([0])) and len(grid[self.y][self.x]) <= 1:
            grid[self.y][self.x] = np.array([0])
            return np.array([self.type])
        
        # Move a single pawn and merge it to the stack
        elif grid[y][x][-1] > self.type:
            grid[self.y][self.x] = np.array([0])
            return np.append(grid[y][x], self.type)
        
        else:
            print("Cant move there definitely don't")
            return 0