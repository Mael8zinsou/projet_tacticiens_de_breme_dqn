from game.env_var import *

class Mouvement:
    def __init__(self):
        pass
    
    # Check if the mouvement is legit for each kind of mouvement    
    def legit_mouv(self, pawn, x, y, grid):
        if (x < len(grid.grid) and y < len(grid.grid)) and (grid.grid[y][x][-1] > pawn.type or grid.grid[y][x][-1] == 0):
            # Check if the pawn is moving in the right direction for L mouvs
            if pawn.mouvement == "L":
                if (2 == distance(pawn, x, y)[0] and 1 == distance(pawn, x, y)[1]) or (1 == distance(pawn, x, y)[0] and 2 == distance(pawn, x, y)[1]):
                    return True
                
            # Check if the pawn is moving in the right direction for + mouvs
            elif pawn.mouvement == "+":
                if (0 != distance(pawn, x, y)[0] and 0 == distance(pawn, x, y)[1]) or (0== distance(pawn, x, y)[0] and 0 != distance(pawn, x, y)[1]):
                    if (self.y == y):
                        if (self.x <x):
                            dest = max(self.x+1, x)
                            src = min(self.x+1, x)
                        else:
                            dest = max(self.x-1, x)
                            src = min(self.x-1, x)
                        for i in range(src, dest+1):
                            if grid.grid[y][i][-1] != 0:
                                return False
                    else:
                        if (self.y <y):
                            dest = max(self.y+1, y)
                            src = min(self.y+1, y)
                        else:
                            dest = max(self.y-1, y)
                            src = min(self.y-1, y)
                        for i in range (src, dest+1):
                            if grid.grid[i][x][-1] != 0:
                                return False
                    return True
                
            # Check if the pawn is moving in the right direction for X mouvs
            elif pawn.mouvement == "X":
                if distance(pawn, x, y)[1] == distance(pawn, x, y)[0]:
                    if (self.y <y):
                        ydest = max(self.y+1, y)
                        ysrc = min(self.y+1, y)
                    else:
                        ydest = max(self.y-1, y)
                        ysrc = min(self.y-1, y)
                    if (self.x <x):
                        xdest = max(self.x+1, x)
                        xsrc = min(self.x+1, x)
                    else:
                        xdest = max(self.x-1, x)
                        xsrc = min(self.x-1, x)
                    
                    for i in range (xsrc, xdest+1):
                        if grid.grid[ydest][i][-1] != 0 and pawn.type >= grid.grid[ydest][i][-1]:
                            return False
                        if ysrc < ydest:
                            ydest -= 1
                        
                    return True
                
            # Check if the pawn is moving in the right direction for * mouvs
            elif pawn.mouvement == "*":
                if (0 != distance(pawn, x, y)[0] and 0 == distance(pawn, x, y)[1]) or (0== distance(pawn, x, y)[0] and 0 != distance(pawn, x, y)[1]) or (distance(pawn, x, y)[1] == distance(pawn, x, y)[0]):
                    if (self.y == y):
                        if (self.x <x):
                            dest = max(self.x+1, x)
                            src = min(self.x+1, x)
                        else:
                            dest = max(self.x-1, x)
                            src = min(self.x-1, x)
                        for i in range(src, dest+1):
                            if grid.grid[y][i][-1] == 1:
                                return False
                    else:
                        if (self.y <y):
                            dest = max(self.y+1, y)
                            src = min(self.y+1, y)
                        else:
                            dest = max(self.y-1, y)
                            src = min(self.y-1, y)
                        for i in range (src, dest+1):
                            if grid.grid[i][x][-1] == 1:
                                return False
                    
                    if (self.y <y):
                        ydest = max(self.y+1, y)
                        ysrc = min(self.y+1, y)
                    else:
                        ydest = max(self.y-1, y)
                        ysrc = min(self.y-1, y)
                    if (self.x <x):
                        xdest = max(self.x+1, x)
                        xsrc = min(self.x+1, x)
                    else:
                        xdest = max(self.x-1, x)
                        xsrc = min(self.x-1, x)
                    
                    for i in range (xsrc, xdest+1):
                        if grid.grid[ydest][i][-1] == 1 and distance(pawn, x, y)[1] == distance(pawn, x, y)[0]:
                            return False
                        if ysrc < ydest:
                            ydest -= 1
                    
                    return True
            else:
                return False