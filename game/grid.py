from game.env_var import *
import numpy as np
from colorama import Back


class Grid:
    def __init__(self, size, pawns):
        self.size = size
        self.grid = np.zeros((size, size), dtype=object)
        for i in range(0, size):
            for j in range(0, size):
                self.grid[i][j] = np.array([0])
        self.all_pawns = pawns
        for pawn in pawns:
            self.grid[pawn.y][pawn.x] = np.array([pawn.type])
        self.aaa = []
        self.isbroken = False

    def __getitem__(self, key):
        return self.grid[key]

    def display(self):
        #print the grid with colors
        '''
        print("------------")
        row = ""
        for i in range(0, self.size):
            print("|", end="")
            for j in range(0, self.size):
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

        print("------------")
        
        # easy way to print the grid
        # for row in self.grid:
        #     print(row)
        # print("\n")'''
        pass

    # Can be used to debug, check if the grid is correct
    def checkgrid(self, round):
        pawns = [1, 2, 3, 4]
        counter = [0, 0, 0, 0]
        for pawn in pawns:
            for i in range(0, self.size):
                for j in range(0, self.size):
                    if pawn in self.grid[j][i]:
                        counter[pawn - 1] += 1
                        if counter[pawn - 1] > 2:
                            self.aaa.append(round)
        for stat in counter:
            if stat != 2:
                if self.aaa != []:
                    print(self.aaa[0])
                    self.isbroken = True

    # Return the final stack of pawns at a given position
    def getfinalstack(self, x, y):
        final_stack = []
        tmpobj = {}
        for pawn in self.all_pawns:
            if pawn.x == x and pawn.y == y:
                tmpobj = {"color": pawn.color, "type": pawn.type, "pos": (pawn.x, pawn.y), "mouvement": pawn.mouvement}

                final_stack.append(tmpobj)

        final_stack.sort(key=lambda x: x["type"], reverse=True)

        return final_stack
