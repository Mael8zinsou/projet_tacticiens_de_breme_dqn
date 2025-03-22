import random

import numpy as np
from game.env_var import *


class Dummyai:
    def __init__(self, color):
        self.color = color
        self.type = "R"

    def playrandom(self, nextmouvs):
        if not nextmouvs:  # Vérifie si nextmouvs est vide
            print("No moves available")
            return None
        # return a random move from the list of all possible moves
        # if the pawn must play array isn't empty, the move is chosen in the list of pawns that must play
        if pawns_must_play[self.color] != []:
            print(nextmouvs)
            while nextmouvs[np.random.randint(0, len(nextmouvs))] not in pawns_must_play[self.color]:
                return nextmouvs[np.random.randint(0, len(nextmouvs))]
        else :
            return nextmouvs[np.random.randint(0, len(nextmouvs))]