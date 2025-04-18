import os
import pandas as pd
from enum import Enum
from datetime import datetime


class PawnType(Enum):
    DONKEY = 4
    DOG = 3
    CAT = 2
    ROOSTER = 1


class DataManager:
    def __init__(self, overwrite=False, url=""):
        """
        :type overwrite: bool
        :type url: str
        :param overwrite:   define if we want to overwrite data into a specific file
        :param url:         the path of the file to overwrite
        """
        self.root = "./data/dataset"
        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_name = f"{self.root}/game_{self.time}.csv" if not overwrite else url

        # We create a dataframe with the columns
        self.df = pd.DataFrame(
            columns=["ai", "victory", "turn", "retreat", "initial_pos", "final_stack", "donkey", "dog", "cat",
                     "rooster"])
        # initial position of each pawn
        self.initial_pos = []
        # Each pawn (no matter the color) has its own list to store its position during the game
        self.donkey = []
        self.dog = []
        self.cat = []
        self.rooster = []

        # Checking if the root directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if overwrite:
            # read previous csv
            previous_dataframe = pd.read_csv(self.file_name)
            self.df = previous_dataframe

        self.df.to_csv(self.file_name, header=True, index=False)

    def print_initial_pos(self):
        print(self.initial_pos)

    def set_initial_pos(self, color, _type, pos):
        self.initial_pos.append({"color": color, "type": _type, "pos": pos})

    # Write data to the CSV file
    def write(self, ai, winner, turn, retreat, final_stack):
        obj = {
            "ai": ai,
            "victory": winner.upper(),
            "turn": turn,
            "retreat": retreat,
            "initial_pos": self.initial_pos,
            "final_stack": final_stack,
            "donkey": self.donkey,
            "dog": self.dog,
            "cat": self.cat,
            "rooster": self.rooster
        }

        self.df.loc[len(self.df)] = obj
        self.donkey, self.dog, self.cat, self.rooster, self.initial_pos = [], [], [], [], []
        self.df.to_csv(self.file_name, header=True, index=False)

    def update_pawn_history(self, color, _type, pos, turn):
        # data is an object containing the color, type, pos and turn of the pawn
        obj = {"color": color, "pos": pos, "type": _type, "turn": turn}
        match PawnType(_type):
            case PawnType.DONKEY:
                self.donkey.append(obj)
            case PawnType.DOG:
                self.dog.append(obj)
            case PawnType.CAT:
                self.cat.append(obj)
            case PawnType.ROOSTER:
                self.rooster.append(obj)

    def to_csv(self):
        self.df.to_csv(self.file_name, header=True, index=False)
