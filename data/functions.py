import numpy as np
import pandas as pd

def get_ai_information(_ai):
    _type = _ai["type"]
    if _type == "M":
        return f"AI {_ai['color']}: Minimax - depth: {_ai['depth']}"
    elif _type == "R":
        return f"AI {_ai['color']}: Random"


def count_stack_by_color(row):
    colors = []
    for pawn in row:
        colors.append(pawn["color"])
    #print("\n", colors)
    #print("Pile de la même couleur: ", len(set(colors)) == 1)
    return len(set(colors)) == 1


# Retourne un dataframe qui contient le nombre de fois où chaque case a été occupée par un pion de la couleur donnée
def get_grid_occupation(df, _type=None, color=None):
    grid = np.zeros((5, 5))
    pawns = ["donkey", "cat", "dog", "rooster"] if _type is None else [_type]

    for _, row in df.iterrows():
        for element in pawns:
            for pawn in row[element]:
                x, y = pawn["pos"]
                if color is None:
                    grid[y][x] += 1
                elif pawn["color"] == color:
                    grid[y][x] += 1
    return pd.DataFrame(grid, columns=["0", "1", "2", "3", "4"])