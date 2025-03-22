# This file contains the environment variables for the game

# Used to calculate the distance between two points and check if the mouvement is legit
basic_mouvements = {
    "*":"",
    "L":[["2","1"],["1","2"]],
    "X":"",
    "+":"",
}

# calculate the distance between two points
def distance(pawn, x, y):
    x = abs(pawn.x - x)
    y = abs(pawn.y - y)
    return [x, y]

# Pawns that must play (that is in the retreat area)
pawns_must_play = {
    "orange":[],
    "blue":[],
}