"""
Ce fichier contient les variables d'environnement pour le jeu Les Tacticiens de Brême.
"""

# Types de mouvements de base pour les pions
# * : Mouvement en étoile (horizontal, vertical ou diagonal)
# L : Mouvement en L (2 cases dans une direction, 1 dans l'autre)
# X : Mouvement en diagonale
# + : Mouvement horizontal ou vertical
basic_mouvements = {
    "*":"",
    "L":[["2","1"],["1","2"]],
    "X":"",
    "+":"",
}

def distance(pawn, x, y):
    """
    Calcule la distance entre un pion et une position cible.

    Args:
        pawn: Le pion dont on veut calculer la distance
        x: Coordonnée x de la position cible
        y: Coordonnée y de la position cible

    Returns:
        list: [distance_x, distance_y] en valeur absolue
    """
    x = abs(pawn.x - x)
    y = abs(pawn.y - y)
    return [x, y]

# Pawns that must play (that is in the retreat area)
# Utilisé pour suivre les pions qui doivent jouer en priorité selon les règles de retraite
pawns_must_play = {
    "orange":[],
    "blue":[],
}

# Constantes additionnelles (n'affectent pas le code existant)
# Ces constantes peuvent être utilisées dans de nouveaux développements
GRID_SIZE = 5
PAWN_TYPES = [1, 2, 3, 4]