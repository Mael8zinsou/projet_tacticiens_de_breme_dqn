import numpy as np
from game.pawn import Pawn
from game.grid import Grid
from game.mouvement import Mouvement
from game.env_var import *
import copy

# Constantes
STACK_MULTIPLIER = 2000
WINNING_STACK_BONUS = 10000
BONUS_COLOR = 500
THREAT_MULTIPLIER = 500
MAX_GRID_SIZE = 5


class Game:
    """
    Classe principale du jeu Les Tacticiens de Brême.
    Gère la logique du jeu, l'initialisation et l'évaluation.
    """

    def __init__(self, data_manager=None, manual_mode=False, use_ai=False, ai_types=None):  
        """  
        Initialise le jeu.  
        
        Args:  
            data_manager: Gestionnaire de données pour enregistrer les statistiques  
            manual_mode: Si True, permet le placement manuel des pions  
            use_ai: Si True, utilise l'IA pour jouer  
            ai_types: Tuple indiquant les types d'IA à utiliser (ex: (0, 1) pour IA aléatoire vs Minimax)  
        """  
        self.data_manager = data_manager  
        self.pawns = []  
        self.num_retreat = 0  
        self.mode = manual_mode  
        self.initializing = True  
        self.use_ai = use_ai  
        self.ai_types = ai_types if ai_types else (0, 0)  
        self.winner = None  # Stocke le gagnant de la partie  

        # Initialisation des pions selon le mode  
        if manual_mode:  
            manual_placement = get_user_input("\nDo you want to place the pawns manually ? (y/n): ", ["y", "Y", "n", "N"])  
            if manual_placement.lower() == "n":  
                self.init_pawns("blue")  
                self.init_pawns("orange")  
            else:  
                self.init_pawns_manually()  
        else:  
            if not ai_types:  
                raise ValueError("AI types must be specified when manual mode is False")  
            self.init_infinite()  

        # Initialisation de la grille et affichage  
        self.grid = Grid(MAX_GRID_SIZE, self.pawns)
        self.grid.all_pawns = self.pawns  # Lien explicite
        
        self.grid.display()  

    def reset(self):  
        """  
        Réinitialise le jeu pour une nouvelle partie.  
        """  
        self.pawns = []  
        self.num_retreat = 0  
        self.initializing = True  
        self.winner = None  
        self.init_infinite()  
        self.grid = Grid(MAX_GRID_SIZE, self.pawns)  
        self.grid.display()  

    # ----  
    # Initialisation des pions  
    # ----  

    def init_infinite(self):  
        """  
        Initialise les pions pour les deux joueurs.  
        """  
        self.init_pawns("blue")  
        self.init_pawns("orange")  
        self.use_ai = True  

    def init_pawns(self, color):  
        """  
        Place les pions automatiquement sur le plateau.  
        
        Args:  
            color: Couleur des pions à placer ("blue" ou "orange")  
        """  
        mouvs = list(basic_mouvements.keys())  # Liste des mouvements possibles  
        used_positions = []  # Liste des positions déjà utilisées  

        for i in range(4):  
            # Générer une position unique  
            nextpos = self._get_unique_random_position(used_positions)  

            # Créer et ajouter le pion  
            y_pos = 0 if color == "blue" else 4  
            pawn = Pawn(nextpos, y_pos, i + 1, mouvs[i], color)  
            self.pawns.append(pawn)  

            # Mettre à jour le data_manager si disponible  
            if self.data_manager:  
                self.data_manager.set_initial_pos(color, i + 1, (nextpos, y_pos))  

    def _get_unique_random_position(self, used_positions):  
        """  
        Génère une position aléatoire unique.  
        
        Args:  
            used_positions: Liste des positions déjà utilisées  
            
        Returns:  
            int: Position unique entre 0 et MAX_GRID_SIZE-1  
        """  
        while True:  
            pos = np.random.randint(0, MAX_GRID_SIZE)  
            if pos not in used_positions:  
                used_positions.append(pos)  
                return pos  

    def init_pawns_manually(self):  
        """  
        Place les pions manuellement sur le plateau.  
        """  
        mouvs = list(basic_mouvements.keys())  
        positions_blue = []  
        positions_orange = []  

        # Placer les pions bleus  
        print("Placement des pions BLEUS")  
        for i in range(4):  
            self._place_single_pawn("blue", i, positions_blue, mouvs)  

        # Placer les pions oranges  
        print("Placement des pions ORANGES")  
        for i in range(4):  
            self._place_single_pawn("orange", i, positions_orange, mouvs)  

        print("All pawns placed")  
        print("----Starting game----")  

    def _place_single_pawn(self, color, index, positions, mouvs):  
        """  
        Place un seul pion manuellement.  
        
        Args:  
            color: Couleur du pion  
            index: Index du pion (0-3)  
            positions: Liste des positions déjà utilisées  
            mouvs: Liste des mouvements disponibles  
        """  
        y = 0 if color == "blue" else 4  

        while True:  
            try:  
                print(f"Placing {color}'s pawn {index + 1}")  
                x = int(input("x (0-4): ") or 1)  

                if 0 <= x < MAX_GRID_SIZE and x not in positions:  
                    pawn = Pawn(x, y, index + 1, mouvs[index], color)  
                    self.pawns.append(pawn)  
                    positions.append(x)  
                    break  
                else:  
                    print("Invalid position or already used")  
            except ValueError:  
                print("Invalid input. Please enter a number.")  

    # ----  
    # Gestion des mouvements  
    # ----  

    def all_next_moves(self, color):  
        """  
        Retourne tous les mouvements possibles pour une couleur donnée.  
        
        Args:  
            color: Couleur des pions ("blue" ou "orange")  
            
        Returns:  
            list: Liste des mouvements possibles sous forme [color, type, x, y]  
        """  
        # Si le jeu est en phase d'initialisation  
        if self.initializing:  
            return self._get_initialization_moves(color)  

        # Sinon, obtenir les mouvements normaux  
        return self._get_normal_moves(color)  

    def _get_initialization_moves(self, color):  
        """  
        Obtient les mouvements possibles pendant la phase d'initialisation.  
        
        Args:  
            color: Couleur des pions  
            
        Returns:  
            list: Liste des mouvements possibles  
        """  
        next_moves = []  
        for pawn in self.pawns:  
            if pawn.color == color and pawn.x == -1:  
                y_pos = 0 if color == "blue" else 4  
                for x in range(MAX_GRID_SIZE):  
                    if self.grid.grid[y_pos][x][0] == 0:  
                        next_moves.append([pawn.color, pawn.type, x, y_pos])  
        return next_moves  

    def _get_normal_moves(self, color):  
        """  
        Obtient les mouvements possibles pendant la phase normale du jeu.  
        
        Args:  
            color: Couleur des pions  
            
        Returns:  
            list: Liste des mouvements possibles  
        """  
        next_moves = []  
        for pawn in self.pawns:  
            # Vérifier si le pion est de la bonne couleur  
            if pawn.color != color:  
                continue  

            # Vérifier si le pion doit jouer (retraite)  
            if pawns_must_play[color] and pawn not in pawns_must_play[color]:  
                continue  

            # Trouver tous les mouvements légitimes pour ce pion  
            for x in range(MAX_GRID_SIZE):  
                for y in range(MAX_GRID_SIZE):  
                    if (pawn.x != x or pawn.y != y) and Mouvement.legit_mouv(pawn, x, y, self.grid):  
                        next_moves.append([pawn.color, pawn.type, x, y])  

        return next_moves  

    def isretraite(self, lastmove):  
        """  
        Vérifie si un pion est dans la zone de retraite et met à jour la liste des pions qui doivent jouer.  
        
        Args:  
            lastmove: Dernier mouvement effectué [color, type, x, y]  
            
        Returns:  
            bool: True si au moins un pion doit jouer en retraite  
        """  
        # Réinitialiser les listes de pions qui doivent jouer  
        pawns_must_play["blue"] = []  
        pawns_must_play["orange"] = []  

        # Si le dernier mouvement n'est pas un pion, ignorer  
        if lastmove[1] == "Pawn":  
            return False  

        # Déterminer les pions qui doivent jouer en fonction du dernier mouvement  
        if lastmove[0] == "blue":  
            self._check_retraite_for_color("orange", 0, lastmove)  
        else:  
            self._check_retraite_for_color("blue", 4, lastmove)  

        # Limiter à un seul pion par couleur  
        for color in ["blue", "orange"]:  
            if len(pawns_must_play[color]) > 1:  
                pawns_must_play[color] = [pawns_must_play[color][-1]]  

        # Retourner True si au moins un pion doit jouer  
        return bool(pawns_must_play["blue"] or pawns_must_play["orange"])  

    def _check_retraite_for_color(self, color, y_pos, lastmove):  
        """  
        Vérifie les pions d'une couleur donnée pour la retraite.  
        
        Args:  
            color: Couleur des pions à vérifier  
            y_pos: Position y de la ligne de retraite  
            lastmove: Dernier mouvement effectué  
        """  
        for pawn in self.pawns:  
            if (pawn.color == color and  
                pawn.y == y_pos and  
                int(lastmove[3]) == y_pos and  
                int(lastmove[2]) == pawn.x and  
                pawn.type < int(float(lastmove[1]))):  
                pawns_must_play[color].append(pawn)  

    def simulate_move(self, color, type, x, y):  
        """  
        Simule un mouvement pour l'IA.  
        
        Args:  
            color: Couleur du pion à déplacer  
            type: Type du pion à déplacer  
            x: Coordonnée x de destination  
            y: Coordonnée y de destination  
            
        Returns:  
            list: Résultat du mouvement [mouvement_réussi, victoire]  
        """  
        for pawn in self.pawns:  
            if pawn.color == color and pawn.type == type:  
                return pawn.move(x, y, self.grid, self.pawns, self, simulate=True)  
        return [False, False]  

    # ----  
    # Évaluation (pour l'IA)  
    # ----  

    def get_color_bottom(self, x, y, stack_value):  
        """  
        Retourne la couleur du pion à la base d'une pile.  
        
        Args:  
            x: Coordonnée x  
            y: Coordonnée y  
            stack_value: Valeur du pion dans la pile  
            
        Returns:  
            str: Couleur du pion ("blue" ou "orange") ou None si non trouvé  
        """  
        for pawn in self.pawns:  
            if pawn.x == x and pawn.y == y and pawn.type == stack_value:  
                return pawn.color  
        return None  

    def evaluateClassic(self, color):  
        """  
        Évalue le plateau selon la stratégie classique.  
        
        Args:  
            color: Couleur du joueur pour lequel évaluer  
            
        Returns:  
            int: Score d'évaluation  
        """  
        score = 0  

        # Parcourir toutes les piles du plateau  
        for x in range(self.grid.size):  
            for y in range(self.grid.size):  
                stack = self.grid.grid[y][x]  
                if len(stack) > 2:  
                    # Obtenir les couleurs des pions dans la pile  
                    base_pawn_color = self.get_color_bottom(x, y, stack[0])  
                    second_pawn_color = self.get_color_bottom(x, y, stack[1])  
                    third_pawn_color = self.get_color_bottom(x, y, stack[2])  

                    # Calculer le score de la pile  
                    score += self._calculate_stack_score(  
                        stack, base_pawn_color, second_pawn_color, third_pawn_color,  
                        color, STACK_MULTIPLIER, BONUS_COLOR, THREAT_MULTIPLIER  
                    )  

        return score  

    def _calculate_stack_score(self, stack, base_color, second_color, third_color, player_color, stack_multiplier, bonus_color, threat_multiplier):  
        """  
        Calcule le score d'une pile en fonction de sa composition.  
        
        Args:  
            stack: Pile de pions  
            base_color: Couleur du pion à la base  
            second_color: Couleur du deuxième pion  
            third_color: Couleur du troisième pion  
            player_color: Couleur du joueur pour lequel évaluer  
            stack_multiplier: Multiplicateur pour les piles  
            bonus_color: Bonus pour les piles de même couleur  
            threat_multiplier: Multiplicateur pour les menaces  
            
        Returns:  
            int: Score de la pile  
        """  
        stack_values = list(stack)  
        stack_score = 0  

        # Vérifier les configurations gagnantes et importantes  
        if stack_values == [4, 3, 2, 1]:  
            return WINNING_STACK_BONUS if base_color == player_color else -WINNING_STACK_BONUS  

        # Autres configurations importantes  
        score_configs = {  
            (4, 3, 2): (10, 10),  
            (4, 3, 1): (5, 6),  
            (4, 3): (2, 3),  
            (4, 2, 1): (1, 5),  
            (4, 2): (2, 3),  
            (4, 1): (-1, 1),  
            (3, 2, 1): (5, 5),  
            (3, 2): (2, 2),  
            (3, 1): (1, 6),  
            (2, 1): (1, 1)  
        }  

        # Calculer le score en fonction de la configuration  
        tuple_stack = tuple(stack_values)  
        if tuple_stack in score_configs:  
            pos_mult, neg_mult = score_configs[tuple_stack]  

            if base_color == player_color:  
                stack_score += (stack_multiplier * pos_mult + bonus_color)  
            else:  
                # Cas spéciaux pour certaines configurations  
                if tuple_stack == (4, 3, 1) and second_color != player_color and third_color == player_color:  
                    stack_score += (stack_multiplier * pos_mult + bonus_color)  
                elif tuple_stack in [(4, 2, 1), (4, 2), (4, 1), (3, 1)] and second_color == player_color:  
                    stack_score += (stack_multiplier * pos_mult + bonus_color)  
                else:  
                    stack_score -= (stack_multiplier * neg_mult + threat_multiplier)  

        return stack_score  

    def evaluateCenter(self, color):  
        """  
        Évalue le plateau en donnant plus d'importance aux positions centrales.  
        
        Args:  
            color: Couleur du joueur pour lequel évaluer  
            
        Returns:  
            int: Score d'évaluation  
        """  
        score = 0  
        central_positions = [(2, 2), (2, 1), (2, 3), (1, 2), (3, 2)]  
        diagonal_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]  
        central_position = (2, 2)  

        # Poids pour les différentes positions  
        weights = {  
            central_position: 1.5,  
            "adjacent": 1.2,  
            "diagonal": 1.1  
        }  

        # Parcourir toutes les piles du plateau  
        for x in range(self.grid.size):  
            for y in range(self.grid.size):  
                stack = self.grid.grid[y][x]  
                if len(stack) > 2:  
                    # Obtenir les couleurs des pions dans la pile  
                    base_pawn_color = self.get_color_bottom(x, y, stack[0])  
                    second_pawn_color = self.get_color_bottom(x, y, stack[1])  
                    third_pawn_color = self.get_color_bottom(x, y, stack[2])  

                    # Calculer le score de base de la pile  
                    stack_score = self._calculate_stack_score(  
                        stack, base_pawn_color, second_pawn_color, third_pawn_color,  
                        color, 100, 50, 500  
                    )  

                    # Appliquer les poids en fonction de la position  
                    if (x, y) == central_position:  
                        stack_score *= weights[central_position]  
                    elif (x, y) in central_positions:  
                        stack_score *= weights["adjacent"]  
                    elif (x, y) in diagonal_positions:  
                        stack_score *= weights["diagonal"]  

                    score += stack_score  

        return score  

    def evaluateRush(self, color):  
        """  
        Évalue le plateau en favorisant la construction rapide de piles.  
        
        Args:  
            color: Couleur du joueur pour lequel évaluer  
            
        Returns:  
            int: Score d'évaluation  
        """  
        score = 0  

        # Parcourir toutes les piles du plateau  
        for x in range(self.grid.size):  
            for y in range(self.grid.size):  
                stack = self.grid.grid[y][x]  
                if not stack or stack[0] == 0:  
                    continue  

                stack_values = list(stack)  
                base_pawn_color = self.get_color_bottom(x, y, stack[0])  

                # Vérifier si la pile appartient au joueur  
                if base_pawn_color == color:  
                    # Attribuer des scores en fonction de la progression vers la pile gagnante  
                    if len(stack) == 1 and stack_values[0] == 4:  
                        score += STACK_MULTIPLIER * 1  
                    elif len(stack) == 2 and stack_values == [4, 3]:  
                        score += STACK_MULTIPLIER * 2  
                    elif len(stack) == 3 and stack_values == [4, 3, 2]:  
                        score += STACK_MULTIPLIER * 3  
                    elif len(stack) == 4 and stack_values == [4, 3, 2, 1]:  
                        score += WINNING_STACK_BONUS  

        return score  

    def evaluateBlock(self, color):  
        """  
        Évalue le plateau en favorisant le blocage des pions adverses.  
        
        Args:  
            color: Couleur du joueur pour lequel évaluer  
            
        Returns:  
            int: Score d'évaluation  
        """  
        score = 0  
        blockally = 5000  
        blockenemy = 15000  
        blockenemy2 = 10000  
        opponent_color = "blue" if color == "orange" else "orange"  

        # Parcourir toutes les piles du plateau  
        for x in range(self.grid.size):  
            for y in range(self.grid.size):  
                stack = self.grid.grid[y][x]  
                if len(stack) <= 1 or stack[0] == 0:  
                    continue  

                base_pawn_color = self.get_color_bottom(x, y, stack[0])  

                # Vérifier les configurations de blocage  
                if stack[0] == 4:  
                    if len(stack) > 1:  
                        if stack[1] == 2 and base_pawn_color == color:  
                            score += blockally  
                        elif stack[1] == 1 and base_pawn_color == opponent_color:  
                            score += blockenemy  

                    if len(stack) > 2 and stack[1] == 3:  
                        if stack[2] == 1 and base_pawn_color == opponent_color:  
                            score += blockenemy2  

        return score  

#Fonction utilitaire pour obtenir l'entrée utilisateur

def get_user_input(message, valid_inputs):
    """
    Demande à l'utilisateur une entrée valide.

    Args:  
        message: Message à afficher  
        valid_inputs: Liste des entrées valides  
        
    Returns:  
        str: Entrée utilisateur valide  
    """  
    user_input = None  
    while user_input not in valid_inputs:  
        user_input = input(message)  
        if user_input not in valid_inputs:  
            print("Invalid input")  
    return user_input