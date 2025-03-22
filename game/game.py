from game.pawn import Pawn
from game.grid import Grid
from game.mouvement import Mouvement
import numpy as np
from game.env_var import *
import copy


def get_user_input(message, valid_inputs):
    user_input = None
    while user_input not in valid_inputs:
        user_input = input(message)
        if user_input not in valid_inputs:
            print("Invalid input")
    return user_input


class Game:
    def __init__(self, data_manager=None, manual_mode=False, use_ai=False, ai_types=None):
        # Data manager instance
        self.data_manager = data_manager
        # List of pawns
        self.pawns = []
        self.num_retreat = 0

        self.mode = manual_mode

        # custom parameters for the game
        self.use_ai = use_ai
        self.ai_types = ai_types if ai_types else (0, 0)

        if manual_mode:
            manual_placement = get_user_input("\nDo you want to place the pawns manually ? (y/n): ",
                                              ["y", "Y", "n", "N"])
            if use_ai:
                ai_type_1 = get_user_input("\nType of AI 1 (1:MinMax or 2:Random): ", ["1", "2"])
                ai_type_2 = get_user_input("\nType of AI 2 (1:MinMax or 2:Random): ", ["1", "2"])
                self.ai_types = (int(ai_type_1), int(ai_type_2))

            print("AI types:", self.ai_types)
            print("Manual placement:", manual_placement)

            if manual_placement.lower() == "n":
                self.init_pawns("blue")
                self.init_pawns("orange")
            else:
                self.init_pawns_manually()
        else:
            if not ai_types:
                raise ValueError("AI types must be specified when manual mode is False")
            
            self.init_infinite()

        # Init the grid and display it
        #self.initializing = True
        #self.init_ai()

        # Init the grid and display it
        self.grid = Grid(5, self.pawns)
        #self.grid.grid[4][4] = np.array([0])
        self.grid.display()
    
    # Reset the game
    def reset(self):
        self.pawns = []
        self.num_retreat = 0
        self.init_infinite()
        self.grid = Grid(5, self.pawns)
        self.grid.display()

    # Init the AI pawns
    def init_infinite(self):
        self.init_pawns("blue")
        self.init_pawns("orange")
        self.use_ai = True

    # Init the pawns manually
    def init_pawns_manually(self):

        usedmouvs = []
        mouvs = []
        allinputsb = []
        allinputso = []

        for key in basic_mouvements.keys():
            mouvs.append(key)
        color = "blue"
        i = 0
        y = 0
        counter = 0
        # Place eaxh pawn manually
        while i < 5:

            if counter % 2 == 0:
                print("Blue is placing ")
                color = "blue"
                y = 0
            else:
                print("Orange is placing ")
                color = "orange"
                y = 4

            try:
                print("Placing", color, "'s ", i + 1)
                print("Used mouvs:", usedmouvs)
                x = int(input("x:") or 1.)

                if color == "blue":
                    if x not in allinputsb and x < 5:
                        pawn = Pawn(x, y, i + 1, mouvs[i], color)
                        self.pawns.append(pawn)
                        allinputsb.append(x)
                        if counter % 2 != 0:
                            i += 1
                        counter += 1
                elif color == "orange":
                    if x not in allinputso and x < 5:
                        pawn = Pawn(x, y, i + 1, mouvs[i], color)
                        self.pawns.append(pawn)
                        allinputso.append(x)
                        if counter % 2 != 0:
                            i += 1
                        counter += 1

                else:
                    print("Invalid input")
                    continue
            except:
                print("Invalid input")
                continue

            # Change the color of the player
            if i == 4 and color == "blue":
                color = "orange"
                i = 0
                y = 4
                allinputs = []
            elif i == 4 and color == "orange":
                break

        print("All pawns placed")
        print("-----Starting game-----")

    # Place the pawns automatically
    def init_pawns(self, color):
        mouvs = []
        for key in basic_mouvements.keys():
            mouvs.append(key)

        usedpos = []
        usedmouvs = []
        for i in range(4):
            nextpos = np.random.randint(0, 5)
            if nextpos in usedpos:
                while nextpos in usedpos:
                    nextpos = np.random.randint(0, 5)
                usedpos.append(nextpos)
            else:
                usedpos.append(nextpos)

            nextmouv = np.random.randint(0, 4)
            if mouvs[nextmouv] in usedmouvs:
                while mouvs[nextmouv] in usedmouvs:
                    nextmouv = np.random.randint(0, 4)
                usedmouvs.append(mouvs[nextmouv])
            else:
                usedmouvs.append(mouvs[nextmouv])

            if color == "blue":
                pawn = Pawn(nextpos, 0, i + 1, mouvs[i], color)
                self.data_manager.set_initial_pos(color, i + 1, (nextpos, 0))
                self.pawns.append(pawn)
            else:
                pawn = Pawn(nextpos, 4, i + 1, mouvs[i], color)
                self.data_manager.set_initial_pos(color, i + 1, (nextpos, 4))
                self.pawns.append(pawn)

    # Check if a pawn is in the retraite area and add it to the list of pawns that must be played
    def isretraite(self, lastmove):
        pawns_must_play["blue"] = []
        pawns_must_play["orange"] = []
        if lastmove[1] != "Pawn":
            if lastmove[0] == "blue":
                for pawn in self.pawns:
                    if pawn.color == "orange" and pawn.y == 0 and int(lastmove[3]) == 0 and int(lastmove[2]) == int(
                            pawn.x) and pawn.type < int(float(lastmove[1])):
                        pawns_must_play["orange"].append(pawn)

            else:
                for pawn in self.pawns:
                    if pawn.color == "blue" and pawn.y == 4 and int(lastmove[3]) == 4 and int(lastmove[2]) == int(
                            pawn.x) and pawn.type < int(float(lastmove[1])):
                        pawns_must_play["blue"].append(pawn)

            for key in pawns_must_play.keys():
                if len(pawns_must_play[key]) > 1:
                    while len(pawns_must_play[key]) >= 2:
                        pawns_must_play[key].pop(0)
            if pawns_must_play["blue"] != [] or pawns_must_play["orange"] != []:
                return True
            else:
                return False

    # Get all the next moves available for a player
    def all_next_moves(self, color):
        next_moves = []
        for pawn in self.pawns:
            for x in range(5):
                for y in range(5):
                    if self.initializing == True and self.grid.grid[y][x] == 0:
                        if y == 0 and pawn.color == color and pawn.x == -1:
                            if [pawn.color, pawn.type, x, 0] not in next_moves:
                                next_moves.append([pawn.color, pawn.type, x, 0])
                        elif y == 4 and pawn.color == color and pawn.x == -1:
                            if [pawn.color, pawn.type, x, 4] not in next_moves:
                                next_moves.append([pawn.color, pawn.type, x, 4])
                    if Mouvement.legit_mouv(pawn, pawn, x, y, self.grid) and pawn.x != x and pawn.y != y and pawn.color == color and self.initializing == False:
                        if pawns_must_play[color] == []:
                            next_moves.append([pawn.color, pawn.type, x, y])
                        else:
                            if pawn in pawns_must_play[color]:
                                next_moves.append([pawn.color, pawn.type, x, y])
        return next_moves

    # Function to simulate a move for the AI
    def simulate_move(self, color, type, x, y):
        for pawn in self.pawns:
            if pawn.color == color and pawn.type == type:
                ispawnmoved = pawn.move(x, y, self.grid, self.pawns, self, simulate=True)

        return ispawnmoved

    # Return the color of the pawn at the bottom of the stack
    def get_color_bottom(self, x, y, stack_value):
        for pawn in self.pawns:
            if pawn.x == x and pawn.y == y and pawn.type == stack_value:
                return pawn.color
        return None

    def evaluateClassic(self, color):
        score = 0
        stack_multiplier = 2000  # Multiplicateur de score pour les piles
        bonus_color = 500  # Bonus pour chaque pion de la couleur du joueur dans une pile
        threat_multiplier = 500  # Malus pour les piles avantageuses de l'adversaire
        opponent_color = "blue" if color == "orange" else "orange"

        # Parcourir toutes les piles du plateau
        for x in range(self.grid.size):
            for y in range(self.grid.size):
                stack = self.grid.grid[y][x]
                stack_size = len(stack)
                if stack_size > 2:
                    base_pawn_color = self.get_color_bottom(x, y, stack[0])
                    secondPawn_color = self.get_color_bottom(x, y, stack[1])
                    thirdPawn_color = self.get_color_bottom(x, y, stack[2])
                    # Attribuer des points pour les configurations avantageuses
                    score += self.calculate_stack_scoreClassic(stack, base_pawn_color, secondPawn_color, thirdPawn_color, color, stack_multiplier, bonus_color, threat_multiplier)

        return score

    def calculate_stack_scoreClassic(self, stack, base_pawn_color, secondPawn_color, thirdPawn_color, color, stack_multiplier, bonus_color, threat_multiplier):
        stack_score = 0
        bonus_color = 150
        winning_stack_bonus = 100000000  # Bonus pour une pile gagnante de 4-3-2-1
        
        stack_values = [pawn for pawn in stack]

        # Calcul du score en fonction de la séquence de chiffres dans la pile 
        if stack_values == [4, 3, 2, 1]:
            if base_pawn_color == color:
                stack_score += winning_stack_bonus
            else:
                stack_score -= winning_stack_bonus
        if stack_values == [4, 3, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 10 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 10 + threat_multiplier)
        if stack_values == [4, 3, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 5 + bonus_color)
            else:
                if secondPawn_color != color and thirdPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 6 + threat_multiplier)
        if stack_values == [4, 3]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 3 + threat_multiplier)
        if stack_values == [4, 2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 5 + threat_multiplier)
        if stack_values == [4, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 3 + threat_multiplier)
        if stack_values == [4, 1]:
            if base_pawn_color == color:
                stack_score -= (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier + threat_multiplier)
        if stack_values == [3, 2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 5 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 5 + threat_multiplier)
        if stack_values == [3, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 2 + threat_multiplier)
        if stack_values == [3, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 6 + threat_multiplier)
        if stack_values == [2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                stack_score -= (stack_multiplier + threat_multiplier)

        return stack_score
    
    def evaluateCenter(self, color):
        score = 0
        stack_multiplier = 100
        bonus_color = 50
        threat_multiplier = 500
        central_positions = [(2, 2), (2, 1), (2, 3), (1, 2), (3, 2)]  # Positions centrales pour la pièce 3
        diagonal_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]  # Positions diagonales pour la pièce 2
        central_position = (2, 2)  # Position centrale pour la pièce 1

        # Poids attribués aux différentes cases en fonction de leur importance stratégique
        weight_central = 1.5  # La case centrale a un poids 1.5 fois plus important
        weight_adjacent = 1.2  # Les cases adjacentes ont un poids 1.2 fois plus important
        weight_diagonal = 1.1  # Les cases diagonales ont un poids 1.1 fois plus important

        # Parcourir toutes les piles du plateau
        for x in range(self.grid.size):
            for y in range(self.grid.size):
                stack = self.grid.grid[y][x]
                stack_size = len(stack)
                if stack_size > 2:
                    base_pawn_color = self.get_color_bottom(x, y, stack[0])
                    secondPawn_color = self.get_color_bottom(x, y, stack[1])
                    thirdPawn_color = self.get_color_bottom(x, y, stack[2])
                    
                    # Calcul du score de la pile
                    stack_score = self.calculate_stack_scoreCenter(stack, base_pawn_color, secondPawn_color, thirdPawn_color, color, stack_multiplier, bonus_color, threat_multiplier)
                    
                    # Appliquer les poids aux positions centrales et diagonales
                    if (x, y) == central_position:
                        stack_score *= weight_central
                    elif (x, y) in central_positions:
                        stack_score *= weight_adjacent
                    elif (x, y) in diagonal_positions:
                        stack_score *= weight_diagonal

                    # Ajouter le score ajusté au score total
                    score += stack_score

        return score


    def calculate_stack_scoreCenter(stack, base_pawn_color, secondPawn_color, thirdPawn_color, color, stack_multiplier, bonus_color, threat_multiplier):
        stack_score = 0
        bonus_color = 150
        winning_stack_bonus = 100000000  # Bonus pour une pile gagnante de 4-3-2-1
        
        stack_values = [pawn for pawn in stack]

        # Calcul du score en fonction de la séquence de chiffres dans la pile 
        if stack_values == [4, 3, 2, 1]:
            if base_pawn_color == color:
                stack_score += winning_stack_bonus
            else:
                stack_score -= winning_stack_bonus
        if stack_values == [4, 3, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 10 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 10 + threat_multiplier)
        if stack_values == [4, 3, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 5 + bonus_color)
            else:
                if secondPawn_color != color and thirdPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 6 + threat_multiplier)
        if stack_values == [4, 3]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 3 + threat_multiplier)
        if stack_values == [4, 2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 5 + threat_multiplier)
        if stack_values == [4, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 3 + threat_multiplier)
        if stack_values == [4, 1]:
            if base_pawn_color == color:
                stack_score -= (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier + threat_multiplier)
        if stack_values == [3, 2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 5 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 5 + threat_multiplier)
        if stack_values == [3, 2]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier * 2 + bonus_color)
            else:
                stack_score -= (stack_multiplier * 2 + threat_multiplier)
        if stack_values == [3, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                if secondPawn_color == color:
                    stack_score += (stack_multiplier * 5 + bonus_color)
                else:
                    stack_score -= (stack_multiplier * 6 + threat_multiplier)
        if stack_values == [2, 1]:
            if base_pawn_color == color:
                stack_score += (stack_multiplier + bonus_color)
            else:
                stack_score -= (stack_multiplier + threat_multiplier)

        return stack_score
    
    def evaluateRush(self, color):
        score = 0
        stack_multiplier = 2000  # Score multiplier for stacks
        winning_stack_bonus = 100000000  # Bonus for a winning stack of 4-3-2-1


        # Traverse all cells of the grid
        for x in range(self.grid.size):
            for y in range(self.grid.size):
                stack = self.grid.grid[y][x]  # Get the stack at this cell
                stack_values = [pawn for pawn in stack]
                base_pawn_color = self.get_color_bottom(x, y, stack[0])
                if base_pawn_color == color:
                    if len(stack) == 1 and stack_values == 4:
                        score += stack_multiplier * 1  # Priority to start the stack with a 4
                    elif len(stack) == 2 and stack_values == [4, 3]:
                        score += stack_multiplier * 2  # Higher priority to add a 3 on top of a 4
                    elif len(stack) == 3 and stack_values == [4, 3, 2]:
                        score += stack_multiplier * 3  # Even higher priority to add a 2 on top of a 4-3
                    elif len(stack) == 4 and stack_values == [4, 3, 2, 1]:
                        score += winning_stack_bonus  # Maximum priority and bonus for completing the stack 4-3-2-1

        return score
    
    def evaluateBlock(self, color):
        score = 0
        blockally = 5000  
        blockenemy = 15000  
        blockenemy2 = 10000
        opponent_color = "blue" if color == "orange" else "orange"

        for x in range(self.grid.size):
            for y in range(self.grid.size):
                stack = self.grid.grid[y][x]
                if len(stack) > 0:
                    base_pawn_color = self.get_color_bottom(x, y, stack[0])
                    if len(stack) > 1:
                        top_pawn = stack[1]

                        if stack[0] == 4:
                            if top_pawn == 2 and base_pawn_color == color:
                                score += blockally
                            elif top_pawn == 1 and base_pawn_color == opponent_color:
                                score += blockenemy
                        if stack[0] == 4 and stack[1] == 3:
                            top_pawn = stack[-1]

                            if top_pawn == 1 and base_pawn_color == opponent_color:
                                score += blockenemy2

        return score
   
    

                    
