import os
from data.data_manager import DataManager
from game.env_var import *
from game.game import Game
from ai.Minimax import Minimax
from ai.dummyAI import Dummyai
import csv
from datetime import datetime

# Parameters
manual_mode = True     # True if we want to place the pawns manually, False if we want to place them randomly
use_ai = True           # True if we want to use AI, False if we want to play manually
ai_types = (1, 1)       # 1 for Minimax, 2 for random AI


def main():
    # Initialize the data manager
    # Note: if you want to overwrite an existing CSV file, set the overwrite parameter to True
    # and the url to the path of the file to overwrite
    data_manager = DataManager(False, "./data/dataset/game_RR.csv")
    # Initialize the game
    # Note: when manual_mode is False and use_ai is True, we need to set the ai_types to 1 for
    # Minimax and 2 for random AI
    game = Game(data_manager, manual_mode, use_ai, ai_types)

    # We get the actual time
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # How many times we want to run the game
    loop = 1
    while loop > 0:
        if loop < 500:
            game.reset()
        print(loop)
        loop -= 1

        is_ai = game.use_ai
        counter = 0
        color = ""
        ispawnmoved = False

        if game.use_ai:
            if game.ai_types[0] == 1 and game.ai_types[1] == 1:
                aiblue = Minimax("blue", game)
                aiorange = Minimax("orange", game)
            elif game.ai_types[0] == 1 and game.ai_types[1] == 2:
                aiblue = Minimax("blue", game)
                aiorange = Dummyai("orange")
            elif game.ai_types[0] == 2 and game.ai_types[1] == 1:
                aiblue = Dummyai("blue")
                aiorange = Minimax("orange", game)
            elif game.ai_types[0] == 2 and game.ai_types[1] == 2:
                aiblue = Dummyai("blue")
                aiorange = Dummyai("orange")
            else:
                aiblue = Minimax("blue", game)
                aiorange = Minimax("orange", game)

        # Create a new CSV file for recording moves
        path = "./CSV/"
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        move_log_filename = f"{path}game_moves_{time}.csv"
        with open(move_log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Color", "Pawn", "X", "Y", "Turn"])  # Write header row

        # Main loop
        while True:
            counterinit = 0
            if game.grid.isbroken:
                print("game is broken")
                print("Game ended in", counter, "turns")
                for pawn in game.pawns:
                    pawn.display()
                break
            with open(move_log_filename, mode='r', newline='') as file:
                last_move = list(csv.reader(file))[-1]

            if game.isretraite(last_move):
                game.num_retreat += 1
            
            # Check if the game is still initializing
            for pawn in game.pawns:
                if pawn.color == "blue" and pawn.y == 0:
                    counterinit += 1
                elif pawn.color == "orange" and pawn.y == 4:
                    counterinit += 1
                    
            if counterinit == 8:
                print("finish initialisation of the game")
                game.initializing = False
            if game.initializing == False:
                game.grid.checkgrid(counter)

            # Counter for the turns and the color of the player
            if counter % 2 == 0:
                #print("Blue turn | turn ", counter)
                color = "blue"
            else:
                #print("Orange turn | turn ", counter)
                color = "orange"

            game.all_next_moves(color)
            move = None
            if game.use_ai:
                ai = aiblue if color == "blue" else aiorange
                if ai.type == "M":
                    print("MINMAX AI")
                    move = ai.playsmart()  # Obtenir le meilleur mouvement de l'IA
                    if move:
                        if move[1] == -1 : # or counter > 200:
                            os.remove(move_log_filename)
                            loop += 1
                            break
                        print(f"AI {color} chooses to move pawn {move[1]} to ({move[2]}, {move[3]})")
                        # On met à jour l'historique du pion
                        data_manager.update_pawn_history(color, move[1], (move[2], move[3]), counter)
                else:
                    print("RANDOM AI")
                    move = ai.playrandom(game.all_next_moves(color))
                    # if move is None:
                    #     os.remove(move_log_filename)
                    #     loop += 1
                    #     break
                    # else:
                    try:
                        pawn_to_move = move[1]
                        x = move[2]
                        y = move[3]
                        print(f"RANDOMAI {color} chooses to move pawn {pawn_to_move} to ({x}, {y})")
                        # On met à jour l'historique du pion
                        data_manager.update_pawn_history(color, move[1], (move[2], move[3]), counter)
                    except:
                        print("No moves available")
                        game.grid.display()
                        loop = 0
                        break
            else:
                try:
                    pawn_to_move = float(input("Select pawn:") or 1.)
                    x = int(input("x:") or 1.)
                    y = int(input("y:") or 1.)
                    move = [color, pawn_to_move, x, y]
                except:
                    print("Invalid input")
                    continue

            if move:

                pawn_to_move, x, y = move[1], move[2], move[3]
                if pawn_to_move == -1:
                    # stocker l'état de la grille car situation bloquante pour un joueur
                    break
                # Vérifier si le pion à déplacer est dans la liste des pions qui doivent jouer
                for pawn in game.pawns:
                    if pawn.type == pawn_to_move and pawn.color == color:
                        if pawns_must_play[color] == []:
                            ispawnmoved = pawn.move(x, y, game.grid, game.pawns, game)
                        else:
                            if pawn in pawns_must_play[color]:
                                ispawnmoved = pawn.move(x, y, game.grid, game.pawns, game)
                                pawns_must_play[color].remove(pawn)
                            else:
                                #print(
                                #    "You must play with the pawn(s) that is in the retraite area. \nPawns in the retraite area:",
                                #    ' and '.join([str(lst.type) for lst in pawns_must_play[color]]))
                                ispawnmoved = [False, False]
                if ispawnmoved[0]:
                    with open(move_log_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([color, pawn_to_move, x, y, counter])  # Write move to CSV file
                # Check if the game is won by a player
                if ispawnmoved[1]:
                    print("Game Over")
                    print("And the winner is......", color, "!!!")
                    print("Game ended in", counter, "turns")
                    final_stack = game.grid.getfinalstack(x, y)
                    ai = [
                        {"type": aiblue.type, "depth": aiblue.base_depth if aiblue.type == "M" else None,
                         "color": "BLUE"},
                        {"type": aiorange.type, "depth": aiorange.base_depth if aiorange.type == "M" else None,
                         "color": "ORANGE"}
                    ]

                    data_manager.write(ai, color, counter, game.num_retreat, final_stack)
                    game.winner = color

                    os.remove(move_log_filename)

                    break

                # Increment the counter if the player played a valid move
                if ispawnmoved[0]:
                    counter += 1

    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration = datetime.strptime(end_time, "%Y%m%d_%H%M%S") - datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    print("Duration:", duration)


if __name__ == "__main__":
    main()
