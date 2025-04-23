import os
from data.data_manager import DataManager
from game.env_var import *
from game.game import Game
from ai.Minimax import Minimax
from ai.dummyAI import Dummyai
from ai.sb3_agent import DQN_sb3_Agent
import csv
from datetime import datetime

# Parameters
MANUAL_MODE = False # True pour placer les pions manuellement, False pour placement aléatoire
USE_AI = True # True pour utiliser l'IA, False pour jouer manuellement
AI_TYPES = (3, 1) # Types d'IA: 1 pour Minimax, 2 pour IA aléatoire, 3 pour SB3 DQN
MAX_TURNS = 1000 # Nombre maximum de tours avant d'arrêter la partie 
MAX_GAMES = 1 # Nombre de parties à jouer

def main():
    """
    Fonction principale du programme.
    Initialise le jeu et gère la boucle principale.
    """
    global MANUAL_MODE, USE_AI, AI_TYPES, MAX_TURNS, MAX_GAMES

    # Initialisation du gestionnaire de données
    # Note: pour écraser un fichier CSV existant, définir overwrite=True
    data_manager = DataManager(False, "./data/dataset/game_RR.csv")

    # Initialisation du jeu
    print("Initialisation du jeu...")
    print("Veuillez choisir le type d'IA, 1 pour Minimax, 2 pour IA aléatoire, 3 pour SB3 DQN")
    AI_TYPES = list(AI_TYPES)  # Convertir le tuple en liste pour le modifier
    AI_TYPES[0] = int(get_user_input("Type d'IA bleue: ", ["1", "2", "3"]))
    AI_TYPES[1] = int(get_user_input("Type d'IA orange: ", ["1", "2", "3"]))
    AI_TYPES = tuple(AI_TYPES)  # Reconvertir la liste en tuple après modification
    print(f"IA bleue: {AI_TYPES[0]}, IA orange: {AI_TYPES[1]}")
    
    game = Game(data_manager, MANUAL_MODE, USE_AI, AI_TYPES)

    # Enregistrement du temps de début  
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  

    # Boucle pour jouer plusieurs parties  
    games_remaining = MAX_GAMES  
    while games_remaining > 0:  
        if games_remaining < 500:  # Protection contre les boucles infinies  
            game.reset()  
            print(f"Partie {MAX_GAMES - games_remaining + 1}/{MAX_GAMES}")  
            games_remaining -= 1  

        # Initialisation des variables de jeu  
        counter = 0  # Compteur de tours  
        color = ""   # Couleur du joueur actuel  
        
        # Initialisation des IA si nécessaire  
        if game.use_ai:
            print("Initialisation des IA..."), print()
            aiblue, aiorange = initialize_ai(game)  

        # Création d'un fichier CSV pour enregistrer les mouvements  
        move_log_filename = create_move_log_file()  

        # Boucle principale de jeu  
        game_result = play_game(game, aiblue, aiorange, data_manager, move_log_filename, counter)  
        
        # Si la partie est terminée normalement, supprimer le fichier de log temporaire  
        if game_result == "completed" and os.path.exists(move_log_filename):  
            os.remove(move_log_filename)  

    # Affichage de la durée totale  
    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
    duration = datetime.strptime(end_time, "%Y%m%d_%H%M%S") - datetime.strptime(start_time, "%Y%m%d_%H%M%S")  
    print(f"Durée totale: {duration}")  

def initialize_ai(game):
    """
    Initialise les IA en fonction des types spécifiés.

    Args:  
        game: Instance du jeu  
        
    Returns:  
        tuple: (IA bleue, IA orange)  
    """  
    ai_type_blue, ai_type_orange = game.ai_types  
    
    # Initialisation de l'IA bleue  
    if ai_type_blue == 1:
        print("L'IA bleue est Minimax"), print()
        aiblue = Minimax("blue", game)  
    elif ai_type_blue == 2:
        print("L'IA bleue est DummyAI"), print()
        aiblue = Dummyai("blue")
    elif ai_type_blue == 3:
        print("L'IA bleue est DQN"), print()
        aiblue = DQN_sb3_Agent("blue", game)
    else:
        print("Type d'IA non reconnu, utilisation de Minimax par défaut /n")
        aiblue = Minimax("blue", game)  # Par défaut  
    
    # Initialisation de l'IA orange  
    if ai_type_orange == 1:
        print("L'IA orange est Minimax"), print()
        aiorange = Minimax("orange", game)  
    elif ai_type_orange == 2:
        print("L'IA orange est DummyAI"), print()
        aiorange = Dummyai("orange")
    elif ai_type_orange == 3:
        print("L'IA orange est DQN"), print()
        aiorange = DQN_sb3_Agent("orange", game)
    else:
        print("Type d'IA non reconnu, utilisation de Minimax par défaut /n")  
        aiorange = Minimax("orange", game)  # Par défaut  
        
    return aiblue, aiorange  

def create_move_log_file():
    """
    Crée un fichier CSV pour enregistrer les mouvements.

    Returns:  
        str: Chemin du fichier créé  
    """  
    path = "./CSV/"  
    os.makedirs(path, exist_ok=True)  # Crée le répertoire s'il n'existe pas  
    
    time = datetime.now().strftime('%Y%m%d_%H%M%S')  
    move_log_filename = f"{path}game_moves_{time}.csv"  
    
    with open(move_log_filename, mode='w', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(["Color", "Pawn", "X", "Y", "Turn"])  # En-tête  
        
    return move_log_filename  

def play_game(game, aiblue, aiorange, data_manager, move_log_filename, counter=0):
    """
    Exécute une partie complète.

    Args:
        game: Instance du jeu
        aiblue: IA pour le joueur bleu
        aiorange: IA pour le joueur orange
        data_manager: Gestionnaire de données
        move_log_filename: Chemin du fichier de log des mouvements
        counter: Compteur de tours initial

    Returns:
        str: Résultat de la partie ("completed", "broken", "timeout")
    """
    while counter < MAX_TURNS:
        # Vérification de l'intégrité de la grille
        if game.grid.isbroken:
            print("La grille est dans un état invalide")
            print(f"Partie terminée en {counter} tours")
            for pawn in game.pawns:
                pawn.display()
            return "broken"

        # Lecture du dernier mouvement pour vérifier la retraite
        try:
            with open(move_log_filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                rows = list(reader)
                if len(rows) > 1:  # S'assurer qu'il y a au moins une ligne après l'en-tête
                    last_move = rows[-1]
                    if game.isretraite(last_move):
                        game.num_retreat += 1
        except (IndexError, FileNotFoundError):
            pass  # Pas de mouvement précédent

        # Vérification de la phase d'initialisation
        check_initialization_phase(game)

        # Vérification de l'intégrité de la grille si le jeu n'est plus en phase d'initialisation
        if not game.initializing:
            game.grid.checkgrid(counter)

        # Détermination du joueur actuel
        color = "blue" if counter % 2 == 0 else "orange"
        print(f"Tour {counter}: {color}")

        # Obtention des mouvements possibles
        possible_moves = game.all_next_moves(color)

        if not possible_moves:
            print(f"Aucun mouvement possible pour {color}")
            return "timeout"

        # Obtention du mouvement à jouer
        move = get_next_move(game, color, aiblue, aiorange, possible_moves)

        # Traitement du mouvement
        if move:
            # Cas spécial: pas de mouvement possible
            if move[1] == -1:
                return "timeout"

            # Exécution du mouvement
            result = execute_move(game, move, color, aiblue, aiorange, data_manager, counter, move_log_filename)

            # Vérification de la victoire
            if result == "victory":
                return "completed"

            # Incrémentation du compteur si le mouvement est valide
            counter += 1
        else:
            print(f"Aucun mouvement valide pour {color}")
            return "timeout"

    # Si on atteint le nombre maximum de tours
    print(f"Partie arrêtée après {MAX_TURNS} tours (limite atteinte)")
    return "timeout"  

def check_initialization_phase(game):
    """
    Vérifie si le jeu est toujours en phase d'initialisation.

    Args:  
        game: Instance du jeu  
    """  
    counterinit = 0  
    for pawn in game.pawns:  
        if pawn.color == "blue" and pawn.y == 0:  
            counterinit += 1  
        elif pawn.color == "orange" and pawn.y == 4:  
            counterinit += 1  
    
    if counterinit == 8:  
        print("Fin de l'initialisation du jeu")  
        game.initializing = False  

def get_next_move(game, color, aiblue, aiorange, possible_moves):
    """
    Obtient le prochain mouvement à jouer.

    Args:  
        game: Instance du jeu  
        color: Couleur du joueur actuel  
        aiblue: IA pour le joueur bleu  
        aiorange: IA pour le joueur orange  
        possible_moves: Liste des mouvements possibles  
        
    Returns:  
        list: Mouvement à jouer [color, pawn_type, x, y]  
    """  
    if game.use_ai:  
        ai = aiblue if color == "blue" else aiorange  
        
        if ai.type == "M":  # Minimax  
            print("IA Minimax")
            move = ai.playsmart()  
            if move:  
                print(f"L'IA {color} choisit de déplacer le pion {move[1]} vers ({move[2]}, {move[3]})")
        elif ai.type == "D":  # SB3 DQN
            print("IA SB3 DQN")
            move = ai.playsmart()
            if move:
                # game.simulate_move(*move) # Simuler le mouvement pour vérifier la validité
                print(f"L'IA {color} choisit de déplacer le pion {move[1]} vers ({move[2]}, {move[3]})")
        else:  # IA aléatoire  
            print("IA aléatoire")  
            move = ai.playrandom(possible_moves)  
            try:  
                pawn_to_move = move[1]  
                x, y = move[2], move[3]  
                print(f"L'IA aléatoire {color} choisit de déplacer le pion {pawn_to_move} vers ({x}, {y})")  
            except:  
                print("Aucun mouvement disponible")  
                return None  
    else:  
        # Mode manuel  
        try:  
            pawn_to_move = float(input("Sélectionnez un pion: ") or 1.)  
            x = int(input("x: ") or 1)  
            y = int(input("y: ") or 1)  
            move = [color, pawn_to_move, x, y]  
        except:  
            print("Entrée invalide")  
            return None  
            
    return move  

def execute_move(game, move, color, aiblue, aiorange, data_manager, counter, move_log_filename):
    """
    Exécute un mouvement et vérifie s'il conduit à une victoire.
    """
    pawn_to_move, x, y = move[1], move[2], move[3]

    # Cas spécial: pas de mouvement possible
    if pawn_to_move == -1:
        return None

    # Mise à jour de l'historique du pion
    if data_manager:
        data_manager.update_pawn_history(color, pawn_to_move, (x, y), counter)

    # Recherche du pion à déplacer
    ispawnmoved = [False, False]
    for pawn in game.pawns:
        if pawn.type == pawn_to_move and pawn.color == color:
            # Vérification des règles de retraite
            if not pawns_must_play[color]:
                ispawnmoved = pawn.move(x, y, game.grid, game.pawns, game)
            elif pawn in pawns_must_play[color]:
                ispawnmoved = pawn.move(x, y, game.grid, game.pawns, game)
                if pawn in pawns_must_play[color]:  # Vérifier si le pion est toujours dans la liste
                    pawns_must_play[color].remove(pawn)
            else:
                print(f"Vous devez jouer avec le(s) pion(s) en retraite: {', '.join([str(p.type) for p in pawns_must_play[color]])}")
                ispawnmoved = [False, False]

    # Si le mouvement est valide, l'enregistrer
    if ispawnmoved[0]:
        with open(move_log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([color, pawn_to_move, x, y, counter])

        # Vérification de la victoire
        if ispawnmoved[1]:
            print("Partie terminée")
            print(f"Le gagnant est... {color} !!!")
            print(f"Partie terminée en {counter} tours")

            # Enregistrement des données de fin de partie
            final_stack = game.grid.getfinalstack(x, y)
            ai_info = get_ai_info(game, aiblue, aiorange)
            if data_manager:
                data_manager.write(ai_info, color, counter, game.num_retreat, final_stack)
            game.winner = color

            return "victory"

        return "valid_move"
    else:
        print(f"Mouvement invalide pour {color}: pion {pawn_to_move} vers ({x}, {y})")
        return "invalid_move"  

def get_ai_info(game, aiblue, aiorange):
    """
    Obtient les informations sur les IA utilisées.

    Args:  
        game: Instance du jeu  
        aiblue: IA pour le joueur bleu  
        aiorange: IA pour le joueur orange  
        
    Returns:  
        list: Informations sur les IA  
    """  
    return [  
        {  
            "type": aiblue.type,  
            "depth": aiblue.base_depth if hasattr(aiblue, "base_depth") else None,  
            "color": "BLUE"  
        },  
        {  
            "type": aiorange.type,  
            "depth": aiorange.base_depth if hasattr(aiorange, "base_depth") else None,  
            "color": "ORANGE"  
        }  
    ]  

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

if __name__ == "__main__":
    main()
