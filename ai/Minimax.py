import copy
import random

class Minimax:
    def __init__(self, color, game):
        self.type = "M"
        self.color = color
        self.game = game
        self.base_depth = self.set_base_depth_by_color(color)
        self.moves_scores = {} 
        self.ispawnmoved = [False, False]


    def set_base_depth_by_color(self, color):
        if color == "blue":
            return 4
        elif color == "orange":
            return 4

    def minimax(self, game, depth, max_depth, is_maximizing, alpha=float('-inf'), beta=float('inf'), move=None):
        if self.color == "blue":
            if depth == max_depth or self.ispawnmoved[1]:
                return game.evaluateClassic(self.color)
        if self.color == "orange":
            if depth == max_depth or self.ispawnmoved[1]:
                return game.evaluateClassic(self.color)

        if is_maximizing:
            max_eval = float('-inf')
            for next_move in game.all_next_moves(self.color):
                simulated_game = copy.deepcopy(game)
                color, piece_type, x, y = next_move
                self.ispawnmoved = simulated_game.simulate_move(color, piece_type, x, y)
                eval = self.minimax(simulated_game, depth + 1, max_depth, False, alpha, beta, next_move)
                if depth == 0:  # Enregistrement à la racine
                    self.moves_scores[tuple(next_move)] = eval
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent_color = "orange" if self.color == "blue" else "blue"
            for next_move in game.all_next_moves(opponent_color):
                simulated_game = copy.deepcopy(game)
                color, piece_type, x, y = next_move
                simulated_game.simulate_move(color, piece_type, x, y)
                eval = self.minimax(simulated_game, depth + 1, max_depth, True, alpha, beta, next_move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def choose_best_move(self):
        if self.moves_scores:
            max_score = max(self.moves_scores.values())
            best_moves = [move for move, score in self.moves_scores.items() if score == max_score]
            # Choisir aléatoirement parmi les meilleurs coups si plusieurs ont le même score
            best_move = random.choice(best_moves)
            print("Best move chosen randomly from top scoring moves:", best_move)
            return list(best_move)
        else:
            print("No valid moves found, returning default move.")
            all_moves = self.game.all_next_moves(self.color)
            if all_moves:
                return random.choice(all_moves)  # Choisir un coup aléatoire parmi tous les coups possibles
            else:
                return [self.color, -1, -1, -1]

    def playsmart(self):
        self.moves_scores = {}
        self.minimax(self.game, 0, self.base_depth, True)
        return self.choose_best_move()



    