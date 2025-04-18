import gym
from gym import spaces
import numpy as np
from game.game import Game
from ai.dummyAI import Dummyai
from ai.Minimax import Minimax

class TacticiensDeBremeEnv(gym.Env):
    def __init__(self, opponent_ai_type="random"):
        super().__init__()
        
        # Initialisation correcte de Game
        self.game = Game(
            data_manager=None,
            manual_mode=False,
            use_ai=True,
            ai_types=(1, 2 if opponent_ai_type == "random" else 1)  # (bleu, orange)
        )
        
        # Configuration des joueurs
        self.dqn_color = "blue"  # Le DQN contrôle toujours les bleus
        self.opponent_color = "orange"
        self.opponent_ai_type = opponent_ai_type
        
        # Initialisation des IAs adverses
        if opponent_ai_type == "minimax":
            self.opponent_ai = Minimax("orange", self.game)
        else:  # "random"
            self.opponent_ai = Dummyai("orange")
        
        # Espaces d'observation et d'action
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5, 5), dtype=int)  # 5x5 grid, 5 layers (base + 4 animaux)
        self.action_space = spaces.Discrete(100)  # 25 cases × 4 directions
        
        # Variables de suivi
        self.n_steps = 0
        self.max_steps = 200

    def reset(self):
        """Réinitialise l'environnement pour une nouvelle partie."""
        self.game = Game(
            data_manager=None,
            manual_mode=False,
            use_ai=True,
            ai_types=(1, 2 if self.opponent_ai_type == "random" else 1)
        )
        # Force l'affichage cohérent
        self.game.grid.all_pawns = self.game.pawns  # Synchronisation cruciale
        
        # Réinit des IAs
        if self.opponent_ai_type == "minimax":
            self.opponent_ai = Minimax(self.opponent_color, self.game)
        else:
            self.opponent_ai = Dummyai(self.opponent_color)
            
        self.n_steps = 0
        return self._get_observation()

    def step(self, action):
        """
        Exécute un pas de temps dans l'environnement.
        
        Args:
            action: Action choisie par le DQN
            
        Returns:
            observation, reward, done, info
        """
        info = {}
        
        # Tour du DQN (bleu)
        dqn_move = self._action_to_move(action)
        success, victory = self._execute_move(dqn_move, "blue")
        
        if victory:
            return self._get_observation(), 1.0, True, {"winner": "blue"}
        
        # Tour de l'IA adverse (orange)
        opponent_move = self._get_opponent_move()
        if opponent_move is None:  # Aucun mouvement possible
            return self._get_observation(), 0.0, True, {"winner": "draw"}
            
        _, opponent_victory = self._execute_move(opponent_move, "orange")
        
        if opponent_victory:
            return self._get_observation(), -1.0, True, {"winner": "orange"}
        
        # Calcul des récompenses et vérification de fin de partie
        reward = self._calculate_reward()
        self.n_steps += 1
        done = self.n_steps >= self.max_steps
        
        return self._get_observation(), reward, done, info

    def _get_opponent_move(self):
        """Obtient le mouvement de l'IA adverse."""
        possible_moves = self.game.all_next_moves("orange")
        
        if not possible_moves:
            return None
            
        if self.opponent_ai_type == "random":
            return self.opponent_ai.playrandom(possible_moves)
        else:  # Minimax
            return self.opponent_ai.playsmart()

    def _execute_move(self, move, color):
        """Exécute un mouvement et retourne (success, victory)."""
        if move is None:
            return False, False
            
        _, pawn_type, x, y = move
        for pawn in self.game.pawns:
            if pawn.color == color and pawn.type == pawn_type:
                return pawn.move(x, y, self.game.grid, self.game.pawns, self.game, simulate=True)
        return False, False

    def _action_to_move(self, action):
        """Convertit une action du DQN en mouvement valide."""
        pawn_type = (action // 25) + 1  # Types 1-4
        x = (action % 25) % 5
        y = (action % 25) // 5
        return ["blue", pawn_type, x, y]

    # def _get_observation(self):
    #     """Convertit l'état du jeu en observation pour le DQN."""
    #     obs = np.zeros((5, 5, 5), dtype=int)
        
    #     for y in range(5):
    #         for x in range(5):
    #             stack = self.game.grid.grid[y][x]
    #             if stack[0] != 0:  # Case non vide
    #                 # Couche 0: propriétaire de la base
    #                 base_color = 1 if self._get_pawn_color(stack[0], x, y) == "blue" else 2
    #                 obs[y, x, 0] = base_color
                    
    #                 # Couches 1-4: présence des pions (1-4)
    #                 for i, pawn_type in enumerate(stack, 1):
    #                     obs[y, x, pawn_type] = 1
                        
    #     return obs

    def _get_observation(self):
        obs = np.zeros((5, 5, 5), dtype=int)
        
        for pawn in self.game.pawns:
            x, y = pawn.x, pawn.y
            if 0 <= x < 5 and 0 <= y < 5:  # Case valide
                # Channel 0: propriétaire de la base
                if obs[y, x, 0] == 0:  # Si la case n'a pas encore de propriétaire
                    obs[y, x, 0] = 1 if pawn.color == "blue" else 2
                
                # Channels 1-4: présence des pions
                obs[y, x, pawn.type] = 1  # Type 1=Âne, 2=Chien, etc.
        
        return obs

    def _get_pawn_color(self, pawn_type, x, y):
        """Trouve la couleur d'un pion spécifique."""
        for pawn in self.game.pawns:
            if pawn.type == pawn_type and pawn.x == x and pawn.y == y:
                return pawn.color
        return None

    def _calculate_reward(self):
        """
        Calcule la récompense intermédiaire.
        Stratégie: bonus pour les piles partielles, malus pour la durée.
        """
        reward = -0.01  # Malus pour chaque tour
        
        # Bonus pour les piles partielles
        for y in range(5):
            for x in range(5):
                stack = self.game.grid.grid[y][x]
                if len(stack) >= 2:
                    # Bonus progressif selon la taille de la pile
                    reward += 0.02 * len(stack)
                    
                    # Bonus supplémentaire si le Coq (4) est en bas
                    if stack[0] == 4:
                        reward += 0.05
                        
        return round(reward, 4)
    
    def render(self, mode='human'):
        """
        Affiche le plateau de jeu en utilisant le système existant de Grid.
        Args:
            mode: 'human' pour l'affichage console, 'rgb_array' pour un rendu image (optionnel)
        """
        if mode == 'human':
            # Utilise la méthode display() existante de Grid
            print("\n=== État du plateau ===")
            # print(f"Tour: {self.n_steps} | Joueur actuel: {self.current_player}")
            self.game.grid.display()
            return True
        elif mode == 'rgb_array':
            # Pour une future intégration graphique
            raise NotImplementedError("Le rendu RGB array n'est pas implémenté")
        else:
            super().render(mode=mode)  # Fallback Gym