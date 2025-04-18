import os
import sys
import gym
from stable_baselines3.common.env_checker import check_env

# Ajout du dossier racine au path  
current_dir = os.path.dirname(os.path.abspath(__file__))  
root_dir = os.path.dirname(current_dir)  
sys.path.append(root_dir)  

from gym_env.tacticiens_env import TacticiensDeBremeEnv

# Créer l'environnement avec affichage
# env = TacticiensDeBremeEnv(opponent_type='random', show_display=True)

# Réinitialiser l'environnement
# obs, _ = env.reset()
# obs = env.reset()

# # Simuler quelques mouvements
# for _ in range(10):  # Simuler 10 mouvements
#     action = env.action_space.sample()  # Choisir une action aléatoire
#     obs, reward, done, _, _ = env.step(action)  # Effectuer le mouvement
#     if done:
#         print("Partie terminée.")
#         break

# # Fermer l'environnement
# env.close()

# env = TacticiensDeBremeEnv(opponent_ai="random")
# obs = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()  # Action aléatoire
#     obs, reward, done, info = env.step(action)
#     print(f"Reward: {reward}, Done: {done}")
#     if done:
#         print("Episode terminé")
#         break

# env.close()

env = TacticiensDeBremeEnv(opponent_ai_type="random")  # ou "minimax"
check_env(env)  # Vérification de l'environnement
# # Test aléatoire
# obs = env.reset()
# print("=== PIONS INITIAUX ===")
# for pawn in env.game.pawns:
#     print(f"{pawn.color} {pawn.type} @ ({pawn.x},{pawn.y})")

# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     print(f"Step: {_}, Reward: {reward}, Done: {done}")
#     env.game.grid.display()  # Affiche la grille après chaque mouvement
#     # Afficher l'état de l'environnement (optionnel)
#     # env.render()  # Affiche l'état actuel de l'environnement
#     if done:
#         print("Exemple d'observation :\n", obs[:, :, 0])  # Affiche les propriétaires des bases
#         print("Coqs présents :\n", obs[:, :, 4])  # Affiche les positions des Coqs
#         for pawn in env.game.pawns:
#             print(f"{pawn.color} {pawn.type} @ ({pawn.x},{pawn.y})")

# # Test avec un agent (exemple simple)
# class RandomAgent:
#     def predict(self, obs):
#         return env.action_space.sample()

# agent = RandomAgent()
# obs = env.reset()
# a = 0
# while a < 10:
#     a += 1
#     action = agent.predict(obs)
#     obs, reward, done, info = env.step(action)
#     if done:
#         print(f"Partie terminée. Gagnant: {info.get('winner', '?')}")
#         continue

#     env = TacticiensDeBremeEnv(opponent_ai_type="random")  # ou "minimax"