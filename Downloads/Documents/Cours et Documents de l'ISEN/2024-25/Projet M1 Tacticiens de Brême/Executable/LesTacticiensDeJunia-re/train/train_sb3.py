import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

import shutil
# if os.path.exists(LOG_DIR):
#     shutil.rmtree(LOG_DIR)
# os.makedirs(LOG_DIR, exist_ok=True)

# Configuration des paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from gym_env.tacticiens_env import TacticiensDeBremeEnv

def setup_environment(opponent_type: str, log_dir: str):
    """Crée et valide l'environnement"""
    env = TacticiensDeBremeEnv(opponent_ai_type=opponent_type)
    
    # Vérification cruciale de la compatibilité
    try:
        check_env(env)
    except Exception as e:
        print(f"Erreur de compatibilité Gymnasium : {e}")
        raise
    
    return Monitor(env, log_dir)

def create_model(env, log_dir):
    """Configure le modèle DQN avec hyperparamètres optimisés"""
    return DQN(
        "MlpPolicy",
        env,
        verbose=2,
        # tensorboard_log=log_dir,
        tensorboard_log=None,  # Désactivé pour éviter les conflits de permissions
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,  # Augmenté pour plus de stabilité
        batch_size=256,
        # gamma=0.99,
        gamma=0.95,  #  Récompense à plus court terme
        # exploration_fraction=0.2,
        exploration_fraction=0.5, # Augmenté pour une exploration plus prolongée
        # exploration_final_eps=0.02,
        exploration_final_eps=0.1,  # Augmenté pour une exploration plus prolongée
        target_update_interval=2000,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs={
            "net_arch": [256, 256],
            "normalize_images": False  # Important pour les observations non-image
        }
    )

def check_permissions():
    test_dir = "./logs/test_dir"
    try:
        os.makedirs(test_dir, exist_ok=True)
        with open(f"{test_dir}/test.txt", "w") as f:
            f.write("test")
        print("✅ Permissions OK")
        shutil.rmtree(test_dir)
    except Exception as e:
        print(f"❌ Erreur : {e}")

def main():
    # 1. Configuration
    # LOG_DIR = "./logs/"
    # os.makedirs(LOG_DIR, exist_ok=True)
    LOG_DIR = os.path.abspath("./logs/")  # Chemin absolu
    os.makedirs(LOG_DIR, exist_ok=True, mode=0o777)  # Droits étendus

    # Ajoute cette vérification pour debug
    print(f"Dossier logs : {LOG_DIR} | Existe : {os.path.exists(LOG_DIR)} | Accès en écriture : {os.access(LOG_DIR, os.W_OK)}")

    # 2. Initialisation Phase 1 (vs Random)
    print("=== PHASE 1 : Entraînement vs IA Aléatoire ===")
    env = setup_environment("random", LOG_DIR)
    
    # 3. Configuration des callbacks
    callbacks = [
        EvalCallback(
            env,
            eval_freq=10000,
            best_model_save_path=LOG_DIR,
            verbose=1
        ),
        CheckpointCallback(
            save_freq=20000,
            save_path=LOG_DIR,
            name_prefix="phase1"
        )
    ]

    # 4. Création et entraînement du modèle
    model = create_model(env, LOG_DIR)
    model.learn(
        total_timesteps=150000,
        callback=callbacks,
        tb_log_name="dqn_phase1",
        progress_bar=True,
        log_interval=10
    )

    # 5. Phase 2 (vs Minimax)
    print("\n=== PHASE 2 : Transfer Learning vs Minimax ===")
    env = setup_environment("minimax", LOG_DIR)
    model.set_env(env)
    
    # Mise à jour des callbacks
    callbacks[0] = EvalCallback(
        env,
        eval_freq=20000,
        best_model_save_path=LOG_DIR,
        verbose=1
    )

    model.learn(
        total_timesteps=300000,
        callback=callbacks,
        tb_log_name="dqn_phase2",
        reset_num_timesteps=False,
        progress_bar=True
    )

    # 6. Sauvegarde finale
    model.save("dqn_tacticiens_final")
    print(f"\nEntraînement terminé. Visualisez les résultats avec :\ntensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    # check_permissions()

    main()