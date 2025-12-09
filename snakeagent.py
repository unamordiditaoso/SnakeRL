import os
from snakeenv import SnakeEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3 import DQN, PPO, A2C

SEED = 42
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
REWARD_TARGET = 10000
TOTAL_TIMESTEPS = 4000000

CHECKPOINT_DIR = "./checkpoints_snake/"
TENSORBOARD_DIR = "./tensorboard_snake/"

def get_next_version(base_dir):
    existing = [d for d in os.listdir(base_dir) if d.startswith("v") and os.path.isdir(os.path.join(base_dir, d))]
    if not existing:
        return 1
    versions = [int(d[1:]) for d in existing if d[1:].isdigit()]
    return max(versions) + 1 if versions else 1

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

version = get_next_version(CHECKPOINT_DIR)
version_tag = f"v{version}"

checkpoint_path = os.path.join(CHECKPOINT_DIR, version_tag)
tensorboard_path = os.path.join(TENSORBOARD_DIR, version_tag)

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)

env = SnakeEnv()
env.reset(seed = SEED)
eval_env = SnakeEnv()
eval_env.reset(seed = SEED)

callback_on_best = StopTrainingOnRewardThreshold(REWARD_TARGET, verbose=1)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    callback_after_eval=callback_on_best,
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=checkpoint_path,
    name_prefix=f"dqn_snake_{version_tag}",
    save_replay_buffer=True,
    save_vecnormalize=False,
)

class CustomTensorboardCallback(BaseCallback):
    """Callback que registra métricas personalizadas en TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Puedes registrar métricas personalizadas
        self.logger.record("training/steps_done", self.num_timesteps)
        return True

MODEL_PATH = "./checkpoints_snake/v0/dqn_snake_80000_steps.zip"

if os.path.exists(MODEL_PATH):
    print(f"📦 Cargando modelo existente desde {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, env=env, seed=SEED, tensorboard_log=tensorboard_path)
else:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=tensorboard_path,
        learning_rate = 1e-3,
        buffer_size = 500000,
        learning_starts = 50000,
        batch_size = 128,
        gamma = 0.99,
        train_freq = 4,
        target_update_interval = 5000, 
        exploration_fraction = 0.95, 
        exploration_initial_eps = 1.0,
        exploration_final_eps = 0.001
    )

#    model = PPO(
#    policy="MlpPolicy",
#    env=env,
#    verbose=1,
#    seed=SEED,
#    tensorboard_log=tensorboard_path,
#    learning_rate=1e-4,
#    n_steps=4096,
#    batch_size=128,
#    n_epochs=10,
#    gamma=0.99,
#    gae_lambda=0.95,
#    clip_range=0.2,
#    ent_coef=0.001,
#)

#    model = A2C(
#        policy="MlpPolicy",
#        env=env,
#        verbose=1,
#        seed=SEED,
#        tensorboard_log=tensorboard_path,
#        learning_rate = 1e-4,
#        n_steps = 5_000,
#        gamma = 0.99,
#        gae_lambda = 0.95,
#        ent_coef = 0.001,
#        vf_coef = 0.5,
#        max_grad_norm = 0.5,
#        rms_prop_eps = 1e-5,
#        use_rms_prop = True,
#    )
    
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback, CustomTensorboardCallback()],
    tb_log_name="DQN_Snake_Run",
)

model.learn(
    total_timesteps=500000,
    tb_log_name="DQN_Snake_Run",
    reset_num_timesteps=False
)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"[Snake · DQN] Recompensa media={mean_reward:.1f} ± {std_reward:.1f} (20 episodios)")

print("\n✅ Entrenamiento completado.")
print(f"   TensorBoard -> tensorboard --logdir={TENSORBOARD_DIR}")