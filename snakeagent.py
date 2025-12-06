import os
from snakeenv import SnakeEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines3 import DQN, PPO, A2C

SEED = 42
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
REWARD_TARGET = 10000
TOTAL_TIMESTEPS = 2000000

CHECKPOINT_DIR = "./checkpoints_snake/"
TENSORBOARD_DIR = "./tensorboard_snake/"

# --- DETERMINAR VERSIÓN ---
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

# --- Guardar modelo cada 100.000 pasos ---
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,               # cada 100.000 pasos
    save_path=checkpoint_path,      # carpeta donde se guardarán
    name_prefix=f"a2c_snake_{version_tag}", # prefijo del nombre del archivo
    save_replay_buffer=True,        # (opcional) guarda también el buffer de replay
    save_vecnormalize=False,        # no usas VecNormalize, así que False
)

class CustomTensorboardCallback(BaseCallback):
    """Callback que registra métricas personalizadas en TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Puedes registrar métricas personalizadas
        self.logger.record("training/steps_done", self.num_timesteps)
        return True

MODEL_PATH = "./checkpoints_snake/v0/dqn_snake_80000_steps.zip"  # ruta a tu archivo

if os.path.exists(MODEL_PATH):
    print(f"📦 Cargando modelo existente desde {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, env=env, seed=SEED, tensorboard_log=tensorboard_path)
else:
#    model = PPO(
#    policy="MlpPolicy",
#    env=env,
#    verbose=1,
#    seed=SEED,
#    tensorboard_log=tensorboard_path,

#    learning_rate=1e-4,         # Más bajo que en DQN; PPO suele usar LR pequeños
#    n_steps=4096,               # Tamaño del batch de rollout (equivale al buffer)
#    batch_size=128,             # Igual que DQN para consistencia
#    n_epochs=10,                # Veces que pasa por cada batch

#    gamma=0.99,                 # Mismo descuento
#    gae_lambda=0.95,            # Valores típicos de PPO (bias/variance trade-off)

#    clip_range=0.2,             # Clip estándar
#    ent_coef=0.001,             # Entropía: actúa como “exploración”
#)
    model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=tensorboard_path,

        # --- Hiperparámetros principales ---
        learning_rate = 1e-4,      # Similar a PPO; A2C es sensible a LR altos
        n_steps = 5_000,           # "Batch" de experiencia antes de actualizar
        gamma = 0.99,              # Igual que en DQN
        gae_lambda = 0.95,         # Ayuda a estabilizar (igual que PPO)

        ent_coef = 0.001,          # Controla la exploración
        vf_coef = 0.5,             # Peso de la loss de valor
        max_grad_norm = 0.5,       # Clipping de gradiente
    
        rms_prop_eps = 1e-5,       # Default recomendado
        use_rms_prop = True,       # A2C funciona mejor con RMSprop
    )
    
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback, CustomTensorboardCallback()],
    tb_log_name="A2C_Snake_Run",
)

model.learn(
    total_timesteps=500000,
    tb_log_name="A2C_Snake_Run",
    reset_num_timesteps=False
)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"[Snake · A2C] Recompensa media={mean_reward:.1f} ± {std_reward:.1f} (20 episodios)")

meta_conseguida = env.goal_reached

print("\n✅ Entrenamiento completado.")
print(f"\n🏁 Meta conseguida: {meta_conseguida} veces.")
print(f"   TensorBoard -> tensorboard --logdir={TENSORBOARD_DIR}")