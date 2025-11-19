import time
import cv2
import numpy as np
from stable_baselines3 import DQN

# Importa tu entorno
from snakeenv import SnakeEnv


MODEL_PATH = "./checkpoints_snake/v20/dqn_snake_v20_3800000_steps.zip"   # ← CAMBIA AQUÍ TU MODELO


def play(model_path, episodes=5):
    env = SnakeEnv()
    model = DQN.load(MODEL_PATH, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        score = 0

        print(f"\n🎮 Episodio {ep+1}")

        while not (done or truncated):
            # Acción del modelo (determinista)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Paso del entorno
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Mostrar ventana
            cv2.imshow("Snake AI", env.img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                print("⛔ Salida manual.")
                return

            score = env.score

        print(f"✔ Episodio terminado | Score = {score} | Reward total = {total_reward:.2f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    play(MODEL_PATH, episodes=10)