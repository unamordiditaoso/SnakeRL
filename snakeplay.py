import time
import cv2
import numpy as np
from stable_baselines3 import DQN

# Importa tu entorno
from snakeenv import SnakeEnv


MODEL_PATH = "./checkpoints_snake/v6/best_model.zip" 


def play(model_path, episodes=5):
    env = SnakeEnv()
    model = DQN.load(MODEL_PATH, env=env)
    contadorObjetivo = 0

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        score = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            #cv2.imshow("Snake", env.img)
            #key = cv2.waitKey(1)
            #if key == ord('q'):
            #    print("⛔ Salida manual.")
            #    return
                        
            score = env.score
        
        if(score >= 50):
            contadorObjetivo += 1

            print(f"\n🎮 Episodio {ep+1}")
            print(f"✔ Episodio terminado | Score = {score} | Reward total = {total_reward:.2f}")

    porcentaje = (contadorObjetivo / 100000) * 100

    print(f"\n" 
      f"✔ Objetivo alcanzado: {contadorObjetivo} veces en 100000 intentos.\n"
      f"📊 Porcentaje de éxito: {porcentaje:.2f}%\n")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play(MODEL_PATH, episodes=100000)