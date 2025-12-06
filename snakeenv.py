# Adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
# Get from Sentdex: https://www.youtube.com/watch?v=uKnjGn8fF70&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1&index=3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30
TIMEOUT_STEPS = 1000

# pip install stable_baselines3
# pip install opencv-python
# pip install tensorboard
# python .\snakeagent.py
# python -m tensorboard.main --logdir=./tensorboard_snake/

# 1. Añadir un timeout ==> 100 pasos sin coger manzana.
# 1. Prohibir suicidio.
# 1. Cuanto más cerca se muera de la fruta más recompensa.

# 2. Cuantas más manzanas coja más recompensa (Modificar la recompensa de la primera manzana)
# 2. Modificar cercanía a la manzana cuando muere a cercanía en general.

# 3. He modificado la recompensa para que sea más fuerte cuanto más cerca este.
# 3. He aumentado la recompensa por coger la manzana. ( + por coger manzana [40 * 1.5])
# 3. He modificado la penalización para que sea proporcional a la distancia.
# 3. He incrementado un poco el timeout.

# 4. He modificado la recompensa ( + por coger manzana [50 * 2])
# 4. He cambiado learning rate (5e-3 -> 1e-5)
# 4. He cambiado el batch size (32 -> 64)
# 4. Nuevo modelo desde 0

# 5. He cargado el modelo de v4 otra vez para entrenarlo. (Modelo cargado: 8000 pasos - mejor resultado.)

# 6. Cambio de modelo a Double DQN + Dueling DQN
# 6. Exploration fraction 0.90 y 2_000_000 de pasos | batch size a 128 y learning rate a 1e-3 | buffer size 200_000

# 7. Exploration fraction 0.95 y 2_000_000 de pasos | TIMEOUT a 1000

# 8. Añadir información de peligro respecto a su cuerpo y penalizar por tomar la dirección peligrosa o por encerrarse.
# 8. Guardar modelo cada 100_000 pasos

# 9. Mayor penalización por chocarse con su cuerpo. 0.25 --> 0.75
# 9. Menor penalización por encerrarse. 5 --> 4

# 10. Mayor penalización por chocarse con su cuerpo. 0.75 --> 1.5
# 10. Menor penalización por encerrarse. 4 --> 3

# 11. Volver a datos de v10
# 11. Añadir penalizaciones por chocarse con su propio cuerpo (20 de penalización) y por chocarse contra las paredes (5 de penalización)

# 12. Modificar penalización por chocarse con su propio cuerpo (20 -> 5).
# 12. Modificar penalización por chocarse contra las paredes (5 -> 1).

# 13. Modificar penalización por chocarse con su propio cuerpo (5 -> 1).
# 13. Modificar penalización por chocarse contra las paredes (1 -> 0.25).

# 14. Quitar penalización por chocarse con su propio cuerpo.
# 14. Quitar penalización por chocarse contra las paredes.
# 14. Añadir penalización por girar cuando había peligro hacia esa dirección (2 de penalización)

# 15. Eliminar penalización por girar cuando había peligro hacia esa dirección tras probar bastantes numeros.
# 15. Añadir recompensa por mantenerse lejos de su cola.

# 16. Eliminar recompensa por mantenerse lejos de su cola.
# 16. Modificar valores del agente:
#       Timesteps: 2000000 -> 4000000
#       Buffer_size: 20000 -> 50000
#       Learning_starts: 10000 -> 50000
#       Target_update_interval: 1000 -> 5000

# 17. Modificar valores del agente:
#       Learning_rate: 1e-3 -> 1e-5

# 18. Modificar valores del agente:
#       Learning_rate: 1e-5 -> 5e-5

# 19. Modificar valores del agente:
#       Learning_rate: 5e-5 -> 5e-3

# 20. V16
# 20. Modificar todas las recompensas //Posible cambio en apple_reward
# 20. Entrenar 500000 pasos mas con el exploration_rate al minimo
# 20. Añadir un contador con el numero de veces que se llega a la meta de 30 de largo.

# 21. Cambios en la recompensas porque la serpiente preferia ir recto todo el rato que ir a por la manzana (por la recompensa de danger).

# 22. Coger modelo v8 poner 4000000 de pasos aprendiendo y 500000 usando todo lo que sabe (Probar con exploration_rate de 0.90 a 0.20 y luego 500000 en 0.01 para futuros intentos).

# 23. Cambio en la apple_reward (50 * 2 -> 40 * 2)

# 24. Cambio en la apple_reward (30 * 2 -> 30 * 1.5)

# 25. Cambio en la apple_reward (30 * 1.5 -> 50 * 1.5)

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0:
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

def is_opposite_direction(new_action, current_action):
    # Evita giros de 180°
    opposite_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    return opposite_pairs.get(new_action) == current_action

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(8+SNAKE_LEN_GOAL,), dtype=np.float64)
        self.truncated = False
        self.steps_since_apple = 0

    def danger(self, direction):
        next_head = list(self.snake_head)
        if direction == 0: next_head[0] -= 10  # left
        if direction == 1: next_head[0] += 10  # right
        if direction == 2: next_head[1] += 10  # down
        if direction == 3: next_head[1] -= 10  # up

        return (
            collision_with_boundaries(next_head) or
            next_head in self.snake_position
        )

    def step(self, action):
        # Bloquear dirección opuesta
        if is_opposite_direction(action, self.prev_button_direction):
            action = self.prev_button_direction
        else:
            self.prev_button_direction = action  # actualizar solo si es válida

        self.prev_actions.append(action)

#        cv2.imshow('a',self.img)
#        cv2.waitKey(1)

#        # Descomentar para usar el script snakeplay.py
#        self.img = np.zeros((500,500,3),dtype='uint8')
#        # Display Apple
#        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
#        # Display Snake
#        for position in self.snake_position:
#            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
#        
#        # Takes step after fixed time
#        t_end = time.time() + 0.05
#        k = -1
#        while time.time() < t_end:
#            if k == -1:
#                k = cv2.waitKey(1)
#            else:
#                continue

        button_direction = action

        # --- Recompensas y penalizaciones ---
        apple_reward = 0
        distance_prev = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        # Movimiento de la serpiente
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        # --- Comprobación de manzana ---
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 50 + self.score * 2  # incrementa con cada manzana
            self.steps_since_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.steps_since_apple += 1

        # --- Timeout ---
        if self.steps_since_apple >= TIMEOUT_STEPS:
            print(self.steps_since_apple)
            self.truncated = True
            self.done = True

        # --- Colisiones ---
        if collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position):
            self.done = True

        # --- Distancia a la manzana ---
        distance_now = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        distance_improvement = distance_prev - distance_now  # positivo si se acerca

        # --- Recompensa total ---
        #  +10 al comer, + (mejor si se acerca), - (si se aleja), - (por morir o timeout)
        self.reward = (
            apple_reward
            + 0.5 * distance_improvement   # más peso a acercarse
            - 0.001 * self.steps_since_apple
        )

        if self.done:
            dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            max_dist = np.sqrt(500**2 + 500**2)
            proximity_factor = dist_to_apple / max_dist
            self.reward -= 5 + 10 * proximity_factor
        
        ##print(self.total_reward)

        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        danger_s = self.danger(self.button_direction)
        danger_l = self.danger((self.button_direction - 1) % 4)
        danger_r = self.danger((self.button_direction + 1) % 4)

        if not danger_s:
            self.reward += 0.05
        else:
            self.reward -= 1.5

        if danger_s and danger_l and danger_r:
            self.reward -= 3

        # create observation:

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length, danger_s, danger_l, danger_r] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.img = np.zeros((500,500,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250,250]

        self.steps_since_apple = 0
        self.prev_reward = 0

        self.done = False
        self.truncated = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1) # to create history

        danger_s = self.danger(self.prev_button_direction)
        danger_l = self.danger((self.prev_button_direction - 1) % 4)
        danger_r = self.danger((self.prev_button_direction + 1) % 4)

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length, danger_s, danger_l, danger_r] + list(self.prev_actions)
        observation = np.array(observation)
        
        info = {}
        
        return observation, info