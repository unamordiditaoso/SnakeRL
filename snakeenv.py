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
        if direction == 0: next_head[0] -= 10  # izquierda
        if direction == 1: next_head[0] += 10  # derecha
        if direction == 2: next_head[1] += 10  # abajo
        if direction == 3: next_head[1] -= 10  # arriba

        return (
            collision_with_boundaries(next_head) or
            next_head in self.snake_position
        )

    def step(self, action):
        if is_opposite_direction(action, self.prev_button_direction):
            action = self.prev_button_direction
        else:
            self.prev_button_direction = action

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

        apple_reward = 0
        distance_prev = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 50 + self.score * 2
            self.steps_since_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.steps_since_apple += 1

        if self.steps_since_apple >= TIMEOUT_STEPS:
            self.truncated = True
            self.done = True

        if collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position):
            self.done = True

        distance_now = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        distance_improvement = distance_prev - distance_now

        self.reward = (
            apple_reward
            + 0.5 * distance_improvementç
            - 0.001 * self.steps_since_apple
        )

        if self.done:
            dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            max_dist = np.sqrt(500**2 + 500**2)
            proximity_factor = dist_to_apple / max_dist
            self.reward -= 5 + 10 * proximity_factor

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

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length, danger_s, danger_l, danger_r] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.img = np.zeros((500,500,3),dtype='uint8')

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

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        danger_s = self.danger(self.prev_button_direction)
        danger_l = self.danger((self.prev_button_direction - 1) % 4)
        danger_r = self.danger((self.prev_button_direction + 1) % 4)

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length, danger_s, danger_l, danger_r] + list(self.prev_actions)
        observation = np.array(observation)
        
        info = {}
        
        return observation, info