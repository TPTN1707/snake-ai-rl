from collections import deque
import os
import random
import numpy as np
import torch
from src.config import BATCH_SIZE, BLOCK_SIZE, GAMMA, LEARNING_RATE, MAX_MEMORY
from src.environment import Direction, Point
from src.model import Linear_QNet, QTrainer

# Check and set device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness controller
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        # Initialize neural network
        self.model = Linear_QNet(11, 256, 3).to(device)

        # Auto-load saved model if it exists
        model_path = "./data/checkpoints/model.pth"
        if os.path.exists(model_path):
            print(f"\n[INFO] Found saved model at '{model_path}'. Loading...")
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                # Skip exploration phase (set games to 80 so epsilon starts at 0)
                self.n_games = 80
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}. Starting fresh.")

        self.trainer = QTrainer(
            self.model, lr=LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game):
        """Extracts an 11-element binary state vector from the current game environment."""
        head = game.head

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Saves a single experience step to replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Samples a random batch and trains the neural network."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the neural network on the immediate single step."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decides the next action based on epsilon-greedy policy."""
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move