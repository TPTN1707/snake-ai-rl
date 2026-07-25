from collections import deque
import random
import numpy as np
import torch
from src.config import BATCH_SIZE, BLOCK_SIZE, GAMMA, LEARNING_RATE, MAX_MEMORY
from src.environment import Direction, Point
from src.model import Linear_QNet, QTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness controller (Exploration vs Exploitation)
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(
            self.model, lr=LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game):
        """Extracts an 11-element binary state vector from the current game environment."""
        head = game.head

        # Define points immediately adjacent to the snake head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Detect current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Danger straight ahead
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # 2. Danger to the right relative to snake direction
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # 3. Danger to the left relative to snake direction
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # 4. Current moving directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # 5. Food position relative to snake head
            game.food.x < game.head.x,  # Food is left
            game.food.x > game.head.x,  # Food is right
            game.food.y < game.head.y,  # Food is up
            game.food.y > game.head.y,  # Food is down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Saves a single experience step to the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Samples a random batch from memory and trains the neural network (Experience Replay)."""
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
        # Exploration vs Exploitation tradeoff
        # As n_games increases, epsilon decreases, making the AI choose smarter moves instead of random
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Explore: Take a random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: Query the neural network for prediction
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move