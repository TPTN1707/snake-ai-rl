# Game Display Settings
BLOCK_SIZE = 20  # Pixel size of each grid cell
WINDOW_WIDTH = 640  # Total width of the game window
WINDOW_HEIGHT = 480  # Total height of the game window
GAME_SPEED = 120  # Set high (e.g., 120+) for fast training, lower (e.g., 20) to watch comfortably

# RGB Colors
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (200, 0, 0)  # Used for the food
COLOR_BLUE_DARK = (0, 0, 100)  # Outer snake body
COLOR_BLUE_LIGHT = (0, 100, 255)  # Inner snake body
COLOR_GREEN = (0, 200, 0)
COLOR_BLACK = (0, 0, 0)  # Background color

# Reinforcement Learning (RL) Hyperparameters
MAX_MEMORY = 100_000  # Max number of experiences to store in replay memory
BATCH_SIZE = 1000  # Number of samples to train on in each batch
LEARNING_RATE = 0.001  # Learning rate for the neural network optimizer
GAMMA = 0.9  # Discount factor for future rewards (must be < 1)