# Snake AI - Reinforcement Learning (DQN)

An interactive Reinforcement Learning project that teaches an AI agent to play the classic Snake game from scratch using Deep Q-Learning (DQN). Built with Python, PyTorch, and Pygame.

## Project Architecture

The project is structured with a clean, modular design separating the game environment, the neural network, and the agent logic:

```text
snake-ai-rl/
├── data/                       # Training checkpoints and output plots
│   ├── checkpoints/            # Saved model weights (.pth)
│   └── plots/                  # Auto-saved progress graphs (.png)
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Hyperparameters and game settings
│   ├── environment.py          # Pygame Snake game environment customized for RL
│   ├── agent.py                # RL Agent (state extraction, memory, epsilon-greedy)
│   ├── model.py                # Deep Q-Network (PyTorch) and Q-Trainer
│   └── utils/
│       ├── __init__.py
│       └── helper.py           # Real-time Matplotlib plotting helper
├── main.py                     # Training orchestrator and entry point
└── requirements.txt            # Dependency list
```

## Features

- **Deep Q-Network (DQN):** A 3-layer neural network built with PyTorch utilizing the Adam optimizer and Mean Squared Error loss.
- **Experience Replay Memory:** Stores up to 100,000 steps of experience and samples random batches (size 1000) to break temporal correlation for stable training.
- **Auto-Save Checkpoints:** Automatically saves the neural network weights (`model.pth`) whenever a new high-score record is set.
- **Auto-Resume Training:** Automatically detects and loads previously saved weights on startup, allowing the agent to play with its learned skills immediately.
- **Dynamic Real-Time Plotting:** Plots both the current score and the accurate running average of the current session, automatically saving the plot as `progress.png` in `data/plots/`.
- **GIL Thread Safety:** Configured to run on a single PyTorch thread to prevent conflicts with Pygame and Matplotlib GUI loops.

## Requirements

- Python 3.11 (Recommended for ML package stability)
- Windows OS (Optimized for Pygame & Matplotlib interactive threads)

## Installation & Execution

1. **Install dependencies:**
   Using `uv` (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```
   Or standard pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training orchestrator:**
   ```bash
   uv run main.py
   ```
   *(Or `python main.py`)*

## Reinforcement Learning Setup

- **State Representation (11-element binary vector):**
  - Danger status [Straight, Right, Left] (Relative to snake head)
  - Current moving direction [Left, Right, Up, Down]
  - Food position relative to head [Left, Right, Up, Down]
- **Action Space:** `[1, 0, 0]` (Straight), `[0, 1, 0]` (Turn Right), `[0, 0, 1]` (Turn Left).
- **Reward System:**
  - Eat food: `+10`
  - Collision (Wall or self): `-10`
  - Loop timeout penalty (Forces game over if the snake wanders without eating).