from src.agent import Agent
from src.environment import SnakeGameAI
from src.utils.helper import plot
import torch

# Limit PyTorch to a single thread to prevent GIL conflicts with Pygame and Matplotlib GUI threads
torch.set_num_threads(1)

def train():
    """Main training loop for the Snake RL Agent."""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Instantiate the agent and the game environment
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # 1. Get the current (old) state of the game
        state_old = agent.get_state(game)

        # 2. Decide the next action based on the state
        final_move = agent.get_action(state_old)

        # 3. Perform the action in the environment
        reward, done, score = game.play_step(final_move)
        state_new = game_state_new = agent.get_state(game)

        # 4. Train short-term memory (immediate single-step learning)
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done
        )

        # 5. Store the experience in replay memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Episode ended: Reset the game and update statistics
            game.reset()
            agent.n_games += 1

            # Train long-term memory (Experience Replay on a random batch)
            agent.train_long_memory()

            # Save the neural network weights if a new high-score record is set
            if score > record:
                record = score
                agent.model.save()

            # Print progress log to terminal
            print(
                f"Game: {agent.n_games} | Score: {score} | Record: {record}"
            )

            # Record scores for real-time visualization
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Dynamically update the progress graph
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()