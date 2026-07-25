import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    """Plots the current score and the running average score of the AI during training."""
    plt.clf()
    plt.title("Training Progress...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot both current scores and running averages
    plt.plot(scores, label="Current Score")
    plt.plot(mean_scores, label="Running Average")
    plt.ylim(ymin=0)

    # Label the last data point with its exact value on the graph
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.show()
    # Pause briefly to allow Pygame and Matplotlib UI to refresh
    plt.pause(0.1)