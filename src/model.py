import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward propagation through ReLU activation."""
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        """Saves PyTorch model state dictionary to the checkpoints directory."""
        model_folder_path = "./data/checkpoints"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class QTrainer:

    def __init__(self, model, lr, gamma):
        self.model = model.to(device)
        self.lr = lr
        self.gamma = gamma
        # Adam optimizer is highly effective for Deep Q-Learning
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss function
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Performs a single optimization step using Bellman's equation."""
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        # Handle 1D single-step arrays by adding a batch dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Predict Q-values with current state
        pred = self.model(state)

        # 2. Calculate target Q-values using Bellman Equation: Q_new = R + gamma * max(Q(next_state))
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = (
                    reward[idx]
                    + self.gamma * torch.max(self.model(next_state[idx])).item()
                )

            # Map the one-hot action vector index (0, 1, or 2) to the update target
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # 3. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()