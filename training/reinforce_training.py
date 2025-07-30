import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.grid_env import CreativeGridEnv

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99
n_episodes = 1000
max_steps = 100
save_path = "models/pg/reinforce_grid.pt"
log_path = "logs/reinforce/rewards.txt"

# Define Policy Network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Utilities
def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

# Ensure model and log directories exist
os.makedirs("models/pg", exist_ok=True)
os.makedirs("logs/reinforce", exist_ok=True)

# Environment Setup
env = CreativeGridEnv()
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
policy_net = PolicyNet(obs_dim, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Logging setup
with open(log_path, "w") as f:
    f.write("episode,total_reward,steps\n")

# Training Loop
for episode in range(n_episodes):
    log_probs = []
    rewards = []

    state, _ = env.reset()
    for step in range(max_steps):
        action, log_prob = select_action(policy_net, state)
        next_state, reward, done, _, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

        if done:
            break

    returns = compute_returns(rewards, gamma)
    loss = -torch.sum(torch.stack(log_probs) * returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    with open(log_path, "a") as f:
        f.write(f"{episode+1},{total_reward},{len(rewards)}\n")

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

# Save Model
torch.save(policy_net.state_dict(), save_path)
print(f"\n REINFORCE model saved to: {save_path}")
print(f" Rewards logged to: {log_path}")