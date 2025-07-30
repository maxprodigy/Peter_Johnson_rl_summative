import torch
import numpy as np
import gymnasium as gym
from torch import nn
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces

# Custom Environment 
class LearningPathEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([3, 2, 6])
        self.action_space = spaces.Discrete(3)
        self.max_courses = 5
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.random.randint(0, 3), np.random.randint(0, 2), 0])
        self.done = False
        return self.state, {}

    def step(self, action):
        interest, skill, courses_taken = self.state
        reward = -5
        if action == interest:
            reward = 10
            if skill == 1:
                reward += 5
        courses_taken += 1
        self.state = np.array([interest, skill, courses_taken])
        if courses_taken >= self.max_courses:
            reward += 50
            self.done = True
        return self.state, reward, self.done, False, {}

# REINFORCE PolicyNet
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Evaluation Functions
def make_vec_env(seed):
    def _init():
        env = LearningPathEnv()
        env.reset(seed=seed)
        return env
    return DummyVecEnv([_init])

def evaluate_sb3_model(model, seeds):
    rewards = []
    for seed in seeds:
        vec_env = make_vec_env(seed)
        obs = vec_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            ep_reward += reward[0]
            done = done[0]
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

def evaluate_reinforce_model(policy_net, seeds):
    rewards = []
    for seed in seeds:
        env = LearningPathEnv()
        env.reset(seed=seed)
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy_net(state_tensor)
            action = torch.argmax(probs, dim=-1).item()
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

# === Run Evaluation ===
seeds = [101, 202, 303, 404, 505]
results = {}

# PPO
ppo = PPO.load("models/ppo_grid.zip")
results["PPO"] = evaluate_sb3_model(ppo, seeds)

# DQN
dqn = DQN.load("models/dqn_grid.zip")
results["DQN"] = evaluate_sb3_model(dqn, seeds)

# A2C
a2c = A2C.load("models/a2c_grid.zip")
results["A2C"] = evaluate_sb3_model(a2c, seeds)

# REINFORCE
policy_net = PolicyNet(3, 3)
policy_net.load_state_dict(torch.load("models/reinforce_grid.pt"))
policy_net.eval()
results["REINFORCE"] = evaluate_reinforce_model(policy_net, seeds)

# Print Results
print("\nGeneralization Test Results:")
print(f"{'Agent':<10} | {'Mean Reward':>12} | {'Std Dev':>8}")
print("-" * 36)
for agent, (mean, std) in results.items():
    print(f"{agent:<10} | {mean:>12.2f} | {std:>8.2f}")