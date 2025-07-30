import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Ensure logs directory exists
log_dir = "logs/dqn"
os.makedirs(log_dir, exist_ok=True)

# Create environment
raw_env = CreativeGridEnv()  
env = Monitor(raw_env, log_dir)
check_env(env, warn=True)

# Define custom architecture
policy_kwargs = dict(net_arch=[128, 128])

# Initialize DQN with exploration settings
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0005,
    buffer_size=20000,
    learning_starts=500,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=250,
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log=log_dir
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save model
os.makedirs("models/dqn", exist_ok=True)
model.save("models/dqn/dqn_grid")

# Quick evaluation
obs, _ = raw_env.reset()
total_reward = 0
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = raw_env.step(action)
    raw_env.render(mode="human")
    total_reward += reward
    if done:
        print(f"\n DQN Agent reached goal with total reward: {total_reward}")
        break

env.close()
