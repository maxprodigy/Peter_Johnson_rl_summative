import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Ensure log directory exists
log_dir = "logs/pg"
os.makedirs(log_dir, exist_ok=True)

# Create unwrapped environment for rendering
raw_env = CreativeGridEnv()
env = Monitor(raw_env, log_dir)
check_env(env, warn=True)

# Define policy network architecture
policy_kwargs = dict(net_arch=[128, 128])

# Initialize PPO agent
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0005,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=log_dir
)

# Train the agent
model.learn(total_timesteps=100000)

# Save model in correct directory
os.makedirs("models", exist_ok=True)
model.save("models/pg/ppo_grid")

# Evaluate briefly using unwrapped environment
obs, _ = raw_env.reset()
total_reward = 0
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = raw_env.step(action)
    raw_env.render(mode="human")
    total_reward += reward
    if done:
        print(f"\n🎯 PPO Agent reached goal with total reward: {total_reward}")
        break

raw_env.close()