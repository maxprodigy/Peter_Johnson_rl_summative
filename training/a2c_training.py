import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Ensure log directory exists
log_dir = "logs/a2c"
os.makedirs(log_dir, exist_ok=True)

# Create unwrapped env for rendering
raw_env = CreativeGridEnv()
env = Monitor(raw_env, log_dir)
check_env(env, warn=True)

# Define custom policy network (optional)
policy_kwargs = dict(net_arch=[128, 128])

# Train A2C agent
model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=0.0007, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=100_000)

# Save model
os.makedirs("models/pg", exist_ok=True)
model.save("models/pg/a2c_grid")

# Evaluate briefly
obs, _ = raw_env.reset()
total_reward = 0
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = raw_env.step(action)
    raw_env.render(mode="human")
    total_reward += reward
    if done:
        print(f"\n A2C Agent reached goal with total reward: {total_reward}")
        break

raw_env.close()
