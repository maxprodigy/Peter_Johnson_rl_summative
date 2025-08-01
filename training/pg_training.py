import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def make_env():
    """Create environment function"""
    def _init():
        env = CreativeGridEnv()
        return env
    return _init

# Ensure log directory exists
log_dir = "logs/ppo_fixed"
models_dir = "models/ppo_fixed"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Create environments
env = DummyVecEnv([make_env()])
eval_env = DummyVecEnv([make_env()])

# Check environment
test_env = CreativeGridEnv()
check_env(test_env, warn=True)

# Enhanced policy network architecture
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks for policy and value
    activation_fn=torch.nn.ReLU
)

# Initialize PPO with optimal hyperparameters for grid world
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,  
    n_steps=2048,        
    batch_size=64,       
    gamma=0.99,          
    gae_lambda=0.95,    
    clip_range=0.2,      
    ent_coef=0.01,       
    vf_coef=0.5,        
    max_grad_norm=0.5,   
    verbose=1,
    tensorboard_log=log_dir,
    device='auto'
)

# Save initial model
initial_model_path = f"{models_dir}/ppo_initial"
model.save(initial_model_path)
print(f"Initial model saved to: {initial_model_path}")

# A callback to stop training when solved
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
eval_callback = EvalCallback(
    eval_env, 
    callback_on_new_best=callback_on_best,
    eval_freq=5000,
    best_model_save_path=models_dir,
    log_path=log_dir,
    verbose=1
)

print("Starting PPO training...")
print("Target: Consistently reach milestone with 150+ reward")

# Train the agent
model.learn(
    total_timesteps=200_000,  
    callback=eval_callback,
    progress_bar=True
)

# Save final model
final_model_path = f"{models_dir}/ppo_final"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save model with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_model_path = f"{models_dir}/ppo_model_{timestamp}"
model.save(timestamped_model_path)
print(f"Timestamped model saved to: {timestamped_model_path}")

# Extended evaluation
print("\n Evaluating trained model...")
eval_episodes = 10
success_count = 0
total_rewards = []

for episode in range(eval_episodes):
    obs = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]  # VecEnv returns arrays
        steps += 1
    
    total_rewards.append(episode_reward)
    if episode_reward > 100:  # Consider success if high reward
        success_count += 1
    
    print(f"Episode {episode + 1}: {episode_reward:.1f} reward in {steps} steps")

print(f"\n EVALUATION RESULTS:")
print(f"Success rate: {success_count}/{eval_episodes} ({success_count*10}%)")
print(f"Average reward: {np.mean(total_rewards):.1f}")
print(f"Best episode: {max(total_rewards):.1f}")
print(f"Worst episode: {min(total_rewards):.1f}")

# Save evaluation results
results_file = f"{models_dir}/evaluation_results.txt"
with open(results_file, 'w') as f:
    f.write(f"PPO Training Results - {timestamp}\n")
    f.write(f"=" * 40 + "\n")
    f.write(f"Success rate: {success_count}/{eval_episodes} ({success_count*10}%)\n")
    f.write(f"Average reward: {np.mean(total_rewards):.1f}\n")
    f.write(f"Best episode: {max(total_rewards):.1f}\n")
    f.write(f"Worst episode: {min(total_rewards):.1f}\n")
    f.write(f"Total timesteps: 200,000\n")
    f.write(f"Model paths:\n")
    f.write(f"  - Initial: {initial_model_path}\n")
    f.write(f"  - Final: {final_model_path}\n")
    f.write(f"  - Timestamped: {timestamped_model_path}\n")

print(f"Evaluation results saved to: {results_file}")

if np.mean(total_rewards) > 100:
    print("Training successful! Agent learned to reach milestone efficiently.")
else:
    print("Training needs improvement. Consider running longer or adjusting hyperparameters.")

# Clean up
env.close()
eval_env.close()

print(f"\n Training complete! Models saved in: {models_dir}")
print("To load a saved model later, use:")
print(f"model = PPO.load('{final_model_path}')")
