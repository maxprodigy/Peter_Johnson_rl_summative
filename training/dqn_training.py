import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Ensure logs directory exists
log_dir = "logs/dqn_fixed"
models_dir = "models/dqn_fixed"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Create environments
env = CreativeGridEnv()
env = Monitor(env, log_dir)
check_env(env, warn=True)

# Create evaluation environment
eval_env = Monitor(CreativeGridEnv(), log_dir + "/eval")

# Enhanced network architecture for better learning
policy_kwargs = dict(
    net_arch=[512, 512, 256],  # Deeper network
    activation_fn=torch.nn.ReLU
    # Note: 'dueling' is not a policy_kwargs parameter, it goes in the DQN constructor
)

# Initialize DQN with optimized hyperparameters
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,       
    buffer_size=50_000,        
    learning_starts=1000,     
    batch_size=128,            
    tau=0.005,                 
    gamma=0.99,              
    train_freq=4,             
    gradient_steps=1,          
    target_update_interval=1000, 
    exploration_fraction=0.4,   
    exploration_initial_eps=1.0, 
    exploration_final_eps=0.05,  
    verbose=1,
    tensorboard_log=log_dir,
    device='auto'
)

# Save initial model
initial_model_path = f"{models_dir}/dqn_initial"
model.save(initial_model_path)
print(f"Initial model saved to: {initial_model_path}")

# A callback for early stopping
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    eval_freq=10_000,
    best_model_save_path=models_dir,
    log_path=log_dir,
    verbose=1,
    deterministic=True
)

print("Starting DQN training...")
print("Using DQN with optimized hyperparameters")

# Train the agent
model.learn(
    total_timesteps=300_000,  # More training for DQN
    callback=eval_callback,
    progress_bar=True
)

# Save final model
final_model_path = f"{models_dir}/dqn_final"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save model with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_model_path = f"{models_dir}/dqn_model_{timestamp}"
model.save(timestamped_model_path)
print(f"Timestamped model saved to: {timestamped_model_path}")

# Comprehensive evaluation
print("\n Evaluating trained DQN model...")
eval_episodes = 15
success_count = 0
total_rewards = []
step_counts = []

for episode in range(eval_episodes):
    obs, _ = eval_env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)
        episode_reward += reward
        steps += 1
    
    total_rewards.append(episode_reward)
    step_counts.append(steps)
    
    if episode_reward > 100:
        success_count += 1
    
    status = "SUCCESS" if episode_reward > 100 else "Failed"
    print(f"Episode {episode + 1}: {episode_reward:.1f} reward in {steps} steps - {status}")

print(f"\n DQN EVALUATION RESULTS:")
print(f"Success rate: {success_count}/{eval_episodes} ({success_count/eval_episodes*100:.1f}%)")
print(f"Average reward: {np.mean(total_rewards):.1f}")
print(f"Average steps: {np.mean(step_counts):.1f}")
print(f"Best episode: {max(total_rewards):.1f}")
print(f"Worst episode: {min(total_rewards):.1f}")

# Save evaluation results
results_file = f"{models_dir}/evaluation_results.txt"
with open(results_file, 'w') as f:
    f.write(f"DQN Training Results - {timestamp}\n")
    f.write(f"=" * 40 + "\n")
    f.write(f"Success rate: {success_count}/{eval_episodes} ({success_count/eval_episodes*100:.1f}%)\n")
    f.write(f"Average reward: {np.mean(total_rewards):.1f}\n")
    f.write(f"Average steps: {np.mean(step_counts):.1f}\n")
    f.write(f"Best episode: {max(total_rewards):.1f}\n")
    f.write(f"Worst episode: {min(total_rewards):.1f}\n")
    f.write(f"Total timesteps: 300,000\n")
    f.write(f"Model paths:\n")
    f.write(f"  - Initial: {initial_model_path}\n")
    f.write(f"  - Final: {final_model_path}\n")
    f.write(f"  - Timestamped: {timestamped_model_path}\n")

print(f"Evaluation results saved to: {results_file}")

if np.mean(total_rewards) > 120:
    print(" DQN training successful!")
else:
    print("DQN needs more training or hyperparameter tuning.")

# Clean up
env.close()
eval_env.close()

print(f"\n Training complete! Models saved in: {models_dir}")
print("To load a saved model later, use:")
print(f"model = DQN.load('{final_model_path}')")

# Optional: a script to test the trained model
test_script = f"""# Test script for trained DQN model
from stable_baselines3 import DQN
from environment.grid_env import CreativeGridEnv

# Load the trained model
model = DQN.load('{final_model_path}')

# Create environment
env = CreativeGridEnv()

# Test the model
obs, _ = env.reset()
for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        print(f"Episode completed in {{step+1}} steps with reward {{reward}}")
        break

env.close()
"""

with open(f"{models_dir}/test_trained_model.py", 'w') as f:
    f.write(test_script)

print(f"Test script created: {models_dir}/test_trained_model.py")
