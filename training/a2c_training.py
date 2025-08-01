import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_env import CreativeGridEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def make_env():
    def _init():
        env = CreativeGridEnv()
        return env
    return _init

# Ensure log directory exists
log_dir = "logs/a2c_fixed"
models_dir = "models/pg/a2c_fixed"
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

# Initialize A2C with optimized hyperparameters for grid world
model = A2C(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=7e-4,     
    n_steps=128,             
    gamma=0.99,              
    gae_lambda=1.0,         
    ent_coef=0.01,           
    vf_coef=0.25,            
    max_grad_norm=0.5,     
    rms_prop_eps=1e-5,       
    use_rms_prop=True,       
    use_sde=False,           
    normalize_advantage=True,     verbose=1,
    tensorboard_log=log_dir,
    device='auto'
)

# Save initial model
initial_model_path = f"{models_dir}/a2c_initial"
model.save(initial_model_path)
print(f"Initial model saved to: {initial_model_path}")

# Create callback to stop training when solved
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
eval_callback = EvalCallback(
    eval_env, 
    callback_on_new_best=callback_on_best,
    eval_freq=5000,
    best_model_save_path=models_dir,
    log_path=log_dir,
    verbose=1
)

print("Starting A2C training...")
print("Target: Consistently reach milestone with 150+ reward")

# Train the agent
model.learn(
    total_timesteps=200_000,  # Extended training for better convergence
    callback=eval_callback,
    progress_bar=True
)

# Save final model
final_model_path = f"{models_dir}/a2c_final"
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save model with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_model_path = f"{models_dir}/a2c_model_{timestamp}"
model.save(timestamped_model_path)
print(f"Timestamped model saved to: {timestamped_model_path}")

# Extended evaluation
print("\n Evaluating trained A2C model...")
eval_episodes = 15
success_count = 0
total_rewards = []
step_counts = []

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
    step_counts.append(steps)
    
    if episode_reward > 100:  # Consider success if high reward
        success_count += 1
    
    status = "SUCCESS" if episode_reward > 100 else "❌ Failed"
    print(f"Episode {episode + 1}: {episode_reward:.1f} reward in {steps} steps - {status}")

print(f"\n A2C EVALUATION RESULTS:")
print(f"Success rate: {success_count}/{eval_episodes} ({success_count/eval_episodes*100:.1f}%)")
print(f"Average reward: {np.mean(total_rewards):.1f}")
print(f"Average steps: {np.mean(step_counts):.1f}")
print(f"Best episode: {max(total_rewards):.1f}")
print(f"Worst episode: {min(total_rewards):.1f}")

# Save evaluation results
results_file = f"{models_dir}/evaluation_results.txt"
with open(results_file, 'w') as f:
    f.write(f"A2C Training Results - {timestamp}\n")
    f.write(f"=" * 40 + "\n")
    f.write(f"Success rate: {success_count}/{eval_episodes} ({success_count/eval_episodes*100:.1f}%)\n")
    f.write(f"Average reward: {np.mean(total_rewards):.1f}\n")
    f.write(f"Average steps: {np.mean(step_counts):.1f}\n")
    f.write(f"Best episode: {max(total_rewards):.1f}\n")
    f.write(f"Worst episode: {min(total_rewards):.1f}\n")
    f.write(f"Total timesteps: 200,000\n")
    f.write(f"\nHyperparameters:\n")
    f.write(f"  - Learning rate: 7e-4\n")
    f.write(f"  - n_steps: 128\n")
    f.write(f"  - gamma: 0.99\n")
    f.write(f"  - gae_lambda: 1.0\n")
    f.write(f"  - ent_coef: 0.01\n")
    f.write(f"  - vf_coef: 0.25\n")
    f.write(f"\nModel paths:\n")
    f.write(f"  - Initial: {initial_model_path}\n")
    f.write(f"  - Final: {final_model_path}\n")
    f.write(f"  - Timestamped: {timestamped_model_path}\n")

print(f"Evaluation results saved to: {results_file}")

if np.mean(total_rewards) > 100:
    print("A2C training successful! Agent learned to reach milestone efficiently.")
else:
    print("A2C training needs improvement. Consider running longer or adjusting hyperparameters.")

# Demonstration with rendering (optional)
print("\n🎮 Running demonstration with rendering...")
raw_env = CreativeGridEnv()
obs, _ = raw_env.reset()
total_reward = 0
steps = 0

for step in range(50):
    # Use the trained model to predict action
    # Convert single obs to batch format for VecEnv trained model
    obs_batch = np.array([obs])
    action, _ = model.predict(obs_batch, deterministic=True)
    
    obs, reward, done, _, _ = raw_env.step(action[0])  # Extract scalar action
    raw_env.render(mode="human")
    total_reward += reward
    steps += 1
    
    if done:
        print(f" A2C Agent reached goal in {steps} steps with total reward: {total_reward}")
        break

if not done:
    print(f"⏱️ Episode ended after {steps} steps with total reward: {total_reward}")

# Clean up
env.close()
eval_env.close()
raw_env.close()

print(f"\n Training complete! Models saved in: {models_dir}")
print("To load a saved model later, use:")
print(f"model = A2C.load('{final_model_path}')")
