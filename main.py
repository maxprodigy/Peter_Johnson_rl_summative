from environment.grid_env import CreativeGridEnv  
from environment.rendering import visualize_grid_env
from stable_baselines3 import A2C, PPO  # or DQN, etc.

if __name__ == "__main__":
    env = CreativeGridEnv()
    model = A2C.load("models/pg/a2c_grid.zip") # Make Changes Based On The Model Path Here
    visualize_grid_env(env, episodes=3, fps=5, tile_size=100, model=model)
