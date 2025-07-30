import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CreativeGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 5
        self.max_steps = 30

        # Observation: agent row and column
        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size])
        self.action_space = spaces.Discrete(4) 

        # Goal and obstacles
        self.goal = (4, 4)
        self.obstacles = {(2, 0), (1, 1), (3, 3)}

        # Tile labels (row, col): label
        self.TILE_TYPES = {
            (0, 1): "Module",
            (0, 3): "Module",
            (1, 0): "Module",
            (1, 1): "Distraction",
            (2, 0): "Distraction",
            (2, 3): "Module",
            (3, 1): "Module",
            (3, 3): "Distraction",
            (4, 1): "Module",
            (4, 4): "Milestone"  # Goal
        }

        self.agent_pos = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start agent in a random valid spot
        while True:
            start_row = np.random.randint(0, self.grid_size)
            start_col = np.random.randint(0, self.grid_size)
            if (start_row, start_col) not in self.obstacles and (start_row, start_col) != self.goal:
                self.agent_pos = [start_row, start_col]
                break
        self.steps_taken = 0
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        row, col = self.agent_pos
        self.steps_taken += 1

        # Move logic
        if action == 0 and row > 0: row -= 1      # up
        elif action == 1 and row < self.grid_size - 1: row += 1  # down
        elif action == 2 and col > 0: col -= 1    # left
        elif action == 3 and col < self.grid_size - 1: col += 1  # right

        # Only
        if (row, col) not in self.obstacles:
            self.agent_pos = [row, col]

        # Reward shaping
        dist_to_goal = abs(self.goal[0] - self.agent_pos[0]) + abs(self.goal[1] - self.agent_pos[1])
        reward = -1  

        if dist_to_goal <= 2:
            reward += 3
        if dist_to_goal <= 1:
            reward += 5
        if self.steps_taken > self.max_steps * 0.8:
            reward -= 3

        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 50
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        return np.array(self.agent_pos, dtype=np.int32), reward, done, False, {}

    def render(self):
        grid = [["⬜" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r, c in self.obstacles:
            grid[r][c] = "🟥"
        grid[self.goal[0]][self.goal[1]] = "🟩"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "🟨"
        print("\n".join([" ".join(row) for row in grid]))
        print()