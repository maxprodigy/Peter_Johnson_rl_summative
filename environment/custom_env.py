import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LearningPathEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Agent profile: interest (0–2), skill level (0–1), courses taken (0–5)
        self.observation_space = spaces.MultiDiscrete([3, 2, 6])
        self.action_space = spaces.Discrete(3)  # 3 courses to pick from

        self.max_courses = 5
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            np.random.randint(0, 3),  # interest
            np.random.randint(0, 2),  # skill level
            0  # courses taken
        ])
        self.done = False
        return self.state, {}

    def step(self, action):
        interest, skill, courses_taken = self.state

        reward = -5  # default penalty

        # Reward if course matches interest
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

    def render(self):
        print(f"State: {self.state}")
