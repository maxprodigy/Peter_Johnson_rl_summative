import sys
import os
import pygame
import time
import torch
import torch.nn as nn
import numpy as np

# Setup for importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.grid_env import CreativeGridEnv

# Load trained REINFORCE model
MODEL_PATH = "models/reinforce_grid.pt"

# Policy Network must match training
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Load environment and model
env = CreativeGridEnv()
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = PolicyNet(obs_dim, n_actions)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()

# Grid display settings
TILE_SIZE = 100
GRID_WIDTH = 5
GRID_HEIGHT = 5
WIDTH = TILE_SIZE * GRID_WIDTH
HEIGHT = TILE_SIZE * GRID_HEIGHT + 50
FPS = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (144, 238, 144)
BLUE = (100, 149, 237)
RED = (255, 99, 71)
YELLOW = (255, 255, 0)

TILE_TYPES = {
    (0, 1): "Module",
    (0, 3): "Module",
    (1, 0): "Module",
    (1, 1): "Distraction",
    (2, 0): "Distraction",
    (2, 3): "Module",
    (3, 1): "Module",
    (4, 1): "Module",
    (4, 4): "Milestone"  # Goal
}

def select_action(state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy_net(state_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    return action

def run_episode():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Creative Agent Simulation (REINFORCE)")

    font = pygame.font.SysFont("consolas", 22)
    label_font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    done = False
    steps = 0
    reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = select_action(obs)
        obs, r, done, _, _ = env.step(action)
        reward += r
        steps += 1

        screen.fill(WHITE)
        agent_pos = env.agent_pos

        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x = col * TILE_SIZE
                y = row * TILE_SIZE
                tile_color = GREEN

                if (row, col) == env.goal:
                    tile_color = RED
                elif (row, col) in env.obstacles:
                    tile_color = BLUE

                pygame.draw.rect(screen, tile_color, (x, y, TILE_SIZE, TILE_SIZE))
                pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 1)

                if (row, col) in TILE_TYPES:
                    label_text = TILE_TYPES[(row, col)]
                    label = label_font.render(label_text, True, BLACK)
                    label_rect = label.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                    screen.blit(label, label_rect)

        ax = agent_pos[1] * TILE_SIZE
        ay = agent_pos[0] * TILE_SIZE
        pygame.draw.rect(screen, YELLOW, (ax, ay, TILE_SIZE, TILE_SIZE))
        agent_label = font.render("Creative", True, BLACK)
        agent_rect = agent_label.get_rect(center=(ax + TILE_SIZE // 2, ay + TILE_SIZE // 2))
        screen.blit(agent_label, agent_rect)

        step_label = font.render(f"Step: {steps}   Reward: {reward}", True, BLACK)
        screen.blit(step_label, (10, HEIGHT - 40))

        pygame.display.flip()
        time.sleep(0.3)
        clock.tick(FPS)

    time.sleep(1)
    pygame.quit()

if __name__ == "__main__":
    for i in range(3):
        print(f"\n🎬 Episode {i + 1}")
        run_episode()
