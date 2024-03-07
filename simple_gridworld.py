import random
import time

import gymnasium
import pygame
from gymnasium import spaces
import numpy as np


ACTION_NAMES = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
}


class TextGridWorld(gymnasium.Env):
    """

    The TextGridWorld class is an environment class that represents a grid world. The grid world is loaded from a text file, where each character represents a cell in the grid. The grid
    * can contain walls ('W'), traps ('T'), the agent ('A'), and the goal ('G').

    This class inherits from the 'gymnasium.Env' class and overrides its methods.

    Args:
        text_file (str): The path to the text file containing the grid. Each line in the file represents a row in the grid, and each character in the line represents a cell in the row.
        cell_size (tuple, optional): The size of each grid cell in pixels. Defaults to (20, 20).
        agent_position (tuple, optional): The initial position of the agent in the grid. If not specified, a random position will be assigned. Defaults to None.
        goal_position (tuple, optional): The position of the goal in the grid. If not specified, a random position will be assigned. Defaults to None.

    Attributes:
        metadata (dict): Metadata about the environment, including the supported render modes.
        grid (numpy array): The grid representing the environment.
        action_space (gymnasium.Space): The action space for the agent.
        observation_space (gymnasium.Space): The observation space for the agent.
        cell_size (tuple): The size of each grid cell in pixels.
        screen_size (tuple): The size of the screen in pixels.
        viewer (pygame.Surface): The surface used for rendering the environment.
        cached_surface (pygame.Surface): The cached surface used for rendering.
        _agent_position (tuple): The initial position of the agent, might be None.
        _goal_position (tuple): The position of the goal, might be None.
        agent_position (tuple): The current position of the agent.
        goal_position (tuple): The current position of the goal.

    Methods:
        load_grid(text_file): Loads the grid from a text file.
        reset_positions(): Resets the positions of the agent and the goal.
        assign_position(occupied_positions): Assigns a position that is not occupied by the agent or the goal.
        step(action): Performs a step in the environment given an action.
        reset(seed=None, options=None): Resets the environment to its initial state.
        get_observation(): Returns the current observation of the environment.
        _render_to_surface(): Renders the environment to the cached surface.
        render(mode='human'): Renders the environment in a specified mode.
        close(): Closes the environment.

    """
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, text_file, cell_size=(20, 20), agent_position=None, goal_position=None):
        super(TextGridWorld, self).__init__()
        self.grid = self.load_grid(text_file)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.grid.shape), shape=(2,), dtype=np.int32)
        self.cell_size = cell_size
        self.screen_size = (self.grid.shape[1] * cell_size[0], self.grid.shape[0] * cell_size[1])
        self.viewer = None
        self.cached_surface = None

        self._agent_position = agent_position
        self._goal_position = goal_position

        self.agent_position = None
        self.goal_position = None

        pygame.init()

        self.reset_positions()

    def load_grid(self, text_file):
        with open(text_file, 'r') as file:
            lines = file.read().splitlines()
        grid = np.array([list(line) for line in lines])
        return grid

    def reset_positions(self):
        occupied_positions = set()
        self.agent_position = self._agent_position if self._agent_position else self.assign_position(occupied_positions)
        occupied_positions.add(self.agent_position)

        self.goal_position = self._goal_position if self._goal_position else self.assign_position(occupied_positions)

    def assign_position(self, occupied_positions):
        while True:
            position = (random.randint(0, self.grid.shape[0] - 1), random.randint(0, self.grid.shape[1] - 1))
            if position not in occupied_positions and self.grid[position] not in ['W', 'X', 'G']:
                return position

    def step(self, action):
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        delta = deltas[action]
        new_position = (self.agent_position[0] + delta[0], self.agent_position[1] + delta[1])

        if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
            if self.grid[new_position] not in ['W', 'X']:
                self.agent_position = new_position

        terminated = self.agent_position == self.goal_position or self.grid[self.agent_position] == 'X'
        truncated = False
        reward = 1 if self.agent_position == self.goal_position else -1 if self.grid[self.agent_position] == 'X' else 0

        self._render_to_surface()
        return self.get_observation(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._agent_position:
            self.agent_position = self.assign_position({self.goal_position})
        if not self._goal_position:
            self.goal_position = self.assign_position({self.agent_position})

        self._render_to_surface()  # 重新渲染环境状态
        return self.get_observation(), {}

    def get_observation(self):
        observation = pygame.surfarray.array3d(self.cached_surface)
        # observation = np.transpose(observation, (1, 0, 2))
        return observation

    def _render_to_surface(self):
        if self.viewer is None:
            self.viewer = pygame.Surface(self.screen_size)
            self.cached_surface = pygame.Surface(self.screen_size)

        self.viewer.fill((255, 255, 255))

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                cell_content = self.grid[y, x]
                rect = pygame.Rect(x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])
                if cell_content == 'W':
                    pygame.draw.rect(self.viewer, (0, 0, 0), rect)
                elif cell_content == 'X':
                    pygame.draw.rect(self.viewer, (255, 0, 0), rect)

        goal_rect = pygame.Rect(self.goal_position[1] * self.cell_size[0], self.goal_position[0] * self.cell_size[1],
                                self.cell_size[0], self.cell_size[1])
        pygame.draw.rect(self.viewer, (0, 255, 0), goal_rect)

        agent_center = (int(self.agent_position[1] * self.cell_size[0] + self.cell_size[0] / 2),
                        int(self.agent_position[0] * self.cell_size[1] + self.cell_size[1] / 2))
        pygame.draw.circle(self.viewer, (0, 0, 255), agent_center, int(self.cell_size[0] / 2))

        self.cached_surface.blit(self.viewer, (0, 0))

    def render(self, mode='rgb_array'):
        if mode == 'human':
            if self.cached_surface is not None:
                pygame.init()
                window = pygame.display.set_mode(self.screen_size)
                window.blit(self.cached_surface, (0, 0))
                pygame.display.flip()
        elif mode == 'rgb_array':
            return self.get_observation()
        elif mode == 'console':
            for y in range(self.grid.shape[0]):
                row = ''
                for x in range(self.grid.shape[1]):
                    if (y, x) == self.agent_position:
                        row += 'A'
                    elif (y, x) == self.goal_position:
                        row += 'G'
                    elif self.grid[y, x] == 'W':
                        row += '#'
                    elif self.grid[y, x] == 'X':
                        row += 'X'
                    else:
                        row += ' '
                print(row)

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None


if __name__ == "__main__":
    env = TextGridWorld('gridworld_empty.txt')
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render(mode='console')
        env.render(mode='human')
        if terminated:
            print("Game Over. Reward:", reward)
            break
        elif truncated:
            print("Episode truncated.")
            break
        time.sleep(1)
    env.close()
