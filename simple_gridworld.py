import random
import time

import gymnasium
import pygame
import torch
from gymnasium import spaces
import numpy as np
from torchvision.transforms import transforms

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

    def __init__(self, text_file, cell_size=(20, 20), obs_size=(128, 128), agent_position=None, goal_position=None, random_traps=0, make_random=False, max_steps=128):
        super(TextGridWorld, self).__init__()
        self.random = make_random
        self.max_steps = max_steps
        self.step_count = 0

        self.grid = self.load_grid(text_file)
        self.cell_size = cell_size
        self.screen_size = (self.grid.shape[1] * cell_size[0], self.grid.shape[0] * cell_size[1])
        self.viewer = None
        self.cached_surface = None

        self._agent_position = agent_position
        self._goal_position = goal_position

        self.num_random_traps = random_traps
        self.pos_ramdom_traps = []

        self.agent_position = None
        self.goal_position = None

        # gymnasium required
        self.obs_size = obs_size  # H, W
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, obs_size[0], obs_size[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        pygame.init()

        self.occupied_positions = set()
        self.reset_positions()

    def load_grid(self, text_file):
        with open(text_file, 'r') as file:
            lines = file.read().splitlines()
        self.grid = np.array([list(line) for line in lines])
        return self.grid

    def rotate_grid(self, grid):
        """Rotate the grid randomly by 0, 90, 180, or 270 degrees."""
        rotations = random.choice([0, 1, 2, 3])
        return np.rot90(grid, k=rotations)

    def flip_grid(self, grid):
        """Flip the grid randomly: horizontally, vertically, or not at all."""
        flip_type = random.choice(["horizontal", "vertical", "none"])
        if flip_type == "horizontal":
            return np.fliplr(grid)
        elif flip_type == "vertical":
            return np.flipud(grid)
        return grid

    def reset_positions(self):
        self.occupied_positions = set()

        self.agent_position = self._agent_position if self._agent_position else self.assign_position()
        self.occupied_positions.add(self.agent_position)

        self.goal_position = self._goal_position if self._goal_position else self.assign_position()
        self.occupied_positions.add(self.goal_position)

        self.pos_ramdom_traps = []
        for i in range(random.randint(0, self.num_random_traps)):
            try:
                trap = self.assign_position()
                self.pos_ramdom_traps.append(trap)
                self.occupied_positions.add(trap)
            except RuntimeError:
                print("Warning: Not enough empty position assignable for random traps.")

    def assign_position(self):
        counter = 0
        while True:
            position = (random.randint(0, self.grid.shape[0] - 1), random.randint(0, self.grid.shape[1] - 1))
            if position not in self.occupied_positions and self.grid[position] not in ['W', 'X', 'G']:
                return position
            counter += 1
            if counter >= 16384:
                raise RuntimeError("Cannot assign any more positions.")

    def step(self, action):
        self.step_count += 1
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        delta = deltas[action]
        new_position = (self.agent_position[0] + delta[0], self.agent_position[1] + delta[1])

        hits_wall = False
        if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
            if self.grid[new_position] not in ['W']:
                self.agent_position = new_position
            elif self.grid[new_position] == 'W':
                hits_wall = True

        terminated = self.agent_position == self.goal_position  # or self.grid[self.agent_position] == 'X'
        truncated = False
        reward = 5 if self.agent_position == self.goal_position else -1 if (self.grid[self.agent_position] == 'X' or self.agent_position in self.pos_ramdom_traps) else -0.01
        if hits_wall:
            reward -= 0.1

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
            reward = 0
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if self.random:
            # Randomly rotate the grid
            self.grid = self.rotate_grid(self.grid)
            # Randomly flip the grid
            self.grid = self.flip_grid(self.grid)
        # if not self._agent_position:
        #     self.agent_position = self.assign_position({self.goal_position})
        # if not self._goal_position:
        #     self.goal_position = self.assign_position({self.agent_position})
        self.reset_positions()

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0
        return observation, {}

    def get_observation(self):
        observation = pygame.surfarray.array3d(self.cached_surface)
        observation = np.transpose(observation, (1, 0, 2))  # WHC - HWC
        return observation

    def _render_to_surface(self):
        if self.viewer is None:
            self.viewer = pygame.Surface(self.screen_size)
            self.cached_surface = pygame.Surface(self.obs_size)  # 使用obs_size代替screen_size

        self.viewer.fill((255, 255, 255))

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                cell_content = self.grid[y, x]
                rect = pygame.Rect(x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])
                if cell_content == 'W':
                    pygame.draw.rect(self.viewer, (0, 0, 0), rect)
                elif cell_content == 'X' or (y, x) in self.pos_ramdom_traps:
                    pygame.draw.rect(self.viewer, (255, 0, 0), rect)

        goal_rect = pygame.Rect(self.goal_position[1] * self.cell_size[0], self.goal_position[0] * self.cell_size[1],
                                self.cell_size[0], self.cell_size[1])
        pygame.draw.rect(self.viewer, (0, 255, 0), goal_rect)

        agent_center = (int(self.agent_position[1] * self.cell_size[0] + self.cell_size[0] / 2),
                        int(self.agent_position[0] * self.cell_size[1] + self.cell_size[1] / 2))
        pygame.draw.circle(self.viewer, (0, 0, 255), agent_center, int(self.cell_size[0] / 2))

        scaled_surface = pygame.transform.scale(self.viewer, self.obs_size)
        self.cached_surface.blit(scaled_surface, (0, 0))

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

    def handle_keyboard_input(self):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
        return action


def preprocess_image(img: np.ndarray, rotate=False, size=None) -> torch.Tensor:
    # Convert the numpy array to a PIL Image
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(img)
    # Initialize the transformation list
    transformations = []
    # Randomly rotate the image
    if rotate:
        rotation_degrees = np.random.choice([0, 90, 180, 270])
        transformations.append(transforms.RandomRotation([rotation_degrees, rotation_degrees]))
    # Resize the image if size is specified
    if size is not None:
        transformations.append(transforms.Resize(size))
    # Convert the PIL Image back to a tensor
    transformations.append(transforms.ToTensor())
    # Compose all transformations
    transform_compose = transforms.Compose(transformations)
    # Apply transformations
    processed_tensor = transform_compose(pil_image)
    # Normalize the tensor to [0, 1] (if not already normalized)
    processed_tensor /= 255.0 if processed_tensor.max() > 1.0 else 1.0
    # Add a batch dimension
    processed_tensor = processed_tensor.unsqueeze(0)
    return processed_tensor


if __name__ == "__main__":
    env = TextGridWorld('envs/simple_grid/gridworld-empty-traps-7.txt', agent_position=(1, 1), goal_position=(3, 3), random_traps=0)
    obs = env.reset()
    done = False
    reward_sum = 0
    while not done:
        env.render(mode='human')
        action = env.handle_keyboard_input()
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated:
                print("Game Over. Reward:", reward_sum)
                done = True
            elif truncated:
                print("Episode truncated.")
                done = True
        time.sleep(0.1)
    env.close()
