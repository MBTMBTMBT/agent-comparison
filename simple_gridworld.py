import collections
import math
import random
from itertools import product

import gymnasium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame
import torch
from gymnasium import spaces
from torchvision.transforms import transforms

ACTION_NAMES = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
}


def rotate_grid(grid, coords):
    """Rotate the grid randomly by 0, 90, 180, or 270 degrees. Update non-None coords accordingly, leaving None values in place.

    Args:
        grid (np.array): The grid to rotate.
        coords (list of tuple or None): List of (row, col) coordinates or None values to update.

    Returns:
        tuple: The rotated grid and the updated list of coordinates with None values preserved.
    """
    rotations = random.choice([0, 1, 2, 3])
    grid_height, grid_width = grid.shape
    new_coords = []
    for coord in coords:
        if coord is not None:
            row, col = coord
            if rotations == 1:  # 90 degrees
                new_coords.append((col, grid_height - 1 - row))
            elif rotations == 2:  # 180 degrees
                new_coords.append((grid_height - 1 - row, grid_width - 1 - col))
            elif rotations == 3:  # 270 degrees
                new_coords.append((grid_width - 1 - col, row))
            else:  # 0 degrees, no change
                new_coords.append(coord)
        else:
            new_coords.append(None)  # Preserve None in its position
    return np.rot90(grid, k=rotations), new_coords


def flip_grid(grid, coords):
    """Flip the grid randomly: horizontally, vertically, or not at all. Update non-None coords accordingly, leaving None values in place.

    Args:
        grid (np.array): The grid to flip.
        coords (list of tuple or None): List of (row, col) coordinates or None values to update.

    Returns:
        tuple: The flipped grid and the updated list of coordinates with None values preserved.
    """
    flip_type = random.choice(["horizontal", "vertical", "none"])
    grid_height, grid_width = grid.shape
    new_coords = []
    for coord in coords:
        if coord is not None:
            row, col = coord
            if flip_type == "horizontal":
                new_coords.append((row, grid_width - 1 - col))
            elif flip_type == "vertical":
                new_coords.append((grid_height - 1 - row, col))
            else:  # No flip, preserve original coord
                new_coords.append(coord)
        else:
            new_coords.append(None)  # Preserve None in its position
    if flip_type == "horizontal":
        return np.fliplr(grid), new_coords
    elif flip_type == "vertical":
        return np.flipud(grid), new_coords
    return grid, new_coords  # No flip performed, return original grid and coords with None preserved.


def get_traversed_grids(pos: np.ndarray, previous_position: np.ndarray):
    # Determine the ranges for each dimension
    ranges = [range(min(a, b), max(a, b) + 1) for a, b in zip(previous_position, pos)]

    # Generate all combinations of these ranges across dimensions
    all_points = list(product(*ranges))

    # Convert tuples back to lists (or to whatever format is preferred)
    traversed_grids = [list(point) for point in all_points]

    return set(tuple(grid) for grid in traversed_grids)


class SimpleGridWorld(gymnasium.Env, collections.abc.Iterator):
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

    def __init__(self, text_file, cell_size=(20, 20), obs_size=(128, 128), agent_position=None, goal_position=None,
                 random_traps=0, make_random=False, max_steps=128,
                 # compute_optimal=False,
                 ):
        super(SimpleGridWorld, self).__init__()
        self.random = make_random
        self.max_steps = max_steps
        self.step_count = 0

        self.grid = self.load_grid(text_file)
        self.cell_size = cell_size
        self.screen_size = (self.grid.shape[1] * cell_size[0], self.grid.shape[0] * cell_size[1])
        self.viewer = None
        self.cached_surface = None
        self.window = None

        self._agent_position = agent_position
        self._goal_position = goal_position

        self.num_random_traps = random_traps
        self.pos_random_traps = []

        self.agent_position = None
        self.goal_position = None

        # gymnasium required
        self.obs_size = obs_size  # H, W
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, obs_size[0], obs_size[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.num_actions = len(ACTION_NAMES)

        pygame.init()

        self.occupied_positions = set()
        self.reset_positions()

        self.shape = self.grid.shape
        self.iter_index = 0
        self.iter_coord = (0, 0)

        self._traj = []

        self.value_grid = np.zeros_like(self.grid, dtype=np.float32)
        self.control_info_grid = np.zeros_like(self.grid, dtype=np.float32)
        self.info_to_go_grid = np.zeros_like(self.grid, dtype=np.float32)
        self.optimal_policy = np.zeros((*self.grid.shape, 4), dtype=np.float16)
        # self.compute_optimal_policy = compute_optimal

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        return self

    def __next__(self):
        temp_agent_position = self.agent_position  # should be helpful for not messing up the coordinates
        if self.iter_index >= len(self):
            raise StopIteration
        row, col = self.iter_coord
        self.agent_position = self.iter_coord
        temp_iter_coord = self.iter_coord
        col += 1
        if col >= self.shape[1]:
            col = 0
            row += 1
        self.iter_coord = (row, col)
        self.iter_index += 1

        if self.grid[self.agent_position[0], self.agent_position[1]] not in ['W']:
            terminated = self.agent_position == self.goal_position  # or self.grid[self.agent_position] == 'X'
            self._render_to_surface()
            observation = self.get_observation()
            observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
            observation /= 255.0 if observation.max() > 1.0 else 1.0

            connections = {}
            rewards = {}
            deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            for action in ACTION_NAMES.keys():
                delta = deltas[action]
                new_position = (self.agent_position[0] + delta[0], self.agent_position[1] + delta[1])
                rewards[len(ACTION_NAMES)] = 5 if self.agent_position == self.goal_position else -1 if (self.grid[
                                                                                                            self.agent_position] == 'X' or self.agent_position in self.pos_random_traps) else -0.01
                if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
                    if self.grid[new_position] not in ['W']:
                        # new position reachable
                        connections[action] = 1.0
                        rewards[action] = 0.0
                        rewards[action] += 5 if new_position == self.goal_position else -1 if (
                                self.grid[new_position] == 'X' or new_position in self.pos_random_traps) else -0.01
                    elif self.grid[new_position] == 'W':
                        # hits wall
                        connections[action] = 0.0
                        rewards[action] = -0.1
                        rewards[action] += rewards[len(ACTION_NAMES)]
            connections = torch.tensor([connections[i] for i in range(len(connections))])
            rewards = torch.tensor([rewards[i] for i in range(len(rewards))])
            self.agent_position = temp_agent_position
            return observation, terminated, temp_iter_coord, connections, rewards, self.optimal_policy[temp_iter_coord]

        self.agent_position = temp_agent_position
        return None, None, temp_iter_coord, None, None, self.optimal_policy[temp_iter_coord]

    def iter_reset(self):
        self.iter_index = 0
        self.iter_coord = (0, 0)

    def directed_reachable_positions(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = position
        potential_neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]  # Neighbors are up, down, left, right
        reachable_positions = []
        if self.grid[x, y] in ['W'] or position == self.goal_position:
            return reachable_positions
        for potential_neighbour in potential_neighbours:
            if 0 <= potential_neighbour[0] < self.grid.shape[0] and 0 <= potential_neighbour[1] < self.grid.shape[1]:
                if self.grid[potential_neighbour[0], potential_neighbour[1]] not in ['W']:
                    reachable_positions.append(potential_neighbour)
        return reachable_positions

    def make_directed_graph(self, filepath: None or str = None, show=False):
        graph = nx.DiGraph()
        node_colors = {}  # Dictionary to store colors keyed by node

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                node = (x, y)
                if self.grid[x, y] not in ['W']:
                    graph.add_node(node)
                    for neighbor in self.directed_reachable_positions(node):
                        graph.add_edge(node, neighbor)

                # Assign colors based on conditions, directly using the node as a key
                if self.grid[x, y] in ['X'] or node in self.pos_random_traps:
                    node_colors[node] = 'red'  # Trap position
                elif node == self.goal_position or node == self._goal_position:
                    node_colors[node] = 'green'  # Goal position
                else:
                    node_colors[node] = 'blue'  # Default color for other nodes

        if filepath is not None:
            # Apply the colors when drawing
            colors = [node_colors.get(node, 'blue') for node in graph.nodes()]
            pos = nx.kamada_kawai_layout(graph)  # Positions for all nodes
            node_sizes = [100 for n in graph.nodes()]
            nx.draw(graph, pos, with_labels=True, arrows=True, node_color=colors, node_size=node_sizes, font_size=8, )
            # plt.show()
            plt.savefig(filepath, dpi=600)  # Set the resolution with the `dpi` argument
            plt.close()

        if show:
            # Apply the colors when drawing
            colors = [node_colors.get(node, 'blue') for node in graph.nodes()]
            pos = nx.kamada_kawai_layout(graph)  # Positions for all nodes
            node_sizes = [100 for n in graph.nodes()]
            nx.draw(graph, pos, with_labels=True, arrows=True, node_color=colors, node_size=node_sizes, font_size=8, )
            plt.show()

        return graph

    def load_grid(self, text_file):
        with open(text_file, 'r') as file:
            lines = file.read().splitlines()
        self.grid = np.array([list(line) for line in lines])
        return self.grid

    def reset_positions(self, agent_only=False):
        self.occupied_positions = set()

        self.agent_position = self._agent_position if self._agent_position else self.assign_position()
        self.occupied_positions.add(self.agent_position)

        if agent_only:
            return

        self.goal_position = self._goal_position if self._goal_position else self.assign_position()
        self.occupied_positions.add(self.goal_position)

        self.pos_random_traps = []
        for i in range(random.randint(0, self.num_random_traps)):
            try:
                trap = self.assign_position()
                self.pos_random_traps.append(trap)
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

        terminated = self.agent_position == self.goal_position  # or self.grid[self.agent_po10.70.27.253sition] == 'X'
        truncated = False
        reward = 5 if self.agent_position == self.goal_position else -1 if (
                self.grid[self.agent_position] == 'X' or self.agent_position in self.pos_random_traps) else -0.01
        if hits_wall:
            reward -= 0.1

        if self.grid[self.agent_position] in ['W']:
            print('traj:', self._traj)
            print(len(self._traj))
            print(self.step_count, self.max_steps)
            print(self.grid)
            print(self.agent_position, self.grid[self.agent_position])
            raise AssertionError

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
            reward = 0

        self._traj.append(action)
        self._traj.append(self.agent_position)
        return observation, reward, terminated, truncated, {"position": self.agent_position,
                                                            "goal_position": self.goal_position,
                                                            "optimal_policy": self.optimal_policy[
                                                                self.agent_position].item()}

    def reset(self, seed=None, options=None, no_random=False, compute_optimal_policy=False):
        super().reset(seed=seed)
        self.step_count = 0
        if not no_random:
            if self.random:
                # Randomly rotate the grid
                self.grid, (self._agent_position, self._goal_position, self.agent_position, self.goal_position) \
                    = rotate_grid(self.grid,
                                  [self._agent_position, self._goal_position, self.agent_position, self.goal_position])
                # Randomly flip the grid
                self.grid, (self._agent_position, self._goal_position, self.agent_position, self.goal_position) \
                    = flip_grid(self.grid,
                                [self._agent_position, self._goal_position, self.agent_position, self.goal_position])
            self.reset_positions()
        else:
            self.reset_positions(agent_only=True)

        assert self.grid[self.agent_position] not in ['W', 'X', 'G']
        self._traj = []
        self._traj.append(self.agent_position)

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0

        # get optimal policy
        if compute_optimal_policy:
            self.update_value_and_optimal_policy()

        return observation, {"position": self.agent_position, "goal_position": self.goal_position,
                             "optimal_policy": self.optimal_policy[self.agent_position]}

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
                elif cell_content == 'X' or (y, x) in self.pos_random_traps:
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
            if self.window is None:
                self.window = pygame.display.set_mode(self.screen_size)

            if self.cached_surface is not None:
                self.window.blit(self.cached_surface, (0, 0))
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
                        row += 'W'
                    elif self.grid[y, x] == 'X':
                        row += 'X'
                    else:
                        row += ' '
                print(row)

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None

    def handle_keyboard_input(self) -> int:
        action = None
        while action is None:
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

    def action_transition_and_reward(self, coord: tuple[int, int]) -> tuple[tuple[tuple[int, int], float], ...]:
        # Define deltas for 4 actions: up, down, left, right
        actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        transitions: list[tuple[tuple[int, int], float]] = []

        for action, delta in actions.items():
            # Compute new position based on the action
            new_position = (coord[0] + delta[0], coord[1] + delta[1])
            if (0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]):
                # Check for walls and special areas
                if self.grid[new_position] == 'W':
                    # If it's a wall, stay in place, apply wall collision penalty
                    reward = -0.1
                    transitions.append((coord, reward))
                else:
                    # If it's not a wall, check for goal and traps
                    if new_position == self.goal_position:
                        reward = 5.0
                    elif new_position in self.pos_random_traps or self.grid[new_position] == 'X':
                        reward = -1.0
                    else:
                        reward = -0.01
                    transitions.append((new_position, reward))
            else:
                # If out of bounds, stay in place, no extra penalty
                transitions.append((coord, -0.01))

        return tuple(transitions)

    # def update_value_and_optimal_policy(self, gamma=0.99,
    #                                     threshold=0.01,
    #                                     max_iterations=16384, ):
    #
    #     for iteration in range(max_iterations):
    #         delta = 0
    #         new_value_grid = np.copy(self.value_grid)
    #
    #         for i in range(self.grid.shape[0]):
    #             for j in range(self.grid.shape[1]):
    #                 if self.grid[i, j] == 'W' or self.grid[i, j] == 'X':  # Assuming 'X' is also an undesirable state
    #                     continue
    #                 max_value = float('-inf')
    #                 for action, (di, dj) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
    #                     ni, nj = i + di, j + dj
    #                     if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1] and self.grid[ni, nj] != 'W':
    #                         # Calculate the reward for moving to the new cell
    #                         reward = 5 if (ni, nj) == self.goal_position else -1 if self.grid[ni, nj] == 'X' or (
    #                             ni, nj) in self.pos_random_traps else -0.01
    #                         value = reward + gamma * self.value_grid[ni, nj]
    #                     else:
    #                         value = -0.01 + gamma * self.value_grid[
    #                             i, j]  # Stay in place if moving out of bounds or into a wall
    #
    #                     if value > max_value:
    #                         max_value = value
    #                         self.optimal_policy[i, j] = action
    #
    #                 new_value_grid[i, j] = max_value
    #                 delta = max(delta, np.max(np.abs(new_value_grid[i, j] - self.value_grid[i, j])))
    #
    #         self.value_grid = new_value_grid
    #
    #         if delta < threshold:
    #             break

    def update_value_and_optimal_policy(self, gamma=0.99, threshold=0.00001, max_iterations=16384):
        # First loop: Value iteration
        for iteration in range(max_iterations):
            delta = 0
            new_value_grid = np.copy(self.value_grid)

            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] == 'W' or self.grid[i, j] == 'X' or (i, j) == self.goal_position:
                        continue

                    action_values = []

                    for action, (di, dj) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1] and self.grid[ni, nj] != 'W':
                            reward = 5 if (ni, nj) == self.goal_position else -1 if (self.grid[ni, nj] == 'X' or (
                                ni, nj) in self.pos_random_traps) else -0.01
                            value = reward + gamma * self.value_grid[ni, nj]
                        else:
                            reward = -0.01 - 0.1
                            value = reward + gamma * self.value_grid[i, j]

                        action_values.append(value)

                    new_value_grid[i, j] = max(action_values)
                    delta = max(delta, np.abs(new_value_grid[i, j] - self.value_grid[i, j]))

            new_value_grid[self.goal_position] = 0  # Ensure the goal position value is locked to 0
            self.value_grid = new_value_grid

            if delta < threshold:
                break

        # Second loop: Calculate optimal policy based on the converged value grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 'W' or self.grid[i, j] == 'X':
                    continue

                action_values = []
                values = []

                for action, (di, dj) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1] and self.grid[ni, nj] != 'W':
                        reward = 5 if (ni, nj) == self.goal_position else -1 if self.grid[ni, nj] == 'X' or (
                            ni, nj) in self.pos_random_traps else -0.01
                        value = reward + gamma * self.value_grid[ni, nj]
                    else:
                        value = -0.01 + gamma * self.value_grid[i, j]

                    action_values.append((action, value))
                    values.append(value)

                max_value = max(values)
                best_actions = [action for action, value in action_values if abs(max_value - value) < threshold]

                prob = 1 / len(best_actions)
                self.optimal_policy[i, j] = np.zeros(4)  # Reset probabilities
                for action in best_actions:
                    self.optimal_policy[i, j, action] = prob

    # def get_optimal_policy_str(self):
    #     policy_map = ''
    #     for y in range(self.grid.shape[0]):
    #         row = ''
    #         for x in range(self.grid.shape[1]):
    #             if (y, x) == self.agent_position:
    #                 row += 'A    '
    #             elif (y, x) == self.goal_position:
    #                 row += 'G    '
    #             elif self.grid[y, x] == 'W':
    #                 row += 'W    '
    #             elif self.grid[y, x] == 'X':
    #                 row += 'X    '
    #             else:
    #                 row += ACTION_NAMES[self.optimal_policy[y, x].item()][0]
    #         policy_map += row + '\n'
    #     return policy_map

    def get_optimal_policy_str(self):
        policy_map = ''
        for y in range(self.grid.shape[0]):
            row = ''
            for x in range(self.grid.shape[1]):
                if (y, x) == self.agent_position:
                    row += 'A    '
                elif (y, x) == self.goal_position:
                    row += 'G    '
                elif self.grid[y, x] == 'W':
                    row += 'W    '
                elif self.grid[y, x] == 'X':
                    row += 'X    '
                else:
                    action_probabilities = self.optimal_policy[y, x]
                    max_prob = np.max(action_probabilities)
                    best_actions = [ACTION_NAMES[i][0] for i, prob in enumerate(action_probabilities) if
                                    prob == max_prob]
                    row += '/'.join(best_actions) + '    '
            policy_map += row + '\n'
        return policy_map


    def update_control_info(self, gamma=0.99, threshold=0.00001, max_iterations=16384):
        self.update_value_and_optimal_policy(gamma, threshold, max_iterations)
        for iteration in range(max_iterations):
            delta = 0
            new_control_info_grid = np.copy(self.control_info_grid)
            new_info_to_go_grid = np.copy(self.info_to_go_grid)

            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] == 'W' or self.grid[i, j] == 'X' or (i, j) == self.goal_position:
                        continue

                    action_values = []

                    for action, (di, dj) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
                        ni, nj = i + di, j + dj
                        prior = np.ones_like(self.optimal_policy[i, j]) / len(self.optimal_policy[i, j])
                        control_info = np.sum(self.optimal_policy[i, j] * np.log2(self.optimal_policy[i, j] / prior))
                        if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1] and self.grid[ni, nj] != 'W':
                            # reward = 5 if (ni, nj) == self.goal_position else -1 if (self.grid[ni, nj] == 'X' or (
                            #     ni, nj) in self.pos_random_traps) else -0.01
                            # value = reward + gamma * self.value_grid[ni, nj]
                            control_info = np.sum(self.optimal_policy[i, j] * np.log2(self.optimal_policy[i, j] / prior))
                        else:
                            # reward = -0.01 - 0.1
                            # value = reward + gamma * self.value_grid[i, j]

                        action_values.append(value)

                    new_value_grid[i, j] = max(action_values)
                    delta = max(delta, np.abs(new_value_grid[i, j] - self.value_grid[i, j]))

            new_value_grid[self.goal_position] = 0  # Ensure the goal position value is locked to 0
            self.value_grid = new_value_grid

            if delta < threshold:
                break


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
    # Add a batch dimensionget_optimal_policy_str
    processed_tensor = processed_tensor.unsqueeze(0)
    return processed_tensor


class SimpleGridWorldWithStateAbstraction(gymnasium.Env):
    def __init__(self, simple_gridworld: SimpleGridWorld, clusters: set[frozenset[tuple[int, int]]],
                 cluster_action_dict: dict = None):
        super(SimpleGridWorldWithStateAbstraction, self).__init__()
        self.simple_gridworld = simple_gridworld
        self.cluster_action_dict = cluster_action_dict

        # get clusters
        def cluster_sort_key(cluster):
            return min(cluster)

        sorted_clusters = sorted(clusters, key=cluster_sort_key)
        self.clusters_in_dict = {i: group for i, group in enumerate(sorted_clusters)}
        self.position_to_cluster = {}
        for i in range(self.simple_gridworld.grid.shape[0]):
            for j in range(self.simple_gridworld.grid.shape[1]):
                for idx in self.clusters_in_dict.keys():
                    if (i, j) in self.clusters_in_dict[idx]:
                        self.position_to_cluster[(i, j)] = idx

        # cancel randomization
        # start position can still be random
        # self.simple_gridworld._agent_position = self.simple_gridworld.agent_position
        self.simple_gridworld.__goal_position = self.simple_gridworld.goal_position
        self.simple_gridworld.random_traps = 0
        self.simple_gridworld.random = False
        self.reset()

    def step(self, action):
        self.simple_gridworld.step_count += 1
        if self.simple_gridworld.step_count >= self.simple_gridworld.max_steps:
            terminated = True
            truncated = True
            reward = 0
        else:
            deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            delta = deltas[action]
            new_position = (
                self.simple_gridworld.agent_position[0] + delta[0], self.simple_gridworld.agent_position[1] + delta[1])

            previous_position = self.simple_gridworld.agent_position

            hits_wall = False
            if 0 <= new_position[0] < self.simple_gridworld.shape[0] and 0 <= new_position[1] < \
                    self.simple_gridworld.grid.shape[1]:
                if self.simple_gridworld.grid[new_position] not in ['W']:
                    self.simple_gridworld.agent_position = new_position
                elif self.simple_gridworld.grid[new_position] == 'W':
                    hits_wall = True

            terminated = self.simple_gridworld.agent_position == self.simple_gridworld.goal_position
            truncated = False
            reward = 5 if self.simple_gridworld.agent_position == self.simple_gridworld.goal_position else -1 if (
                    self.simple_gridworld.grid[
                        self.simple_gridworld.agent_position] == 'X' or self.simple_gridworld.agent_position in self.simple_gridworld.pos_random_traps) else 0.0
            if hits_wall:
                reward -= 0.1
                reward -= 0.01
            else:
                new_position = self.simple_gridworld.agent_position
                momentum = np.array(new_position, dtype=np.float32) - np.array(previous_position, dtype=np.float32)
                new_position_group = self.position_to_cluster[new_position]
                old_position_group = self.position_to_cluster[previous_position]
                # print(f"group {old_position_group} -> group {new_position_group}")

                # get random new position
                rand_new_position = new_position
                if old_position_group == new_position_group and (
                        self.cluster_action_dict is None or self.cluster_action_dict[previous_position] == action):
                    radian_th = 0.01
                    radian = 3.14159265 * 0.5 * radian_th  # expected radian
                    rand_candidates = [pos for pos in self.clusters_in_dict[new_position_group]]
                    random.shuffle(rand_candidates)
                    for pos in rand_candidates:
                        rand_momentum = np.array(pos) - np.array(previous_position)
                        # compute radian:
                        # Calculate the dot product
                        dot_product = np.dot(momentum, rand_momentum)
                        # Calculate the magnitudes
                        magnitude_a = np.linalg.norm(momentum)
                        magnitude_b = np.linalg.norm(rand_momentum)
                        # if moving only one block or not moving at all:
                        if magnitude_b == 0:
                            continue
                        # compute the cosine of the angle
                        cos_angle = dot_product / (magnitude_a * magnitude_b)
                        # Calculate the angle in radians
                        angle_radians = np.arccos(cos_angle)
                        if angle_radians <= radian:
                            passed_grids = get_traversed_grids(np.array(pos), np.array(previous_position))
                            continue_outer = False

                            # generally avoid flying over someone not in the cluster.
                            for passed_grid in passed_grids:
                                if passed_grid not in rand_candidates or self.simple_gridworld.grid[passed_grid] in [
                                    'W']:
                                    continue_outer = True
                                    break
                            if continue_outer:
                                continue

                            distance_to_old = np.linalg.norm(np.array(pos) - np.array(previous_position))
                            distance_to_new = np.linalg.norm(np.array(pos) - np.array(new_position))
                            if distance_to_old >= distance_to_new:
                                rand_new_position = pos
                                # print(momentum, rand_momentum, general_momentum)
                                reward -= 0.01 * np.sum(np.abs(rand_momentum))
                                for passed_grid in passed_grids:
                                    if passed_grid in self.simple_gridworld.pos_random_traps or \
                                            self.simple_gridworld.grid[passed_grid] in ['X']:
                                        reward -= 1
                                break
                # print(f"pos {previous_position} -> pos {rand_new_position}")
                self.simple_gridworld.agent_position = rand_new_position

        self.simple_gridworld._render_to_surface()
        observation = self.simple_gridworld.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0
        # print(reward)
        return observation, reward, terminated, truncated, {"position": self.simple_gridworld.agent_position}

    def reset(self, seed=None, options=None):
        return self.simple_gridworld.reset(seed, options, no_random=True)

    def render(self, mode="rgb_array"):
        return self.simple_gridworld.render(mode)

    def close(self):
        self.simple_gridworld.close()

    def handle_keyboard_input(self):
        return self.simple_gridworld.handle_keyboard_input()


if __name__ == "__main__":
    env = SimpleGridWorld('envs/simple_grid/gridworld-maze-13.txt', make_random=True, random_traps=0,
                          agent_position=None, goal_position=(1, 8), max_steps=1024, )
    obs, info = env.reset()
    env.update_value_and_optimal_policy()
    print('optimal policy:')
    print(env.get_optimal_policy_str())
    print(env.value_grid)
    print(env.optimal_policy)
    # coord = info['position']
    # done = False
    # reward_sum = 0
    # while not done:
    #     env.render(mode='console')
    #     action = env.optimal_policy[coord]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     coord = info['position']
    #     reward_sum += reward
    #     if terminated:
    #         print("Game Over. Reward:", reward_sum)
    #         done = True
    #     elif truncated:
    #         print("Episode truncated.")
    #         done = True
    #     # time.sleep(0.1)
    # env.close()

    # from stable_baselines3 import PPO
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # from behaviour_tracer import BaselinePPOSimpleGridBehaviourIterSampler
    #
    # dummy_env = DummyVecEnv([lambda: env])
    # prior_agent = PPO.load("saved-models/simple-gridworld-ppo-prior-149.zip", env=env, verbose=1)
    # agent = PPO.load("saved-models/simple-gridworld-ppo-149.zip", env=env, verbose=1)
    # sampler = BaselinePPOSimpleGridBehaviourIterSampler(env, agent, prior_agent, reset_env=True)
    # sampler.sample()
    # clusters, cluster_action_dict = sampler.make_clusters(64, return_actions_dict=True)
    # # print(clusters)
    # sampler.plot_classified_grid("./gridworld-maze-31.gif", None, None, clusters=clusters)
    # env = SimpleGridWorldWithStateAbstraction(env, clusters, )  # cluster_action_dict)
    # # env.make_directed_graph(show=True)
    # # exit()
    # obs = env.reset()
    # done = False
    # reward_sum = 0
    # while not done:
    #     env.render(mode='human')
    #     action = env.handle_keyboard_input()
    #     if action is not None:
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         reward_sum += reward
    #         if terminated:
    #             print("Game Over. Reward:", reward_sum)
    #             done = True
    #         elif truncated:
    #             print("Episode truncated.")
    #             done = True
    #     # time.sleep(0.1)
    # env.close()
