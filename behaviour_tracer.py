import os

import imageio
from PIL import Image
import numpy as np
from simple_gridworld import SimpleGridWorld
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import BytesIO
import networkx as nx

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


def non_zero_softmax(x: torch.Tensor, epsilon=1e-10) -> torch.Tensor:
    x += epsilon
    exp_x = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x


class TimeStep:
    def __init__(
            self,
            obs: torch.Tensor,
            action: int,
            action_distribution: torch.Tensor,
            prior_action: int,
            prior_action_distribution: torch.Tensor,
            reward: float,
            done: bool,
            info: dict,
            previous_control_info_sum: float = 0.0,
    ):
        self.obs = obs
        self.action = action
        self.action_distribution = non_zero_softmax(action_distribution)
        self.prior_action = prior_action
        self.prior_action_distribution = non_zero_softmax(prior_action_distribution)
        self.reward = reward
        self.done = done
        self.info = info

        # get control information
        self.delta_control_info_tensor: torch.Tensor = self.action_distribution * torch.log2(
            self.action_distribution / self.prior_action_distribution)
        self.delta_control_info: float = float(torch.sum(self.delta_control_info_tensor))
        self.control_info_sum = previous_control_info_sum + self.delta_control_info

    def to_dict(self) -> dict:
        return {
            'obs': self.obs.tolist(),
            'action': self.action,
            'action_distribution': self.action_distribution.tolist(),
            'prior_action': self.prior_action,
            'prior_action_distribution': self.prior_action_distribution.tolist(),
            'reward': self.reward,
            'done': self.done,
            'info': self.info,
            'delta_control_info_tensor': self.delta_control_info_tensor.tolist(),
            'delta_control_info': self.delta_control_info,
            'control_info_sum': self.control_info_sum,
        }


class BehaviourTrajectory:
    def __init__(
            self,
    ):
        self.trajectory: list[TimeStep] = []

    def add_trajectory(
            self,
            obs: torch.Tensor,
            action: int,
            action_distribution: torch.Tensor,
            prior_action: int,
            prior_action_distribution: torch.Tensor,
            reward: float,
            done: bool,
            info: dict,
            previous_control_info_sum: float = 0.0,
    ):
        time_step = TimeStep(
            obs,
            action,
            action_distribution,
            prior_action,
            prior_action_distribution,
            reward,
            done,
            info,
            previous_control_info_sum
        )
        self.trajectory.append(time_step)

    def conclude_trajectory(self) -> dict:
        conclusion = {
            'obs': [],
            'action': [],
            'action_distribution': [],
            'prior_action': [],
            'prior_action_distribution': [],
            'reward': [],
            'done': [],
            'info': [],
            'delta_control_info_tensor': [],
            'delta_control_info': [],
            'control_info_sum': [],
        }
        for time_step in self.trajectory:
            dict_time_tep = time_step.to_dict()
            for key in dict_time_tep.keys():
                conclusion[key].append(dict_time_tep[key])

        return conclusion


def _merge(pair_to_merge: frozenset, clusters: set[frozenset[tuple[int, int]]]) -> set[frozenset[tuple[int, int]]]:
    u, v = pair_to_merge
    u_group, v_group = None, None
    for group in clusters:
        if u in group:
            u_group = group
        if v in group:
            v_group = group
        if u_group is not None and v_group is not None:
            new_clusters = clusters - {u_group, v_group}  # Remove the old groups
            new_clusters.add(u_group.union(v_group))  # Add the merged group
            return new_clusters
    return clusters


class SimpleGridDeltaInfo:
    def __init__(
            self,
            env: SimpleGridWorld,
    ):
        self.env = env
        self.dict_record = {}
        self.delta_info_grid = torch.tensor(np.zeros_like(self.env.grid, dtype=np.float32))
        self.num_actions: int = self.env.num_actions
        self.delta_info_times_action_distribution = torch.zeros(size=self.delta_info_grid.shape + (self.num_actions,),
                                                                dtype=torch.float32)
        self.grid_feature_vectors = torch.zeros(size=self.delta_info_grid.shape + (self.num_actions,),
                                                dtype=torch.float32)
        self.available_positions: list[tuple[int, int]] = []

    def add(
            self,
            action: int,
            action_distribution: torch.Tensor,
            prior_action: int,
            prior_action_distribution: torch.Tensor,
            terminated: bool,
            position: tuple[int, int],
    ):
        action_distribution = non_zero_softmax(action_distribution)
        prior_action_distribution = non_zero_softmax(prior_action_distribution)
        delta_control_info_tensor: torch.Tensor = action_distribution * torch.log2(
            action_distribution / prior_action_distribution)
        delta_control_info: float = float(torch.sum(delta_control_info_tensor))
        self.dict_record[position] = {
            "action": action,
            "action_distribution": action_distribution,
            "prior_action": prior_action,
            "prior_action_distribution": prior_action_distribution,
            "terminated": terminated,
            "delta_control_info_tensor": delta_control_info_tensor,
            "delta_control_info": delta_control_info
        }
        self.delta_info_grid[position] = delta_control_info
        self.delta_info_times_action_distribution[position, :] = delta_control_info * action_distribution
        self.available_positions.append(position)
        self.grid_feature_vectors[position, :] = self.delta_info_times_action_distribution[position, :]

    def _compute_distances(self) -> tuple[
        list[float or np.inf], dict[int, tuple[tuple[int, int], tuple[int, int]]], dict[
            tuple[tuple[int, int], tuple[int, int]], int]]:
        grid_feature_vectors = self.grid_feature_vectors.detach().cpu().numpy()

        # Extract vectors from available positions
        vectors = np.array([grid_feature_vectors[x, y] for x, y in self.available_positions])

        # Compute pairwise distances using broadcasting
        diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Flatten the upper triangle of the distance matrix to get the condensed distance vector
        i_upper = np.triu_indices(len(self.available_positions), k=1)
        distances = distances[i_upper].tolist()

        # Create mapping dictionaries
        serial_to_pairs = {}
        pairs_to_serial = {}
        for i, (pos1_idx, pos2_idx) in enumerate(zip(*i_upper)):
            pos1: tuple[int, int] = self.available_positions[pos1_idx]
            pos2: tuple[int, int] = self.available_positions[pos2_idx]

            # Map serial number to position pairs
            serial_to_pairs[i] = (pos1, pos2)

            # Map position pairs (as unordered) to serial number
            pairs_to_serial[(pos1, pos2)] = i
            pairs_to_serial[(pos2, pos1)] = i

        return distances, serial_to_pairs, pairs_to_serial

    def compute_distances(self) -> dict[frozenset, float]:
        grid_feature_vectors = self.grid_feature_vectors.detach().cpu().numpy()
        connection_graph = self.env.make_directed_graph()

        bidirectional_pairs = []
        for (u, v) in connection_graph.edges():
            if (v, u) in connection_graph.edges():
                bidirectional_pairs.append(frozenset((u, v)))

        distances = {}
        for u, v in bidirectional_pairs:
            vector_u = grid_feature_vectors[u]
            vector_v = grid_feature_vectors[v]
            distances[frozenset((u, v))] = np.sqrt(np.sum((vector_u - vector_v) ** 2)).item()

        return distances

    def compute_merge_sequence(self):
        distances = self.compute_distances()
        merge_sequence: list[frozenset] = sorted(distances, key=lambda k: distances[k])
        return merge_sequence

    def make_cluster(self, merge_sequence: list[frozenset], num_clusters_to_keep: int, ) -> set[
        frozenset[tuple[int, int]]]:
        graph = self.env.make_directed_graph()
        # init clusters
        clusters: set[frozenset[tuple[int, int]]] = set()
        for node in graph.nodes:
            clusters.add(frozenset([node]))  # Use frozenset for individual nodes
        while len(clusters) > num_clusters_to_keep:
            if not merge_sequence:
                print("Warning: More groups kept than expected.")
                break
            pair_to_merge = merge_sequence.pop(0)
            clusters = _merge(pair_to_merge, clusters)
        return clusters

    def _compute_linkages(self, return_info=False):
        distances, serial_to_pairs, pairs_to_serial = self.compute_distances()
        connection_graph = self.env.make_directed_graph()

        # Find bidirectionally connected pairs
        # bidirectional_pairs = {(u, v) for u, v in connection_graph.edges() if connection_graph.has_edge(v, u)}
        # bidirectional_pairs.update({(v, u) for u, v in connection_graph.edges() if connection_graph.has_edge(v, u)})
        bidirectional_pairs = []
        for (u, v) in connection_graph.edges():
            if (v, u) in connection_graph.edges():
                bidirectional_pairs.append((u, v))
                bidirectional_pairs.append((v, u))

        for i in serial_to_pairs.keys():
            if serial_to_pairs[i] not in bidirectional_pairs:
                a = serial_to_pairs[i]
                distances[i] = -1

        distances = np.array(distances, dtype=np.float32)

        max_distance = np.max(distances)
        large_distance_value = max_distance * 10  # Example: 10 times the max distance
        distances = np.where(distances == -1, large_distance_value, distances)

        # Perform hierarchical clustering
        links = linkage(distances, method='single')

        # Decide the number of clusters or a cutoff to form clusters
        # clusters = fcluster(links, t=1.5, criterion='distance')

        if return_info:
            return links, distances, serial_to_pairs, pairs_to_serial

        return links

    def plot_grid(self, filepath):
        height, width = self.delta_info_grid.shape
        fig, ax = plt.subplots()
        # Initialize with a black background and ensure alpha channel is set for opacity
        color_grid = np.zeros((height, width, 4))  # RGBA format
        color_grid[:, :, 3] = 1  # Set alpha channel to 1 for all grid cells

        delta_min, delta_max = 0, float(torch.max(self.delta_info_grid))
        norm = mcolors.Normalize(vmin=delta_min, vmax=delta_max)
        cmap = cm.get_cmap('Blues')

        for (i, j), info in self.dict_record.items():
            if self.env.grid[(i, j)] == 'X':  # Check if the cell is a trap
                color_grid[i, j] = [1, 0, 0, 1]  # Red for traps
            elif info['terminated']:
                color_grid[i, j] = [0, 1, 0, 1]  # Green for terminated states
            else:
                delta_info = info['delta_control_info']
                color_grid[i, j] = cmap(norm(delta_info))[:3] + (1,)

        ax.imshow(color_grid, origin='upper', extent=[0, width, 0, height])

        smappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(smappable, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Delta Control Info')

        # Now, including traps in the annotation process
        for (i, j), info in self.dict_record.items():
            if not info['terminated']:  # Annotations for all non-terminated states, including traps
                color = 'white' if self.env.grid[(i, j)] == 'X' else 'black'
                # Directly use (i, j) for text positioning under origin='upper'
                ax.text(j + 0.5, height - i - 0.5, f"{info['delta_control_info']:.2f}", ha='center', va='center',
                        color=color, fontsize=6)

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the figure
        plt.savefig(filepath, dpi=600)  # Set the resolution with the `dpi` argument
        plt.close()

    def plot_classified_grid(self, merge_sequence: list[frozenset], filepath, max_num_groups: int or None=None):
        # init clusters
        clusters: set[frozenset[tuple[int, int]]] = set()
        graph = self.env.make_directed_graph()
        for node in graph.nodes:
            clusters.add(frozenset([node]))  # Use frozenset for individual nodes
        if max_num_groups is None:
            max_num_groups = len(clusters)
        min_num_groups = 2
        frames = []  # List to store frames for the GIF
        previous_num_clusters = len(clusters)
        for g in range(max_num_groups, min_num_groups - 1, -1):  # Ensure correct loop direction
            if len(clusters) > g:
                if len(merge_sequence) == 0:
                    print("Warning: More groups kept than expected.")
                    break
                pair_to_merge = merge_sequence.pop(0)
                clusters = _merge(pair_to_merge, clusters)
            clusters_in_dict = {i: group for i, group in enumerate(clusters)}
            position_to_cluster = {}
            for (i, j), info in self.dict_record.items():
                for idx in clusters_in_dict.keys():
                    if (i, j) in clusters_in_dict[idx]:
                        position_to_cluster[(i, j)] = idx

            height, width = self.delta_info_grid.shape
            fig, ax = plt.subplots()
            color_grid = np.zeros((height, width, 4))  # RGBA format
            color_grid[:, :, 3] = 1  # Set alpha channel to 1 for all grid cells

            # Generate a color for each cluster
            num_clusters = len(clusters)
            colors = cm.get_cmap('gist_rainbow', num_clusters)

            # if previous_num_clusters == num_clusters:
            #     plt.close(fig)
            #     break
            #
            # previous_num_clusters = num_clusters

            for pos, cluster_label in position_to_cluster.items():
                color_grid[pos] = colors(cluster_label - 1)[:3] + (1,)  # Set RGB and keep alpha=1

            for (i, j), info in self.dict_record.items():
                if (i, j) in position_to_cluster:  # Keep original color if not in position_to_cluster
                    if self.env.grid[(i, j)] == 'X':  # or rand traps...
                        pass
                        # color_grid[i, j] = [1, 0, 0, 1]  # Red for traps
                    elif info['terminated']:
                        color_grid[i, j] = [0, 1, 0, 1]  # Green for terminated states
                else:
                    # Keep the original color based on delta_control_info
                    pass

            # Debugging: Directly set a few grid cells to check color assignment
            # for i in range(5):
            #     for j in range(5):
            #         color_grid[i, j] = [1, 0, 0, 1]  # Red, fully opaque

            ax.imshow(color_grid, origin='upper', extent=[0, width, 0, height])
            ax.set_title(f'Total Classes: {num_clusters}')

            # Change label to group index
            for pos, cluster_label in position_to_cluster.items():
                ax.text(pos[1] + 0.5, height - pos[0] - 0.5, str(cluster_label), ha='center', va='center',
                        color='white', fontsize=6)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            plt.close(fig)

        # Create GIF
        imageio.mimsave(filepath, frames, fps=1)


    def plot_graph(self, filepath):
        graph = self.env.make_directed_graph()
        a = graph.edges()
        fig, ax = plt.subplots()
        # Assuming `graph` is a NetworkX graph where nodes are tuples (i, j)
        pos = {node: (node[1], -node[0]) for node in graph.nodes()}  # Inverting y to match image origin='upper'
        # pos = nx.kamada_kawai_layout(graph)

        # Initialize node color list and labels dict
        node_colors = []
        labels = {}

        delta_min, delta_max = 0, float(torch.max(self.delta_info_grid))
        norm = mcolors.Normalize(vmin=delta_min, vmax=delta_max)
        cmap = cm.get_cmap('Blues')

        for node in graph.nodes:
            i, j = node
            if self.env.grid[(i, j)] == 'X':
                node_colors.append([1, 0, 0, 1])  # Red for traps
                # Label with delta control info
                labels[node] = f"{self.dict_record[(i, j)]['delta_control_info']:.2f}"
            elif self.dict_record[(i, j)]['terminated']:
                node_colors.append([0, 1, 0, 1])  # Green for terminated states
            else:
                delta_info = self.dict_record[(i, j)]['delta_control_info']
                node_colors.append(cmap(norm(delta_info))[:3] + (1,))  # Color based on delta_info
                # Label with delta control info
                labels[node] = f"{self.dict_record[(i, j)]['delta_control_info']:.2f}"

        # Drawing
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=100, edgecolors='black')
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_color='black')

        # Colorbar for delta control info
        smappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(smappable, ax=ax, orientation='vertical', label='Delta Control Info')

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the figure
        plt.savefig(filepath, dpi=600)  # Set the resolution with the `dpi` argument
        plt.close()

    def plot_actions(self, filepath):
        height, width = self.delta_info_grid.shape
        fig, ax = plt.subplots()
        color_grid = np.zeros((height, width, 4))  # Initialize with a black background
        color_grid[:, :, 3] = 1  # Set alpha channel to 1

        delta_min, delta_max = 0, float(torch.max(self.delta_info_grid))
        norm = mcolors.Normalize(vmin=delta_min, vmax=delta_max)
        cmap = cm.get_cmap('Blues')

        # Draw background color
        for (i, j), info in self.dict_record.items():
            if self.env.grid[(i, j)] == 'X':  # Check for traps and mark them in red
                color_grid[i, j] = [1, 0, 0, 1]
            elif info['terminated']:
                color_grid[i, j] = [0, 1, 0, 1]
            else:
                color_grid[i, j] = cmap(norm(info['delta_control_info']))[:3] + (1,)

        ax.imshow(color_grid, origin='upper', extent=[0, width, 0, height])

        # Action mapping
        action_mapping = {
            0: (0, 1),  # Up
            1: (0, -1),  # Down
            2: (-1, 0),  # Left
            3: (1, 0),  # Right
        }

        # Scale factors for arrow size, adjust as needed
        scale_length = 0.5  # Adjust for overall arrow length
        scale_width = 0.05  # Adjust for overall arrow width

        # Draw arrows for agent and prior actions
        for (i, j), info in self.dict_record.items():
            if info['terminated']:  # or self.env.grid[(i, j)] == 'X':
                continue  # Skip arrows for terminated states and traps

            start_x = j + 0.5
            start_y = height - i - 0.5  # Adjusted for origin='upper'

            # Iterate through all actions for each cell for both agent and prior
            for action in range(4):
                # Use action probabilities to determine arrow length
                agent_action_prob = info['action_distribution'][action]
                prior_action_prob = info['prior_action_distribution'][action]

                dx, dy = action_mapping[action]

                # Agent action arrow
                if agent_action_prob > 0:  # Draw only if there is a non-zero probability
                    ax.arrow(start_x, start_y, dx * agent_action_prob * scale_length,
                             dy * agent_action_prob * scale_length, head_width=scale_width, head_length=scale_width,
                             fc='gold', ec='orange')

                # Prior action arrow
                if prior_action_prob > 0:  # Draw only if there is a non-zero probability
                    # Slightly offset prior arrows for visibility
                    # ax.arrow(start_x + dx * 0.1, start_y + dy * 0.1, dx * prior_action_prob * scale_length,
                    #          dy * prior_action_prob * scale_length, head_width=scale_width, head_length=scale_width,
                    #          fc='lightblue', ec='lightblue')
                    ax.arrow(start_x, start_y, dx * prior_action_prob * scale_length,
                             dy * prior_action_prob * scale_length, head_width=scale_width, head_length=scale_width,
                             fc='lightblue', ec='lightblue')

        # Add legend
        ax.plot([], [], color='orange', marker='>', markersize=10, label='Agent action', linestyle='None')
        ax.plot([], [], color='lightblue', marker='>', markersize=10, label='Prior action', linestyle='None')
        ax.legend(loc='upper right')

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the figure
        plt.savefig(filepath, dpi=600)  # Set the resolution with the `dpi` argument
        plt.close()


class BaselinePPOSimpleGridBehaviourIterSampler:
    def __init__(
            self,
            env: SimpleGridWorld,
            agent: PPO,
            prior_agent: PPO,
    ):
        self.env = env
        self.agent = agent
        self.prior_agent = prior_agent
        self.record = SimpleGridDeltaInfo(self.env)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self):
        for observation, terminated, position in self.env:
            if observation is not None and terminated is not None:
                action, _states = self.agent.predict(observation, deterministic=True)
                prior_action, _states = self.prior_agent.predict(observation, deterministic=True)

                dis = self.agent.policy.get_distribution(observation.unsqueeze(0).to(torch.device(self.device)))
                probs = dis.distribution.probs
                action_distribution = probs.to(torch.device('cpu')).squeeze()

                dis = self.prior_agent.policy.get_distribution(observation.unsqueeze(0).to(torch.device(self.device)))
                probs = dis.distribution.probs
                prior_action_distribution = probs.to(torch.device('cpu')).squeeze()

                self.record.add(
                    action.item(),
                    action_distribution.detach(),
                    prior_action.item(),
                    prior_action_distribution.detach(),
                    terminated,
                    position,
                )
        self.env.iter_reset()

    def plot_grid(self, filepath):
        self.record.plot_grid(filepath)

    def plot_actions(self, filepath):
        self.record.plot_actions(filepath)

    def plot_graph(self, filepath):
        self.record.plot_graph(filepath)

    def plot_classified_grid(self, filepath, max_num_groups: int or None=None):
        merge_sequence = self.record.compute_merge_sequence()
        self.record.plot_classified_grid(merge_sequence, filepath, max_num_groups=max_num_groups)


if __name__ == "__main__":
    from train_baseline import make_env
    from functools import partial
    from stable_baselines3.common.env_util import DummyVecEnv

    test_env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": (5, 5),
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-traps-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": (3, 3),
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-maze-traps-7.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": (1, 5),
        #     "num_random_traps": 0,
        #     "make_random": True,
        #     "max_steps": 256,
        # },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-corridors-traps-13.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": (1, 1),
        #     "num_random_traps": 0,
        #     "make_random": True,
        #     "max_steps": 128,
        # },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": (1, 1),
        #     "num_random_traps": 0,
        #     "make_random": True,
        #     "max_steps": 512,
        # },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": (1, 1),
        #     "num_random_traps": 0,
        #     "make_random": True,
        #     "max_steps": 512,
        # },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": (1, 1),
        #     "num_random_traps": 0,
        #     "make_random": True,
        #     "max_steps": 512,
        # },
    ]

    env_fns = [partial(make_env, config) for config in test_env_configurations]
    env = DummyVecEnv(env_fns)
    prior_agent = PPO.load("saved-models/simple-gridworld-ppo-prior-149.zip", env=env, verbose=1)
    agent = PPO.load("saved-models/simple-gridworld-ppo-149.zip", env=env, verbose=1)

    for config in test_env_configurations:
        test_env = make_env(config)
        sampler = BaselinePPOSimpleGridBehaviourIterSampler(test_env, agent, prior_agent)
        sampler.sample()
        sampler.plot_grid(f"results/{config['env_type']}_{config['env_file'].split('/')[-1]}_delta-info.png")
        sampler.plot_actions(f"results/{config['env_type']}_{config['env_file'].split('/')[-1]}_action.png")
        sampler.plot_graph(f"results/{config['env_type']}_{config['env_file'].split('/')[-1]}_graph.png")
        sampler.plot_classified_grid(f"results/{config['env_type']}_{config['env_file'].split('/')[-1]}_classification.gif", max_num_groups=None)
