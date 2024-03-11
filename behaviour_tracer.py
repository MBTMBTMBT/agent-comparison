import numpy as np
from simple_gridworld import SimpleGridWorld
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
        self.delta_control_info_tensor: torch.Tensor = torch.abs(self.action_distribution * torch.log2(self.action_distribution / self.prior_action_distribution))
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


class SimpleGridDeltaInfo:
    def __init__(
            self,
            env: SimpleGridWorld,
    ):
        self.env = env
        self.dict_record = {}
        self.delta_info_grid = torch.tensor(np.zeros_like(self.env.grid, dtype=np.float32))

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
        delta_control_info_tensor: torch.Tensor = torch.abs(action_distribution * torch.log2(action_distribution / prior_action_distribution))
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

    def plot_grid(self):
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

        plt.show()

    def plot_actions(self):
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
            if info['terminated'] or self.env.grid[(i, j)] == 'X':
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

        plt.show()


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

    def plot_grid(self):
        self.record.plot_grid()

    def plot_actions(self):
        self.record.plot_actions()


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
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-traps-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": (1, 5),
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 256,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-corridors-traps-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": (1, 11),
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
    ]

    env_fns = [partial(make_env, config) for config in test_env_configurations]
    env = DummyVecEnv(env_fns)
    prior_agent = PPO.load("saved-models/simple-gridworld-ppo-36.zip", env=env, verbose=1)
    agent = PPO.load("saved-models/simple-gridworld-ppo-48.zip", env=env, verbose=1)

    for config in test_env_configurations:
        test_env = make_env(config)
        sampler = BaselinePPOSimpleGridBehaviourIterSampler(test_env, agent, prior_agent)
        sampler.sample()
        sampler.plot_grid()
        sampler.plot_actions()
