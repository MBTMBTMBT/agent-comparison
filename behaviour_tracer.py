import numpy as np
from simple_gridworld import SimpleGridWorld
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


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
        self.delta_control_info_tensor: torch.Tensor = self.action_distribution * torch.log2(self.action_distribution / self.prior_action_distribution)
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
        self.delta_info_grid = torch.zeros_like(self.env.grid, dtype=np.float32)

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

    def plot_grid(self):
        height, width = self.delta_info_grid.shape
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.grid(which='both')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for i in range(height):
            for j in range(width):
                if self.env.grid[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
                elif (i, j) in self.dict_record and self.dict_record[(i, j)]['terminated']:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))
                elif (i, j) in self.dict_record:
                    info = self.dict_record[(i, j)]
                    color_value = 1.0 - min(info['delta_control_info'] / 10, 1.0)
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color=(color_value, color_value, color_value)))
                    ax.text(j + 0.5, i + 0.5, f"{info['delta_control_info']:.2f}",
                            horizontalalignment='center', verticalalignment='center', color='white')
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

        plt.gca().invert_yaxis()
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
                    action_distribution.detach().numpy(),
                    prior_action.item(),
                    prior_action_distribution.detach().numpy(),
                    terminated,
                    position,
                )
