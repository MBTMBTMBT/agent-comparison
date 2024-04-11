import gymnasium as gym
import torch
from torch.utils.data import Dataset


class TransitionBuffer(Dataset):
    def __init__(self, transition_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.transition_pairs = transition_pairs

    def __len__(self):
        return len(self.transition_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transition_pairs[idx]


class RandomSampler:
    def __init__(self) -> None:
        self.transition_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def sample(self, env: gym.Env, max_step=4096) -> tuple[int, int]:
        current_size = len(self.transition_pairs)
        obs, info = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = env.action_space.sample()
            if action is not None:
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.transition_pairs.append((obs, torch.tensor(action), torch.tensor(next_obs)))
                obs = next_obs
                reward_sum += reward
                if terminated:
                    done = True
                elif truncated:
                    done = True
                max_step -= 1
                if max_step <= 0:
                    done = True
        env.close()
        increased_size = len(self.transition_pairs) - current_size
        new_size = len(self.transition_pairs)
        return increased_size, new_size


class SamplerWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SamplerWrapper, self).__init__(env)
        self.env = env
        self.transition_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.current_obs: torch.Tensor or None = None

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.transition_pairs.append((self.current_obs, torch.tensor(action), torch.tensor(next_obs)))
        self.current_obs = next_obs
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_obs = observation
        return observation, info
