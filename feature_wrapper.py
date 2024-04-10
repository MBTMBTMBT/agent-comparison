import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper
import gymnasium as gym
import numpy as np


# class FeatureWrapper(VecEnvWrapper):
#     def __init__(self, venv, feature_extractor: torch.nn.Module, obs_shape: tuple[int], device=torch.device('cpu')):
#         super(FeatureWrapper, self).__init__(venv)
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
#         self.feature_extractor = feature_extractor.to(device)
#         self.device = device
#
#     def reset(self):
#         obs = self.venv.reset()
#         obs = self.process_obs(obs)
#         return obs
#
#     def step_wait(self):
#         obs, rewards, dones, infos = self.venv.step_wait()
#         obs = self.process_obs(obs)
#         return obs, rewards, dones, infos
#
#     def process_obs(self, obs):
#         obs = torch.tensor(obs, device=self.device).float()
#         with torch.no_grad():
#             obs = self.feature_extractor(obs).cpu()
#         return obs

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_extractor: torch.nn.Module, features_dim: int, device=torch.device('cpu')):
        # The output dimensions of your feature extractor
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        self._feature_extractor = feature_extractor.to(device)
        self.device = device

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device).float()
        with torch.no_grad():
            return self._feature_extractor(observations).to(self.device)
