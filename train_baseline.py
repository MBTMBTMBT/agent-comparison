import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium
import torch
from simple_gridworld import TextGridWorld
import torch.nn.functional as F


class FlexibleImageEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64):
        super(FlexibleImageEncoder, self).__init__(observation_space, features_dim=features_dim)

        input_channels = observation_space.shape[0]  # HxWxC
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, features_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Adaptive pooling allows for flexible input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def make_env():
    env = TextGridWorld(text_file="gridworld_empty.txt", cell_size=(20, 20))
    return env


if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        features_extractor_class=FlexibleImageEncoder,
        features_extractor_kwargs=dict(features_dim=64),  # output_size需要与你FlexibleImageEncoder中的fc层输出一致
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("ppo_textgridworld")

    model = PPO.load("ppo_textgridworld")
    env = TextGridWorld('gridworld_empty.txt')

    obs, _ = env.reset()
    env.render(mode='human')
    terminated, truncated = False, False
    count = 0
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        count += 1
        env.render(mode='human')
        time.sleep(1)

    print("Count:", count)
