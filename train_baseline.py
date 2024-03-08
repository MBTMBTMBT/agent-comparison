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
    from utils import *
    from simple_gridworld import ACTION_NAMES
    env_name = 'gridworld_empty.txt'

    env = DummyVecEnv([make_env, make_env, make_env, make_env,])

    policy_kwargs = dict(
        features_extractor_class=FlexibleImageEncoder,
        features_extractor_kwargs=dict(features_dim=64),  # output_size需要与你FlexibleImageEncoder中的fc层输出一致
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    for i in range(500):
        model.learn(total_timesteps=20000, progress_bar=True)
        model.save("ppo_textgridworld")

        # model = PPO.load("ppo_textgridworld")
        env = TextGridWorld(env_name)

        obs, _ = env.reset()
        # env.render(mode='human')
        terminated, truncated = False, False
        rendered, action, probs = None, None, None
        count = 0
        trajectory = []
        rewards = [0.0, ]
        while not (terminated or truncated):
            rendered = env.render(mode='rgb_array')
            action, _states = model.predict(obs, deterministic=False)
            dis = model.policy.get_distribution(obs.unsqueeze(0).to(torch.device('cuda')))
            probs = dis.distribution.probs
            probs = probs.to(torch.device('cpu')).squeeze()
            trajectory.append((rendered, action.item(), probs))
            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            count += 1
            if count >= 256:
                break
            # env.render(mode='human')
            # time.sleep(1)

        print("Test:", i, "Count:", count)
        if rendered is not None and action is not None and probs is not None:
            rendered = env.render(mode='rgb_array')
            trajectory.append((rendered, action.item(), probs))
        save_trajectory_as_gif(trajectory, rewards, ACTION_NAMES, filename=env_name + f"_trajectory_{i}.gif")

    # # 假设你的模型是model，环境是env
    # obs = env.reset()
    #
    # # 使用模型的策略网络直接处理观测
    # action, _states = model.predict(obs, deterministic=False)  # 获取动作
    #
    # # 要获取动作概率分布，你可以直接使用策略网络
    # action_probs = model.policy.forward(obs[None, :])  # 添加额外的批处理维度
    #
    # # action_probs是一个元组，包含(logits, values)等，具体取决于策略的类型
    # # 对于离散动作空间，你通常关注logits（即未归一化的概率对数）
    #
    # # 如果你想要概率分布，可以这样做：
    # distribution = model.policy.action_dist.proba_distribution(action_probs[0])
    # probs = distribution.probs  # 对于离散动作空间，获取概率分布
    #
    # # 对于连续动作空间，distribution对象将提供不同的属性和方法来访问分布的参数
    #
    # # 注意：这个示例假设你的环境只有一个观测和动作。
    # # 对于向量化环境，处理方法会略有不同，你可能需要对每个环境分别处理。
