from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from simple_gridworld import TextGridWorld
import gymnasium


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


def make_env(configure: dict) -> gymnasium.Env:
    env = None
    if configure["env_type"] == "SimpleGridworld":
        if "cell_size" in configure.keys():
            cell_size = configure["cell_size"]
        else:
            cell_size = (20, 20)
        if "obs_size" in configure.keys():
            obs_size = configure["obs_size"]
        else:
            obs_size = (128, 128)
        if "agent_position" in configure.keys():
            agent_position = configure["agent_position"]
        else:
            agent_position = None
        if "goal_position" in configure.keys():
            goal_position = configure["goal_position"]
        else:
            goal_position = None
        env = TextGridWorld(text_file=configure["env_file"], cell_size=cell_size, obs_size=obs_size, agent_position=agent_position, goal_position=goal_position)
    return env


if __name__ == "__main__":
    from utils import *
    from simple_gridworld import ACTION_NAMES
    from functools import partial

    env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-5.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-5.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-two-rooms-5.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None
        },
    ]
    env_fns = [partial(make_env, config) for config in env_configurations]

    env = DummyVecEnv(env_fns)

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
