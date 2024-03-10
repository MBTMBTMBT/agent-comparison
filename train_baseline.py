from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch.nn as nn
# import torch
from simple_gridworld import TextGridWorld
import gymnasium
# import os


# class FlexibleImageEncoder(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim: int = 64):
#         super(FlexibleImageEncoder, self).__init__(observation_space, features_dim=features_dim)
#
#         input_channels = observation_space.shape[0]  # HxWxC
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, features_dim, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         # Adaptive pooling allows for flexible input sizes
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(features_dim, features_dim)
#
#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         x = self.cnn(observations)
#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.fc(x)
#         return x


def make_env(configure: dict) -> gymnasium.Env:
    env = None
    if configure["env_type"] == "SimpleGridworld":
        if "cell_size" in configure.keys():
            cell_size = configure["cell_size"]
            if cell_size is None:
                cell_size = (20, 20)
        else:
            cell_size = (20, 20)
        if "obs_size" in configure.keys():
            obs_size = configure["obs_size"]
            if obs_size is None:
                obs_size = (128, 128)
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
        if "num_random_traps" in configure.keys():
            num_random_traps = configure["num_random_traps"]
        else:
            num_random_traps = 0
        if "make_random" in configure.keys():
            make_random = configure["make_random"]
        else:
            make_random = False
        if "max_steps" in configure.keys():
            max_steps = configure["max_steps"]
        else:
            max_steps = 128
        env = TextGridWorld(
            text_file=configure["env_file"],
            cell_size=cell_size,
            obs_size=obs_size,
            agent_position=agent_position,
            goal_position=goal_position,
            random_traps=num_random_traps,
            make_random=make_random,
            max_steps=max_steps
        )
    return env


def save_model(model, iteration, base_name="simple-gridworld-ppo", save_dir="saved-models"):
    """Save the model with a custom base name, iteration number, and directory."""
    model_path = os.path.join(save_dir, f"{base_name}-{iteration}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def find_newest_model(base_name="simple-gridworld-ppo", save_dir="saved-models"):
    """Find the most recently saved model based on iteration number and custom base name."""
    model_files = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith('.zip')]
    if not model_files:
        return None
    # Extracting iteration numbers and finding the latest model
    latest_model = max(model_files, key=lambda x: int(x.replace(base_name + '-', '').split('.')[0]))
    return os.path.join(save_dir, latest_model)


if __name__ == "__main__":
    from utils import *
    from simple_gridworld import ACTION_NAMES
    from functools import partial

    train_env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 256,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-two-rooms-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-many-rooms-9.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-many-rooms-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
    ]

    test_env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-traps-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
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
            "goal_position": None,
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
            "goal_position": None,
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
    ]

    env_fns = [partial(make_env, config) for config in train_env_configurations]

    env = DummyVecEnv(env_fns)

    # policy_kwargs = dict(
    #     features_extractor_class=FlexibleImageEncoder,
    #     features_extractor_kwargs=dict(features_dim=64),
    # )

    # dir names
    base_name = "simple-gridworld-ppo"
    save_dir = "saved-models"

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the newest model based on the custom base name and directory
    newest_model_path = find_newest_model(base_name=base_name, save_dir=save_dir)
    if newest_model_path:
        print(f"Loading model from {newest_model_path}")
        model = PPO.load(newest_model_path, env=env, verbose=1)
    else:
        print("Creating a new model")
        model = PPO("CnnPolicy", env, policy_kwargs={"normalize_images": False}, verbose=1)  # policy_kwargs=policy_kwargs,

    for i in range(100):
        model.learn(total_timesteps=500000, progress_bar=True)
        save_model(model, i, base_name, save_dir)

        for config in train_env_configurations + test_env_configurations:
            test_env = make_env(config)
            obs, _ = test_env.reset()
            terminated, truncated = False, False
            rendered, action, probs = None, None, None
            count = 0
            sum_reward = 0
            trajectory = []
            rewards = [0.0, ]
            while not (terminated or truncated):
                rendered = test_env.render(mode='rgb_array')
                action, _states = model.predict(obs, deterministic=False)
                dis = model.policy.get_distribution(obs.unsqueeze(0).to(torch.device('cuda')))
                probs = dis.distribution.probs
                probs = probs.to(torch.device('cpu')).squeeze()
                trajectory.append((rendered, action.item(), probs))
                obs, reward, terminated, truncated, info = test_env.step(action.item())
                rewards.append(reward)
                count += 1
                sum_reward += reward
                if count >= 256:
                    break

            print("Test:", i, f"Test on {config['env_file']} completed.", "Step:", count, "Reward:", sum_reward)
            if rendered is not None and action is not None and probs is not None:
                rendered = test_env.render(mode='rgb_array')
                trajectory.append((rendered, action.item(), probs))
            save_trajectory_as_gif(trajectory, rewards, ACTION_NAMES, filename=config["env_file"].split('/')[-1] + f"_trajectory_{i}.gif")
