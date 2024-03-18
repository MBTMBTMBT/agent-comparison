import copy
import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio
import stable_baselines3.common.on_policy_algorithm
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from behaviour_tracer import BaselinePPOSimpleGridBehaviourIterSampler
from simple_gridworld import SimpleGridWorld, SimpleGridWorldWithStateAbstraction


def find_latest_checkpoint(model_dir):
    """Find the latest model checkpoint in the given directory."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None

    # Extracting the epoch number from the model filename using regex
    checkpoints.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    return os.path.join(model_dir, checkpoints[-1])


def create_image_with_action(action_dict: dict[int, str], image, action, q_vals: torch.Tensor, step_number, reward):
    """
    Creates an image with the action text, Q values histogram, and additional details overlay.

    Parameters:
    - image: The image array in the correct format for matplotlib.
    - action: The action taken in this step.
    - q_vals: A tensor of Q values.
    - step_number: The current step number.
    - reward: The reward received after taking the action.
    """
    # Convert action number to descriptive name and prepare the text
    action_text = action_dict.get(action, f"Action {action}")
    details_text = f"Step: {step_number}, Reward: {reward}"

    # Ensure the image is in uint8 format for display
    image = image.astype(np.uint8)

    # Create figure with two subplots: one for the image and one for the Q values histogram
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the image
    axs[0].imshow(image)
    axs[0].text(0.5, -0.1, action_text, color='white', transform=axs[0].transAxes,
                ha="center", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    axs[0].text(0.5, -0.15, details_text, color='white', transform=axs[0].transAxes,
                ha="center", fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    axs[0].axis('off')

    # Plot the histogram of Q values
    # Convert Q values tensor to numpy array
    q_vals_np = torch.squeeze(q_vals).detach().cpu().numpy()

    # Number of bins/datasets - assuming one bin per Q value for simplicity
    num_bins = len(q_vals_np)

    # Prepare the actions labels for the histogram
    # Assuming actions labels are simply 0 to N-1 (or however they are defined)
    actions_labels = [f"Action {i}" for i in range(num_bins)]

    # If you want each bin to be the same color, this step might be optional
    color = 'skyblue'

    # Plot the histogram of Q values
    q_vals_np = torch.squeeze(q_vals).detach().cpu().numpy()

    # Prepare the action names using the action_dict
    actions_labels = [action_dict.get(i, f"Action {i}") for i in range(len(q_vals_np))]

    # Plot the histogram of Q values
    axs[1].bar(actions_labels, q_vals_np, color='skyblue')

    # Set the title and labels for the axes
    axs[1].set_title('Action Distribution')
    axs[1].set_xlabel('Actions')
    axs[1].set_ylabel('Action Value')

    # Adjust x-axis to show action labels correctly
    axs[1].set_xticks(range(len(actions_labels)))
    axs[1].set_xticklabels(actions_labels, rotation=45, ha="right")  # Rotate for better visibility

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Convert the Matplotlib figure to an image array and close the figure to free memory
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return img_array


def save_trajectory_as_gif(trajectory, rewards, action_dict: dict[int, str], folder="trajectories",
                           filename="trajectory.gif"):
    """
    Saves the trajectory as a GIF in a specified folder, including step numbers and rewards.

    Parameters:
    - trajectory: List of tuples, each containing (image, action).
    - rewards: List of rewards for each step in the trajectory.
    """
    # Ensure the target folder exists
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    images_with_actions = [create_image_with_action(action_dict, img, action, q_vals, step_number, rewards[step_number])
                           for step_number, (img, action, q_vals) in enumerate(trajectory)]
    imageio.mimsave(filepath, images_with_actions, fps=10)


def make_env(configure: dict) -> SimpleGridWorld:
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
        env = SimpleGridWorld(
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


def make_abs_env(
        configure: dict,
        prior_agent: stable_baselines3.common.on_policy_algorithm.BaseAlgorithm,
        agent: stable_baselines3.common.on_policy_algorithm.BaseAlgorithm,
        num_clusters: int,
        abs_rate=0.5,
) -> SimpleGridWorldWithStateAbstraction or SimpleGridWorld:
    env = make_env(configure)
    if random.random() > abs_rate:
        return env
    sampler = BaselinePPOSimpleGridBehaviourIterSampler(env, agent, prior_agent, reset_env=True)
    sampler.sample()
    cluster = sampler.make_cluster(num_clusters)
    env = SimpleGridWorldWithStateAbstraction(env, cluster)
    return env


def save_model(model, iteration, base_name="simple-gridworld-ppo", save_dir="saved-models"):
    """Save the model with a custom base name, iteration number, and directory."""
    model_path = os.path.join(save_dir, f"{base_name}-{iteration}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def find_newest_model(base_name="simple-gridworld-ppo", save_dir="saved-models"):
    """Find the most recently saved model based on iteration number and custom base name, and return its path and
    iteration number."""
    model_files = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith('.zip')]
    if not model_files:
        return None, -1  # Return None for both model path and iteration number if no model files found
    # Extracting iteration numbers
    iteration_numbers = []
    for f in model_files:
        try:
            iteration_numbers.append(int(f.replace(base_name + '-', '').split('.')[0]))
        except ValueError:
            pass
    # iteration_numbers = [int(f.replace(base_name + '-', '').split('.')[0]) for f in model_files]
    # Finding the index of the latest model
    latest_model_index = iteration_numbers.index(max(iteration_numbers))
    # Getting the latest model file name
    latest_model = model_files[latest_model_index]
    # Extracting the iteration number of the latest model
    latest_model_iteration = iteration_numbers[latest_model_index]
    if latest_model_iteration is None:
        latest_model_iteration = -1
    return os.path.join(save_dir, latest_model), latest_model_iteration


class TestAndLogCallback(BaseCallback):
    def __init__(
            self,
            eval_env_configurations: list[dict],
            log_path: str,
            # session_name: str,
            n_eval_episodes=10,
            eval_freq=10000,
            deterministic=False,
            render=False,
            verbose=1,
    ):
        super(TestAndLogCallback, self).__init__(verbose)
        self.eval_env_configs = eval_env_configurations
        self.env_names = [each['env_file'].split('/')[-1] for each in eval_env_configurations]
        self.envs = [make_env(each) for each in eval_env_configurations]
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.log_path = log_path
        # self.session_name = session_name
        self.n_eval_episodes = n_eval_episodes
        # For TensorBoard logging
        self.tb_writer = None
        self.eval_timesteps = []

    def _init_callback(self) -> None:
        if self.tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def _on_step(self) -> bool:
        super(TestAndLogCallback, self)._on_step()
        if self.n_calls % self.eval_freq == 0:
            for env_name, env in zip(self.env_names, self.envs):
                # Manually evaluate the policy on each environment
                mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=self.n_eval_episodes,
                                                          deterministic=self.deterministic, render=self.render)

                # Log results for each environment under a unique name
                self.tb_writer.add_scalar(f'{env_name}/mean_reward', mean_reward, self.num_timesteps)
                self.tb_writer.add_scalar(f'{env_name}/std_reward', std_reward, self.num_timesteps)

                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps}. {env_name} Mean reward: {mean_reward} +/- {std_reward}.")
        return True

    def _on_training_end(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()


class UpdateEnvCallback(BaseCallback):
    def __init__(
            self,
            env_configurations: list[dict],
            num_clusters_start: int,
            num_clusters_end: int,
            update_env_freq=10000,
            update_num_clusters_freq=10000,
            update_agent_freq=200000,
            verbose=1,
            abs_rate=0.5,
    ):
        super(UpdateEnvCallback, self).__init__(verbose)
        self.env_configs = env_configurations
        self.num_clusters_start = num_clusters_start
        self.num_clusters = num_clusters_start
        self.num_clusters_end = num_clusters_end
        self.update_env_freq = update_env_freq
        self.update_agent_freq = update_agent_freq
        self.update_num_clusters_freq = update_num_clusters_freq
        self.prior_agent = None
        self.abs_rate = abs_rate

    def _on_step(self) -> bool:
        if self.prior_agent is None:
            self.prior_agent = PPO("CnnPolicy", self.model.env, policy_kwargs={"normalize_images": False}, verbose=1)
            # Copy the weights from the current model to the new_prior_agent
            self.prior_agent.set_parameters(self.model.get_parameters())
            print("Updated prior agent.")
        if self.n_calls % self.update_agent_freq == 0:
            self.prior_agent = PPO("CnnPolicy", self.model.env, policy_kwargs={"normalize_images": False}, verbose=1)
            # Copy the weights from the current model to the new_prior_agent
            self.prior_agent.set_parameters(self.model.get_parameters())
            print("Updated prior agent.")
        if self.n_calls % self.update_env_freq == 0:
            for i in range(len(self.model.env.envs)):
                new_env = make_abs_env(self.env_configs[i], self.prior_agent, self.model, self.num_clusters, self.abs_rate)
                self.model.env.envs[i] = new_env
                if self.verbose:
                    print(f"Updated environment {i} at step {self.num_timesteps}.")
        if self.n_calls % self.update_num_clusters_freq == 0:
            if self.num_clusters < self.num_clusters_end:
                self.num_clusters += 1
                print(f"Updated number of clusters: {self.num_clusters} at step {self.num_timesteps}.")
        return True
