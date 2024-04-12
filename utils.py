import collections
import math
import os
import random
import re

import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import imageio
import stable_baselines3.common.on_policy_algorithm
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import DataLoader, Subset

from behaviour_tracer import BaselinePPOSimpleGridBehaviourIterSampler
from env_sampler import TransitionBuffer
from feature_model import FeatureNet
from simple_gridworld import SimpleGridWorld, SimpleGridWorldWithStateAbstraction
from typing import Type


def find_latest_checkpoint(model_dir, start_with='model_epoch_'):
    """Find the latest model checkpoint in the given directory."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith(start_with) and f.endswith('.pth')]
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


def make_env(configure: dict, wrapper: Type[gymnasium.Wrapper] = None) -> SimpleGridWorld:
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
        if wrapper is not None:
            env = wrapper(env)
    return env


def make_abs_env(
        configure: dict,
        prior_agent: stable_baselines3.common.on_policy_algorithm.BaseAlgorithm,
        agent: stable_baselines3.common.on_policy_algorithm.BaseAlgorithm,
        # num_clusters: int,
        abs_rate=0.5,
        control_info_weight=100.0,
        plot_path=None,
        plot_path_cluster=None,
) -> SimpleGridWorldWithStateAbstraction or SimpleGridWorld:
    env = make_env(configure)
    if random.random() > abs_rate or ("do_abs" in configure.keys() and not configure["do_abs"]):
        return env
    sampler = BaselinePPOSimpleGridBehaviourIterSampler(env, agent, prior_agent, control_info_weight, reset_env=True)
    sampler.sample()
    if plot_path is not None:
        sampler.plot_grid(plot_path)
    if "num_clusters" in configure.keys():
        num_clusters = configure["num_clusters"]
    else:
        num_clusters = 16384
    clusters = sampler.make_clusters(num_clusters)
    if plot_path_cluster is not None:
        sampler.plot_classified_grid(plot_path_cluster, clusters=clusters)
    env = SimpleGridWorldWithStateAbstraction(env, clusters)
    return env


# def load_model(model, optimizer, path="model.pth", cpu_only=False):
#     if cpu_only:
#         checkpoint = torch.load(path, map_location=torch.device('cpu'))
#     else:
#         checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     if optimizer is not None:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#     return model, optimizer, epoch, best_val_loss


def old_save_model(model, iteration, base_name="simple-gridworld-ppo", save_dir="saved-models"):
    """Save the model with a custom base name, iteration number, and directory."""
    model_path = os.path.join(save_dir, f"{base_name}-{iteration}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def old_find_newest_model(base_name="simple-gridworld-ppo", save_dir="saved-models"):
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


def save_model(model, num_epoch: int, num_step: int, base_name: str, save_dir: str):
    model_path = os.path.join(save_dir, f'{base_name}-EPOCH{num_epoch}-STEP{num_step}.zip')
    model.save(model_path)
    print(f"Model saved to {model_path}")


def find_newest_model(base_name: str, save_dir: str):
    """
    Find the model with the same base name but the largest num_epoch.
    Return model path, epoch, and number of steps.
    """
    model_files = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith('.zip')]
    if not model_files:
        return None, -1, -1  # No model found

    max_epoch = -1
    num_steps = -1
    model_path = None

    for f in model_files:
        parts = f.split('-')
        try:
            epoch_part = parts[1]  # Assuming format is always correct
            step_part = parts[2]

            epoch_num = int(epoch_part.replace('EPOCH', ''))
            step_num = int(step_part.replace('STEP', '').split('.')[0])  # Removing '.zip' and converting to int

            if epoch_num > max_epoch:
                max_epoch = epoch_num
                num_steps = step_num
                model_path = os.path.join(save_dir, f)
        except ValueError:
            # If parsing fails, skip this file
            continue

    return model_path, max_epoch, num_steps


def plot_decoded_images(iterable_env: collections.abc.Iterator, encoder: torch.nn.Module, decoder: torch.nn.Module,
                        save_path: str, device=torch.device("cpu")):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    # reset iterator before using it
    iterable_env.iter_reset()
    # Store z vectors from all iterations
    z_vectors = []
    fake_x_imgs = []
    real_x_imgs = []
    for observation, terminated, position, connections, reward in iterable_env:
        if observation is not None:
            observation = torch.unsqueeze(observation, dim=0).to(device)
            with torch.no_grad():
                z = encoder(observation)
                fake_x = decoder(z)
                z = z.detach().cpu().numpy()
                fake_x = fake_x.detach().cpu().numpy()
                z_vectors.append(z.squeeze(0))
                fake_x_imgs.append(fake_x.squeeze(0))
                real_x = observation.detach().cpu().numpy()
                real_x_imgs.append(real_x.squeeze(0))
    # plot reconstructed xs:
    plt.figure()
    num_xs = len(fake_x_imgs)
    grid_size = math.ceil(math.sqrt(num_xs))  # Calculate grid size that's as square as possible
    # Create a figure to hold the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2), dpi=100)
    # Flatten axes array for easier indexing
    axes = axes.ravel()
    for i, img in enumerate(fake_x_imgs):
        # Transpose the image from [channels, height, width] to [height, width, channels] for plotting
        img_transposed = img.transpose((1, 2, 0))
        image_clipped = np.clip(img_transposed, 0, 1)
        # Plot the image in its subplot
        axes[i].imshow(image_clipped)
        axes[i].axis('off')  # Hide the axis
        # Hide any unused subplots if the number of images is not a perfect square
        for j in range(i + 1, grid_size ** 2):
            axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path.split('.')[-2]+'_decoded.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    # plot reconstructed xs:
    plt.figure()
    # Create a figure to hold the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2), dpi=100)
    # Flatten axes array for easier indexing
    axes = axes.ravel()
    for i, img in enumerate(real_x_imgs):
        # Transpose the image from [channels, height, width] to [height, width, channels] for plotting
        img_transposed = img.transpose((1, 2, 0))
        image_clipped = np.clip(img_transposed, 0, 1)
        # Plot the image in its subplot
        axes[i].imshow(image_clipped)
        axes[i].axis('off')  # Hide the axis
        # Hide any unused subplots if the number of images is not a perfect square
        for j in range(i + 1, grid_size ** 2):
            axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path.split('.')[-2] + '_original.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_representations(iterable_env: collections.abc.Iterator, encoder: torch.nn.Module, num_dims: int,
                         save_path: str, device=torch.device("cpu")):
    assert num_dims == 2 or num_dims == 3
    encoder.to(device)
    encoder.eval()
    z_vectors = []
    for observation, terminated, position, connections, reward in iterable_env:
        if observation is not None:
            observation = torch.unsqueeze(observation, dim=0).to(device)
            with torch.no_grad():
                z = encoder(observation)
                z = z.detach().cpu().numpy()
                z_vectors.append(z.squeeze(0))
    if num_dims == 2:
        plt.figure(figsize=(8, 8))
        for z in z_vectors:
            plt.scatter(z[0], z[1])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(save_path)
        plt.close()
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for z in z_vectors:
            ax.scatter(z[0], z[1], z[2])
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        # different view directions
        views = [(30, 30), (30, 60), (30, 90), (60, 30), (60, 60), (90, 90), ]  # List of (elev, azim) pairs
        # Save the plot
        save_name = os.path.basename(save_path).split(".")[0]
        save_dir = os.path.dirname(save_path)
        for i, (elev, azim) in enumerate(views, start=1):
            ax.view_init(elev=elev, azim=azim)
            plt.draw()  # Update the plot with the new view
            # Save each view to a different file
            save_path_ = os.path.join(save_dir, f"{save_name}_view{i}.png")
            plt.savefig(save_path_)
            # print(f"Saved plot to {save_path}")
        plt.close(fig)  # Close the plot figure after saving all views


class StepCounterCallback(BaseCallback):
    def __init__(self, init_counter_val=0, verbose=0,):
        super(StepCounterCallback, self).__init__(verbose)
        self.step_count = init_counter_val

    def _on_step(self) -> bool:
        self.step_count += 1
        # can also control training by returning False to stop
        # for example, stop after 10,000 steps
        # return self.step_count <= 10000
        return True


class TestAndLogCallback(BaseCallback):
    def __init__(
            self,
            eval_env_configurations: list[dict],
            log_path: str,
            # session_name: str,
            n_eval_episodes=10,
            eval_freq=10000,
            start_num_steps: int or None = None,
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

        self.start_num_steps = self.num_timesteps
        if start_num_steps is not None:
            self.start_num_steps = start_num_steps

    def _init_callback(self) -> None:
        if self.tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def _on_step(self) -> bool:
        super(TestAndLogCallback, self)._on_step()
        self.start_num_steps += 1
        # print(self.n_calls, self.eval_freq)
        if self.n_calls % self.eval_freq == 0:
            for env_name, env in zip(self.env_names, self.envs):
                # Manually evaluate the policy on each environment
                mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=self.n_eval_episodes,
                                                          deterministic=self.deterministic, render=self.render)

                # Log results for each environment under a unique name
                self.tb_writer.add_scalar(f'{env_name}/mean_reward', mean_reward, self.start_num_steps)
                self.tb_writer.add_scalar(f'{env_name}/std_reward', std_reward, self.start_num_steps)

                if self.verbose > 0:
                    print(f"Step: {self.start_num_steps}. {env_name} Mean reward: {mean_reward} +/- {std_reward}.")
        return True

    def _on_training_end(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()


class UpdateAbsEnvCallback(BaseCallback):
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
            control_info_weight: float = 101.0,
            plot_dir=None,
    ):
        super(UpdateAbsEnvCallback, self).__init__(verbose)
        self.env_configs = env_configurations
        self.num_clusters_start = num_clusters_start
        self.num_clusters = num_clusters_start
        self.num_clusters_end = num_clusters_end
        self.update_env_freq = update_env_freq
        self.update_agent_freq = update_agent_freq
        self.update_num_clusters_freq = update_num_clusters_freq
        self.prior_agent = None
        self.abs_rate = abs_rate
        self.control_info_weight = control_info_weight
        self.plot_dir = plot_dir

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
                if self.plot_dir is not None:
                    plot_path = os.path.join(self.plot_dir, self.env_configs[i]['env_file'].split('/')[-1].split('.')[
                        0] + f"-step{self.n_calls}.png")
                    plot_path_cluster = os.path.join(self.plot_dir,
                                                     self.env_configs[i]['env_file'].split('/')[-1].split('.')[
                                                         0] + f"-step{self.n_calls}.gif")
                else:
                    plot_path = None
                    plot_path_cluster = None
                new_env = make_abs_env(self.env_configs[i], self.prior_agent, self.model, self.abs_rate,
                                       self.control_info_weight, plot_path=plot_path,
                                       plot_path_cluster=plot_path_cluster)
                self.model.env.envs[i] = new_env
                if self.verbose:
                    print(f"Updated environment {i} at step {self.num_timesteps}.")
        if self.n_calls % self.update_num_clusters_freq == 0:
            if self.num_clusters < self.num_clusters_end:
                self.num_clusters += 1
                print(f"Updated number of clusters: {self.num_clusters} at step {self.num_timesteps}.")
        return True


class UpdateFeatureExtractorCallback(BaseCallback):
    def __init__(
            self,
            feature_extractor_full_model: FeatureNet,
            env_configurations: list[dict],
            buffer_size_to_train=16384,
            sample_rate=1,
            replay_times=4,
            batch_size=64,
            verbose=1,
            plot_dir=None,
            device=torch.device('cpu'),
            tb_writer=None,
            counter=0,
            # show_progress_bar=False,
    ):
        super(UpdateFeatureExtractorCallback, self).__init__(verbose)
        self.env_configs = env_configurations
        self.envs = [make_env(each) for each in env_configurations]
        self.feature_extractor_full_model = feature_extractor_full_model
        self.buffer_size_to_train = buffer_size_to_train
        self.sample_rate = sample_rate
        self.replay_times = replay_times
        self.plot_dir = plot_dir
        self.batch_size = batch_size
        self.device = device
        self.tb_writer = tb_writer
        self.counter = counter
        # self.show_progress_bar = show_progress_bar

        # self.model_updated_flag = False  # to know when to save the model
        self.do_plot = plot_dir is not None
        self.verbose = verbose

    def _on_step(self) -> bool:
        # check if buffer is filled
        if self.get_buffer_size() < self.buffer_size_to_train:
            return True

        # def model_checksum(model):
        #     checksum = torch.tensor(0.0).to(self.device)
        #     for param in model.parameters():
        #         checksum += torch.sum(param.data)
        #     return checksum.item()

        # # Inspecting weights before PPO instantiation
        # # print("Weights before:", list(feature_extractor.parameters())[0].data)
        # initial_checksum = model_checksum(self.feature_extractor_full_model)
        # print(f"Checksum before this round of training: {initial_checksum}")

        print('Training Feature extractor ...')
        self.feature_extractor_full_model.to(self.device)
        self.feature_extractor_full_model.train()
        transition_buffer = self.get_buffer_obj()
        if self.sample_rate <= 0.9999:
            # Generate random indices
            num_samples = int(len(transition_buffer) * self.sample_rate)
            indices = np.random.choice(len(transition_buffer), num_samples, replace=False)
            # Create the subset
            transition_buffer = Subset(transition_buffer, indices)
        # empty sampler buffers:
        self.empty_sampler_buffers()
        dataloader = DataLoader(transition_buffer, batch_size=self.batch_size, shuffle=True)
        # dataloader = DataLoader(transition_buffer, batch_size=1, shuffle=True)
        # __counter = 0
        for _ in range(self.replay_times):
            for x0, a, x1 in dataloader:
                x0 = x0.to(self.device)
                a = a.to(self.device)
                x1 = x1.to(self.device)

                # if __counter < 5:
                #     __counter += 1
                #
                #     ACTION_NAMES = {
                #         0: 'UP',
                #         1: 'DOWN',
                #         2: 'LEFT',
                #         3: 'RIGHT',
                #     }
                #
                #     # Convert tensors to numpy for matplotlib
                #     x0_np = x0.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)  # Transpose to channel-last format
                #     x1_np = x1.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)  # Transpose to channel-last format
                #
                #     # Clamp values to [0, 1] range to ensure proper display
                #     x0_np = x0_np.clip(0, 1)
                #     x1_np = x1_np.clip(0, 1)
                #
                #     # Plotting
                #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                #
                #     # Display x0
                #     axs[0].imshow(x0_np)
                #     axs[0].axis('off')  # Hide axes for better visualization
                #     axs[0].set_title('x0')
                #
                #     # Display action in the middle
                #     action_name = ACTION_NAMES[a.detach().cpu().item()]  # Get action name
                #     axs[1].text(0.5, 0.5, action_name, fontsize=15, ha='center')
                #     axs[1].axis('off')
                #     axs[1].set_title('Action')
                #
                #     # Display x1
                #     axs[2].imshow(x1_np)
                #     axs[2].axis('off')
                #     axs[2].set_title('x1')
                #
                #     plt.tight_layout()
                #     plt.show()

                loss_val, inv_loss_val, ratio_loss_val, pixel_loss_val = self.feature_extractor_full_model.train_batch(
                    x0, x1, a)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('loss', loss_val, self.counter)
                    self.tb_writer.add_scalar('inv_loss', inv_loss_val, self.counter)
                    self.tb_writer.add_scalar('ratio_loss', ratio_loss_val, self.counter)
                    self.tb_writer.add_scalar('pixel_loss', pixel_loss_val, self.counter)
                self.counter += 1
                if self.verbose:
                    print(
                        f"Updated feature extractor at step {self.counter}, loss {loss_val:.3f}, inv loss: {inv_loss_val:.3f}, ratio loss: {ratio_loss_val:.3f}, pixel loss: {pixel_loss_val:.3f}")

        # self.model_updated_flag = True

        if self.do_plot and self.plot_dir is not None:
            for config, env in zip(self.env_configs, self.envs):
                env_path = config['env_file']
                env_name = env_path.split('/')[-1].split('.')[0]
                if not os.path.isdir(self.plot_dir):
                    os.makedirs(self.plot_dir)
                save_path = os.path.join(self.plot_dir, f"{env_name}{self.counter}.png")
                if self.feature_extractor_full_model.decoder is not None:
                    plot_decoded_images(env, self.feature_extractor_full_model.phi,
                                        self.feature_extractor_full_model.decoder, save_path, self.device)

                if self.feature_extractor_full_model.n_latent_dims == 2 or self.feature_extractor_full_model.n_latent_dims == 3:
                    plot_representations(env, self.feature_extractor_full_model.phi, self.feature_extractor_full_model.n_latent_dims, save_path, self.device)

        # initial_checksum = model_checksum(self.feature_extractor_full_model)
        # print(f"Checksum after this round of training: {initial_checksum}")

        return True

    def get_buffer_size(self):
        total_size = 0
        for i, sampler_wrapper in enumerate(self.model.env.envs):
            # print(f'env {i}: {len(sampler_wrapper.transition_pairs)}')
            total_size += len(sampler_wrapper.transition_pairs)
        return total_size

    def get_buffer_obj(self) -> TransitionBuffer:
        transition_pairs = []
        for sampler_wrapper in self.model.env.envs:
            transition_pairs += sampler_wrapper.transition_pairs
        transition_buffer = TransitionBuffer(transition_pairs)
        return transition_buffer

    def empty_sampler_buffers(self):
        for sampler_wrapper in self.model.env.envs:
            sampler_wrapper.transition_pairs = []
