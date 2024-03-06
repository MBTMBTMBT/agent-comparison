import os
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch


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
    axs[1].set_title('Q Values Distribution')
    axs[1].set_xlabel('Actions')
    axs[1].set_ylabel('Q Value')

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


def save_trajectory_as_gif(trajectory, rewards, action_dict: dict[int, str], folder="trajectories", filename="trajectory.gif"):
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

