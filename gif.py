import numpy as np
import matplotlib.pyplot as plt


def create_image_with_action(action_dict: dict[int, str], image, action, step_number, reward):
    """
    Creates an image with the action text and additional details overlay.

    Parameters:
    - image: The image array in the correct format for matplotlib.
    - action: The action taken in this step.
    - step_number: The current step number.
    - reward: The reward received after taking the action.
    """
    # Convert action number to descriptive name and prepare the text
    action_text = action_dict.get(action, f"Action {action}")
    details_text = f"Step: {step_number}, Reward: {reward}"

    # Normalize or convert the image if necessary
    image = image.astype(np.uint8)  # Ensure image is in uint8 format for display

    fig, ax = plt.subplots()
    ax.imshow(image)
    # Position the action text
    ax.text(0.5, -0.1, action_text, color='white', transform=ax.transAxes,
            ha="center", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    # Position the details text (step number and reward)
    ax.text(0.5, -0.15, details_text, color='white', transform=ax.transAxes,
            ha="center", fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    ax.axis('off')

    # Convert the Matplotlib figure to an image array and close the figure to free memory
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return img_array


def save_trajectory_as_gif(self, trajectory, rewards, folder="trajectories", filename="trajectory.gif"):
    """
    Saves the trajectory as a GIF in a specified folder, including step numbers and rewards.

    Parameters:
    - trajectory: List of tuples, each containing (image, action).
    - rewards: List of rewards for each step in the trajectory.
    """
    # Ensure the target folder exists
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    images_with_actions = [self.create_image_with_action(img, action, step_number, rewards[step_number])
                           for step_number, (img, action) in enumerate(trajectory)]
    imageio.mimsave(filepath, images_with_actions, fps=3)