import gymnasium
import torchvision.transforms as transforms
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from minigrid_custom_env import CustomEnvFromFile
from abstract_agent import AbstractAgent
from replay_buffer import DiscretePrioritizedReplayBuffer
from ppo import PPOWithImageEncoder

from utils import *


def preprocess_observation(obs: dict, rotate=False, size=None) -> torch.Tensor:
    """
    Preprocess the observation obtained from the environment to be suitable for the CNN.
    This function extracts, randomly rotates, and normalizes the 'image' part of the observation.

    :param obs: dict, The observation dictionary received from the environment.
                Expected to have a key 'image' containing the visual representation.
    :return: torch.Tensor, The normalized and randomly rotated image observation.
    """
    # Extract the 'image' array from the observation dictionary
    image_obs = obs['image']
    rotated_tensor = preprocess_image(image_obs, rotate, size)
    return rotated_tensor


def preprocess_image(img: np.ndarray, rotate=False, size=None) -> torch.Tensor:
    # Convert the numpy array to a PIL Image
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(img)

    # Initialize the transformation list
    transformations = []

    # Randomly rotate the image
    if rotate:
        rotation_degrees = np.random.choice([0, 90, 180, 270])
        transformations.append(transforms.RandomRotation([rotation_degrees, rotation_degrees]))

    # Resize the image if size is specified
    if size is not None:
        transformations.append(transforms.Resize(size))

    # Convert the PIL Image back to a tensor
    transformations.append(transforms.ToTensor())

    # Compose all transformations
    transform_compose = transforms.Compose(transformations)

    # Apply transformations
    processed_tensor = transform_compose(pil_image)

    # Normalize the tensor to [0, 1] (if not already normalized)
    processed_tensor /= 255.0 if processed_tensor.max() > 1.0 else 1.0

    # Add a batch dimension
    processed_tensor = processed_tensor.unsqueeze(0)

    return processed_tensor


def run_training(
        env: gymnasium.Env,
        agent: AbstractAgent,
        episodes: int = 100,
        env_name: str = ""
) -> None:
    """
    Runs the training loop for a specified number of episodes using PPO.

    Args:
        env (CustomEnvFromFile): The environment instance where the agent will be trained.
        agent (PPOAgent): The agent to be trained with PPO.
        episodes (int): The total number of episodes to run for training.
        env_name (str): A name for the environment, used for saving outputs.

    Returns:
        None
    """
    for e in range(episodes):
        time_step = 0
        total_reward = 0
        rewards = [0.0, ]
        trajectory = []  # List to record each step for the GIF.
        obs, _ = env.reset()  # Reset the environment at the start of each episode.
        rendered = env.render()
        state = preprocess_image(rendered, rotate=False, size=(128, 128))  # Preprocess the observation for the agent.
        state_rotated = preprocess_image(rendered, rotate=True, size=(128, 128))
        # state_img = obs['image']  # Store the original 'image' observation for visualization.
        # episode_states, episode_actions, episode_rewards, episode_log_probs = [], [], [], []

        for time in range(env.max_steps):
            time_step += 1

            action = agent.select_action(state_rotated,
                                         return_distribution=False)  # Agent selects an action based on the current state.
            next_obs, reward, terminated, truncated, info = env.step(action)  # Execute the action.
            done = terminated or truncated  # Check if the episode has ended.
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            trajectory.append((rendered, action))  # Append the step for the GIF.
            rendered = env.render()
            next_state = preprocess_image(rendered, rotate=False, size=(128, 128))  # Preprocess the new observation.
            next_state_rotated = preprocess_image(rendered, rotate=True, size=(128, 128))
            state = next_state  # Update the current state for the next iteration.
            state_rotated = next_state_rotated
            total_reward += reward
            rewards.append(reward)

            # if time_step % update_timestep == 0:
            #     agent.update()

            if done:
                print(f"Episode: {e}/{episodes}, Time: {time + 1}, Reward: {total_reward}")
                trajectory.append((rendered, action))
                # Save the recorded trajectory as a GIF after each episode.
                save_trajectory_as_gif(trajectory, rewards, ACTION_NAMES, filename=env_name + f"_trajectory_{e}.gif")
                agent.update()
                break


def run_and_sample(
        env: gymnasium.Env,
        agent: PPOWithImageEncoder,
        replay_buffer: DiscretePrioritizedReplayBuffer,
        episodes: int = 1,
        env_name: str = "",
        save_gif=False,
):
    for e in range(episodes):
        time_step = 0
        total_reward = 0
        rewards = [0.0, ]
        trajectory = []  # List to record each step for the GIF.
        obs, _ = env.reset()  # Reset the environment at the start of each episode.
        rendered = env.render()
        state = preprocess_image(rendered, rotate=False, size=(128, 128))  # Preprocess the observation for the agent.
        state_rotated = preprocess_image(rendered, rotate=True, size=(128, 128))

        for time in range(env.max_steps):
            time_step += 1

            action = agent.select_action(state,
                                         return_distribution=False)  # Agent selects an action based on the current state.
            next_obs, reward, terminated, truncated, info = env.step(action)  # Execute the action.
            done = terminated or truncated  # Check if the episode has ended.
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            trajectory.append((rendered, action))  # Append the step for the GIF.
            rendered = env.render()
            next_state = preprocess_image(rendered, rotate=False, size=(128, 128))  # Preprocess the new observation.
            next_state_rotated = preprocess_image(rendered, rotate=True, size=(128, 128))
            state = next_state  # Update the current state for the next iteration.
            state_rotated = next_state_rotated
            total_reward += reward
            rewards.append(reward)

            if done:
                print(f"Episode: {e}/{episodes}, Time: {time + 1}, Reward: {total_reward}")
                trajectory.append((rendered, action))
                # Save the recorded trajectory as a GIF after each episode.
                if save_gif:
                    save_trajectory_as_gif(trajectory, rewards, ACTION_NAMES, filename=env_name + f"_trajectory_{e}.gif")
                agent.charge_replay_buffer(replay_buffer)
                break


if __name__ == "__main__":
    import os
    from minigrid_custom_env import ACTION_NAMES

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = False
    max_ep_len = 512  # max timesteps in one episode
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)  # save model frequency (in num timesteps)
    #####################################################
    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 10  # update policy for K epochs
    batch_size = 64
    replay_size = max_ep_len
    eps_clip = 0.05  # clip parameter for PPO
    gamma = 0.95  # discount factor
    lr_encoder = 0.005  # learning rate for encoder network
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.0005  # learning rate for critic network
    input_channels = 3  # RGB input
    state_dim = 128  # lenth of the vector encoded by encoder
    action_dim = len(ACTION_NAMES)  # actions
    #####################################################

    # check model saving dir
    session_name = "saved-models"
    if not os.path.isdir(session_name):
        os.makedirs(session_name)

    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name)

    # List of environments to train on
    environment_files = [
        'simple_test_corridor_mini.txt',
        'simple_test_corridor.txt',
        'simple_test_corridor_long.txt',
        'simple_test_openspace0.txt',
        'simple_test_openspace1.txt',
        'simple_test_openspace2.txt',
        'simple_test_openspace3.txt',
        'simple_test_openspace4.txt',
        'simple_test_two_rooms0.txt',
        'simple_test_two_rooms1.txt',
        'simple_test_maze_small.txt',
        'simple_test_door_key.txt',
        # Add more file paths as needed
    ]

    # Training settings
    episodes_per_env = {
        'simple_test_corridor_mini.txt': 5,
        'simple_test_corridor.txt': 5,
        'simple_test_corridor_long.txt': 5,
        'simple_test_openspace0.txt': 3,
        'simple_test_openspace1.txt': 3,
        'simple_test_openspace2.txt': 3,
        'simple_test_openspace3.txt': 3,
        'simple_test_openspace4.txt': 3,
        'simple_test_two_rooms0.txt': 5,
        'simple_test_two_rooms1.txt': 5,
        'simple_test_maze_small.txt': 5,
        'simple_test_door_key.txt': 5,
        # Define episodes for more environments as needed
    }

    ################## init the model ###################
    ppo_agent = PPOWithImageEncoder(
        input_channels, state_dim, action_dim,
        lr_encoder, lr_actor, lr_critic,
        gamma,
        K_epochs,
        batch_size,
        replay_size,
        eps_clip,
        has_continuous_action_space,
        device=device
    )
    # load parameters if it has any
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        counter, performance = ppo_agent.load(latest_checkpoint)
        counter += 1
    else:
        counter = 0
        performance = float('inf')

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        # print("starting std of action distribution : ", action_std)
        # print("decay rate of std of action distribution : ", action_std_decay_rate)
        # print("minimum std of action distribution : ", min_action_std)
        # print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    # if random_seed:
    #     print("--------------------------------------------------------------------------------------------")
    #     print("setting random seed to ", random_seed)
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")
    replay_buffer = DiscretePrioritizedReplayBuffer(
        output_capacity=16384,
        total_capacity=16384,
        image_size=(128, 128)
    )

    for turn in range(100):
        for env_file in environment_files:
            # Initialize environment
            env = RGBImgObsWrapper(FullyObsWrapper(
                CustomEnvFromFile(txt_file_path=env_file, render_mode='rgb_array', size=None, max_steps=max_ep_len, agent_start_pos=(1,1))))

            # Run training for the current environment
            print(f"Sampling on {env_file}, turn {turn}...")

            save_gif = turn % 5 == 4
            if save_gif:
                print("Saving trajectories.")

            run_and_sample(env, ppo_agent, replay_buffer, episodes=episodes_per_env[env_file], env_name=env_file, save_gif=save_gif)

        for i in range(2):
            replay_buffer.refresh_output_buffer()
            ppo_agent.update_with_replay_buffer(replay_buffer)
        replay_buffer.shrink_base_buffer()
        ppo_agent.save(f"{session_name}/model_epoch_{counter}.pth", counter)
        counter += 1
