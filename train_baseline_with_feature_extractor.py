from feature_wrapper import CustomFeatureExtractor

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from configurations import mix_sampling
    from feature_model import FeatureNet
    from utils import find_latest_checkpoint
    from env_sampler import TransitionBuffer, RandomSampler, SamplerWrapper
    from utils import make_env
    from stable_baselines3 import PPO
    from configurations import *
    from utils import *
    from functools import partial
    from stable_baselines3.common.env_util import DummyVecEnv

    TRAIN_CONFIGS = maze13_sampling
    EVAL_CONFIGS = maze13_sampling

    # model configs
    NUM_ACTIONS = 4
    LATENT_DIMS = 64
    RECONSTRUCT_SIZE = (96, 96)
    RECONSTRUCT_SCALE = 2

    # sampler configs
    SAMPLE_SIZE = 16384
    SAMPLE_REPLAY_TIME = 1
    MAX_SAMPLE_STEP = SAMPLE_SIZE // len(TRAIN_CONFIGS) // SAMPLE_REPLAY_TIME
    VEC_ENV_REPEAT_TIME = 1
    SAMPLE_RATE = 1.0  # not for pre-training

    # train hyperparams
    WEIGHTS = {
        'inv': 1.0,
        'dis': 1.0,
        'dec': 0.0,
        'rwd': 0.0,
    }
    BATCH_SIZE = 32
    LR = 1e-4

    # train configs
    PRE_TRAIN_STEPS = SAMPLE_SIZE * SAMPLE_REPLAY_TIME // BATCH_SIZE * 0
    SAVE_FREQ = PRE_TRAIN_STEPS // 5

    EPOCHS = 3
    NUM_STEPS_PER_EPOCH = MAX_SAMPLE_STEP * len(TRAIN_CONFIGS) * SAMPLE_REPLAY_TIME * 20

    # eval configs
    NUM_EVAL_EPISODES = 10
    EVAL_FREQ = NUM_STEPS_PER_EPOCH // len(TRAIN_CONFIGS) // SAMPLE_REPLAY_TIME // 5

    # CNN DECODER (CONTROL GROUP)
    USE_CNN_DECODER = True

    session_name = "ppo_feature_extractor_maze13_64d_cnn"
    feature_model_name = 'feature_model_step'
    baseline_model_name = 'baseline_model'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, img_size=RECONSTRUCT_SIZE,
                                   initial_scale_factor=RECONSTRUCT_SCALE, weights=WEIGHTS, device=device).to(device)

    if not os.path.isdir(session_name):
        os.makedirs(session_name)
    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name, feature_model_name)

    # init tensorboard writer
    tb_writer = SummaryWriter(log_dir=session_name)

    # default values (to avoid editor warnings)
    feature_extractor_step_counter = 0
    performance = float('inf')

    if not USE_CNN_DECODER:  # using abs state feature extractor

        # load parameters if it has any
        if latest_checkpoint:
            print(f"Loading model from {latest_checkpoint}")
            epoch_counter, feature_extractor_step_counter, performance = feature_extractor.load(latest_checkpoint)
            epoch_counter += 1
            feature_extractor_step_counter += 1
        else:
            epoch_counter = 0
            feature_extractor_step_counter = 0
            performance = float('inf')

        print("Start pre-training:")

        # Initialize the progress bar
        pbar = tqdm(total=PRE_TRAIN_STEPS, initial=feature_extractor_step_counter, desc="Pre-Train Progress")

        _save_counter = 0
        while feature_extractor_step_counter < PRE_TRAIN_STEPS:
            sampler = RandomSampler()
            envs = [make_env(config) for config in TRAIN_CONFIGS]
            while len(sampler.transition_pairs) < SAMPLE_SIZE:
                for env in envs:
                    sampler.sample(env, MAX_SAMPLE_STEP)

            transition_buffer = TransitionBuffer(sampler.transition_pairs)
            dataloader = DataLoader(transition_buffer, batch_size=BATCH_SIZE, shuffle=True)
            # dataloader = DataLoader(transition_buffer, batch_size=1, shuffle=True)
            loss_val = 0.0
            for _ in range(SAMPLE_REPLAY_TIME):
                # __counter = 0
                for x0, a, x1, r in dataloader:
                    x0 = x0.to(device)
                    a = a.to(device)
                    x1 = x1.to(device)
                    r = r.to(device)

                    # if __counter < 20:
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

                    loss_val, inv_loss_val, ratio_loss_val, pixel_loss_val, reward_loss_val = feature_extractor.train_batch(x0, x1, a, r)
                    tb_writer.add_scalar('loss', loss_val, feature_extractor_step_counter)
                    tb_writer.add_scalar('inv_loss', inv_loss_val, feature_extractor_step_counter)
                    tb_writer.add_scalar('ratio_loss', ratio_loss_val, feature_extractor_step_counter)
                    tb_writer.add_scalar('pixel_loss', pixel_loss_val, feature_extractor_step_counter)
                    tb_writer.add_scalar('reward_loss', reward_loss_val, feature_extractor_step_counter)

                    # Update the progress bar description with the latest loss values
                    pbar.set_description(
                        f"L:{loss_val:.3f}-Inv:{inv_loss_val:.3f}-Rat:{ratio_loss_val:.3f}-Pix:{pixel_loss_val:.3f}-Rwd:{reward_loss_val:.3f}")
                    pbar.update(1)  # Assuming each batch represents a single step, adjust as necessary

                    feature_extractor_step_counter += 1
                    if feature_extractor_step_counter >= PRE_TRAIN_STEPS:
                        break
                if feature_extractor_step_counter >= PRE_TRAIN_STEPS:
                    break

            __save_counter = feature_extractor_step_counter // SAVE_FREQ
            if __save_counter > _save_counter:
                _save_name = f"{session_name}/{feature_model_name}_{feature_extractor_step_counter}.pth"
                print(f"Saving model with name {_save_name}")
                feature_extractor.save(_save_name, epoch_counter, feature_extractor_step_counter, performance)
                _save_counter = __save_counter

                _plot_dir = os.path.join(session_name, 'plots')
                for config, env in zip(TRAIN_CONFIGS, envs):
                    env_path = config['env_file']
                    env_name = env_path.split('/')[-1].split('.')[0]
                    if not os.path.isdir(_plot_dir):
                        os.makedirs(_plot_dir)
                    save_path = os.path.join(_plot_dir, f"{env_name}-{feature_extractor_step_counter}.png")

                    if feature_extractor.decoder is not None:
                        plot_decoded_images(env, feature_extractor.phi,
                                            feature_extractor.decoder, save_path, device)
                    if LATENT_DIMS == 2 or LATENT_DIMS == 3:
                        plot_representations(env, feature_extractor.phi, LATENT_DIMS, save_path, device)

        pbar.close()

    _train_env_configurations = TRAIN_CONFIGS
    train_env_configurations = []
    for _ in range(VEC_ENV_REPEAT_TIME):
        train_env_configurations += _train_env_configurations

    if not USE_CNN_DECODER:
        env_fns = [partial(make_env, config, SamplerWrapper) for config in train_env_configurations]
    else:
        env_fns = [partial(make_env, config) for config in train_env_configurations]
    env = DummyVecEnv(env_fns)

    # Check for the latest saved model
    model_path, epoch_counter, agent_step_counter = find_newest_model(baseline_model_name, session_name)
    epoch_counter += 1

    # get callbacks
    step_counter_callback = StepCounterCallback(init_counter_val=agent_step_counter)
    test_and_log_callback = TestAndLogCallback(
        EVAL_CONFIGS,
        session_name,
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        start_num_steps=agent_step_counter,
        deterministic=False,
        render=False,
        verbose=1,
    )

    # def model_checksum(model):
    #     checksum = torch.tensor(0.0).to(device)
    #     for param in model.parameters():
    #         checksum += torch.sum(param.data)
    #     return checksum.item()

    if not USE_CNN_DECODER:

        # # Inspecting weights before PPO instantiation
        # print("Weights before:", list(feature_extractor.parameters())[0].data)
        # initial_checksum = model_checksum(feature_extractor)
        # print(f"Initial Checksum: {initial_checksum}")

        # Save the original state of the feature extractor
        original_state_dict = feature_extractor.state_dict()

        # get a temporary feature extractor
        temp_feature_extractor = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, img_size=RECONSTRUCT_SIZE,
                                            initial_scale_factor=RECONSTRUCT_SCALE, weights=WEIGHTS, device=device).to(device)

        # make model
        model = PPO("MlpPolicy", env, n_steps=MAX_SAMPLE_STEP, verbose=1, policy_kwargs={
            'features_extractor_class': CustomFeatureExtractor,
            'features_extractor_kwargs': {'feature_extractor': temp_feature_extractor, 'features_dim': LATENT_DIMS,
                                          'device': device}
        })

        # Restore the original parameters to the feature extractor
        feature_extractor = model.policy.features_extractor.feature_extractor
        feature_extractor.load_state_dict(original_state_dict)

        # # Inspecting weights after PPO instantiation
        # print("Weights after:", list(feature_extractor.parameters())[0].data)
        # post_initialization_checksum = model_checksum(feature_extractor)
        # print(f"Post-Initialization Checksum: {post_initialization_checksum}")

    else:
        model = PPO("CnnPolicy", env, n_steps=MAX_SAMPLE_STEP, policy_kwargs={"normalize_images": False}, verbose=1)

    if model_path is not None:
        print(f"Loading model from {model_path}")
        model.load(model_path)

    print("Start Training:")

    while epoch_counter < EPOCHS:
        print("Epoch {}".format(epoch_counter))

        if not USE_CNN_DECODER:
            update_feature_extractor_callback = UpdateFeatureExtractorCallback(
                feature_extractor,
                env_configurations=TRAIN_CONFIGS,
                buffer_size_to_train=SAMPLE_SIZE,
                sample_rate=SAMPLE_RATE,
                replay_times=SAMPLE_REPLAY_TIME,
                batch_size=BATCH_SIZE,
                verbose=0,
                plot_dir=os.path.join(session_name, 'plots'),
                device=device,
                tb_writer=tb_writer,
                counter=feature_extractor_step_counter,
                # show_progress_bar=False,
            )

            model.learn(
                total_timesteps=NUM_STEPS_PER_EPOCH,
                callback=[step_counter_callback, update_feature_extractor_callback, test_and_log_callback],
                progress_bar=True
            )

            feature_extractor_step_counter = update_feature_extractor_callback.counter

            _save_name = f"{session_name}/{feature_model_name}_{feature_extractor_step_counter}.pth"
            print(f"Model save to {_save_name}")
            feature_extractor.save(_save_name, epoch_counter, feature_extractor_step_counter, performance)

        else:
            model.learn(
                total_timesteps=NUM_STEPS_PER_EPOCH,
                callback=[step_counter_callback, test_and_log_callback],
                progress_bar=True
            )

        agent_step_counter = step_counter_callback.step_count
        save_model(model, num_epoch=epoch_counter, num_step=agent_step_counter, base_name=baseline_model_name,
                   save_dir=session_name)
        epoch_counter += 1
