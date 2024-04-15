if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import math

    import torch
    from configurations import maze13_sampling, mix_sampling, four_room13_sampling
    from feature_model import FeatureNet

    CONFIGS = maze13_sampling

    # model configs
    NUM_ACTIONS = 4
    LATENT_DIMS = 2
    RECONSTRUCT_SIZE = (96, 96)
    RECONSTRUCT_SCALE = 2

    # sampler configs
    SAMPLE_SIZE = 16384
    SAMPLE_REPLAY_TIME = 4
    MAX_SAMPLE_STEP = 4096

    # train hyperparams
    WEIGHTS = {
        'inv': 1.0,
        'dis': 1.0,
        'neighbour': 1.0,
        'dec': 0.0,
        'rwd': 0.0,
        'demo': 0.0,
    }
    BATCH_SIZE = 64
    LR = 1e-4

    # train configs
    EPOCHS = 80
    SAVE_FREQ = 1
    TEST_FREQ = 1

    session_name = "learn_feature_maze13_neighbour"
    feature_model_name = 'feature_model_step'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, img_size=RECONSTRUCT_SIZE, initial_scale_factor=RECONSTRUCT_SCALE, device=device, weights=WEIGHTS).to(device)

    from utils import find_latest_checkpoint, plot_decoded_images, plot_representations
    import os

    if not os.path.isdir(session_name):
        os.makedirs(session_name)
    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name)

    # load parameters if it has any
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        epoch_counter, step_counter, performance = model.load(latest_checkpoint)
        epoch_counter += 1
        step_counter += 1
    else:
        epoch_counter = 0
        step_counter = 0
        performance = float('inf')

    # init tensorboard writer
    tb_writer = SummaryWriter(log_dir=session_name)

    from env_sampler import TransitionBuffer, RandomSampler
    from utils import make_env

    progress_bar = tqdm(range(epoch_counter, EPOCHS), desc=f'Training Epoch {epoch_counter}')
    for i, batch in enumerate(progress_bar):
        sampler = RandomSampler(not WEIGHTS['demo'] == 0.0)
        envs = [make_env(config) for config in CONFIGS]
        while len(sampler.transition_pairs) < SAMPLE_SIZE:
            for env in envs:
                sampler.sample(env, MAX_SAMPLE_STEP)

        transition_buffer = TransitionBuffer(sampler.transition_pairs)
        dataloader = DataLoader(transition_buffer, batch_size=BATCH_SIZE, shuffle=True)
        loss_val = 0.0
        for _ in range(SAMPLE_REPLAY_TIME):
            for batch in dataloader:
                d = None
                if WEIGHTS['demo'] == 0.0:
                    x0, a, x1, r = batch  # Assuming batch format is [x0, a, x1, r] when 'demo' is not used
                else:
                    x0, a, x1, r, d = batch  # Assuming batch format is [x0, a, x1, r, d] when 'demo' is used

                # Move tensors to the appropriate device
                x0 = x0.to(device)
                a = a.to(device)
                x1 = x1.to(device)
                r = r.to(device)
                if d is not None:
                    d = d.to(device)

                # Train batch and calculate losses
                loss_vals = model.train_batch(x0, x1, a, r, d)

                # Log values to TensorBoard
                names = ['loss', 'inv_loss', 'neighbour_loss', 'ratio_loss', 'pixel_loss', 'reward_loss', 'demo_loss']
                for name, val in zip(names, loss_vals):
                    tb_writer.add_scalar(name, val, step_counter)

                # Increment the step counter
                step_counter += 1

        if epoch_counter % TEST_FREQ == 0:
            _plot_dir = os.path.join(session_name, 'plots')
            for config, env in zip(CONFIGS, envs):
                env_path = config['env_file']
                env_name = env_path.split('/')[-1].split('.')[0]
                if not os.path.isdir(_plot_dir):
                    os.makedirs(_plot_dir)
                save_path = os.path.join(_plot_dir, f"{env_name}-{step_counter}.png")

                if model.decoder is not None:
                    plot_decoded_images(env, model.phi,
                                        model.decoder, save_path, device)
                if LATENT_DIMS == 2 or LATENT_DIMS == 3:
                    plot_representations(env, model.phi, LATENT_DIMS, save_path, device)

        if epoch_counter % SAVE_FREQ == 0:
            model.save(f"{session_name}/model_epoch_{epoch_counter}.pth", epoch_counter, step_counter, performance)
        epoch_counter += 1
        progress_bar.set_description(
            f'Train Epoch {epoch_counter}: Loss: {loss_val:.2f}')
