from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from simple_gridworld import SimpleGridWorld

if __name__ == '__main__':
    import torch
    from configurations import maze13_sampling, mix_sampling, four_room13_sampling
    from feature_model import FeatureNet

    CONFIGS = mix_sampling
    NUM_ACTIONS = 4
    LATENT_DIMS = 3

    SAMPLE_SIZE = 16384
    SAMPLE_REPLAY_TIME = 4
    MAX_SAMPLE_STEP = 4096
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 1000
    SAVE_FREQ = 5
    TEST_FREQ = 5

    session_name = "learn_feature_3d"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, device=device).to(device)

    from utils import find_latest_checkpoint
    import os

    if not os.path.isdir(session_name):
        os.makedirs(session_name)
    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name)

    # load parameters if it has any
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        counter, _counter, performance = model.load(latest_checkpoint)
        counter += 1
        _counter += 1
    else:
        counter = 0
        _counter = 0
        performance = float('inf')

    # init tensorboard writer
    tb_writer = SummaryWriter(log_dir=session_name)

    from env_sampler import TransitionBuffer, RandomSampler
    from utils import make_env

    progress_bar = tqdm(range(counter, EPOCHS), desc=f'Training Epoch {counter}')
    for i, batch in enumerate(progress_bar):
        sampler = RandomSampler()
        envs = [make_env(config) for config in CONFIGS]
        while len(sampler.transition_pairs) < SAMPLE_SIZE:
            for env in envs:
                sampler.sample(env, MAX_SAMPLE_STEP)

        transition_buffer = TransitionBuffer(sampler.transition_pairs)
        dataloader = DataLoader(transition_buffer, batch_size=BATCH_SIZE, shuffle=True)
        loss_val = 0.0
        for _ in range(SAMPLE_REPLAY_TIME):
            for x0, a, x1 in dataloader:
                x0 = x0.to(device)
                a = a.to(device)
                x1 = x1.to(device)
                loss_val = model.train_batch(x0, x1, a).detach().cpu().item()
                tb_writer.add_scalar('loss', loss_val, _counter)
                _counter += 1

        if counter % TEST_FREQ == 0:
            encoder = model.phi
            for config, env in zip(CONFIGS, envs):
                env_path = config['env_file']
                env_name = env_path.split('/')[-1].split('.')[0]
                # reset iterator before using it
                env.iter_reset()
                # Store z vectors from all iterations
                z_vectors = []
                for observation, terminated, position, connections, reward in env:
                    if observation is not None:
                        observation = torch.unsqueeze(observation, dim=0).to(device)
                        with torch.no_grad():
                            z = encoder(observation).detach().cpu().numpy()
                            z_vectors.append(z.squeeze(0))

                # After collecting all z vectors, plot them based on LATENT_DIMS
                if LATENT_DIMS == 2:
                    plt.figure(figsize=(8, 8))
                    for z in z_vectors:
                        plt.scatter(z[0], z[1])
                    plt.xlabel("Dimension 1")
                    plt.ylabel("Dimension 2")
                elif LATENT_DIMS == 3:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    for z in z_vectors:
                        ax.scatter(z[0], z[1], z[2])
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_zlabel("Dimension 3")

                    # different view directions
                    views = [(30, 30), (30, 60), (30, 90), (60, 30), (60, 60), (90, 90),]  # List of (elev, azim) pairs

                    # Save the plot
                    img_save_dir = os.path.join(session_name, "saved_encoded_images")
                    if not os.path.isdir(img_save_dir):
                        os.makedirs(img_save_dir)

                    for i, (elev, azim) in enumerate(views, start=1):
                        ax.view_init(elev=elev, azim=azim)
                        plt.draw()  # Update the plot with the new view

                        # Save each view to a different file
                        save_path = os.path.join(img_save_dir,
                                                 f"{env_name}_latent{LATENT_DIMS}_{counter}_view{i}.png")
                        plt.savefig(save_path)
                        print(f"Saved plot to {save_path}")

                    plt.close(fig)  # Close the plot figure after saving all views

        if counter % SAVE_FREQ == 0:
            model.save(f"{session_name}/model_epoch_{counter}.pth", counter, _counter, performance)
        counter += 1
        progress_bar.set_description(
            f'Train Epoch {counter}: Loss: {loss_val:.2f}')
