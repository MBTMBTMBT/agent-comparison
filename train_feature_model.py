from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from simple_gridworld import SimpleGridWorld

if __name__ == '__main__':
    import torch
    from configurations import maze13_sampling
    from feature_model import FeatureNet

    CONFIGS = maze13_sampling
    NUM_ACTIONS = 4
    LATENT_DIMS = 2

    SAMPLE_SIZE = 10240
    SAMPLE_REPLAY_TIME = 4
    MAX_SAMPLE_STEP = 4096
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 500
    SAVE_FREQ = 10
    TEST_FREQ = 10

    session_name = "learn_feature"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureNet(4, n_latent_dims=LATENT_DIMS, lr=LR, device=device).to(device)

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
                        observation = torch.unsqueeze(observation, dim=0).to(device)  # pad one dimension to the beginning
                        with torch.no_grad():
                            z = encoder(observation).detach().cpu().numpy()
                            z_vectors.append(z.squeeze(0))
                # After collecting all z vectors, plot them
                plt.figure(figsize=(8, 8))
                for z in z_vectors:
                    plt.scatter(z[0], z[1])  # Assuming z is a 2D vector

                plt.title(f"Encoded Env: {env_name}")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")

                # Save the plot
                img_save_dir = os.path.join(session_name, "saved_encoded_images")
                if not os.path.isdir(img_save_dir):
                    os.makedirs(img_save_dir)
                save_path = os.path.join(img_save_dir, f"{env_name}_{counter}.png")
                plt.savefig(save_path)
                plt.close()  # Close the plot figure

        if counter % SAVE_FREQ == 0:
            model.save(f"{session_name}/model_epoch_{counter}.pth", counter, _counter, performance)
        counter += 1
        progress_bar.set_description(
            f'Train Epoch {counter}: Loss: {loss_val:.2f}')
