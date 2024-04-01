from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    import torch
    from configurations import maze13_sampling
    from feature_model import FeatureNet

    CONFIGS = maze13_sampling
    NUM_ACTIONS = 4
    LATENT_DIMS = 2

    SAMPLE_SIZE = 4096
    MAX_SAMPLE_STEP = 4096
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 100

    session_name = "test"
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
        counter, performance = model.load(latest_checkpoint)
        counter += 1
    else:
        counter = 0
        performance = float('inf')

    from env_sampler import TransitionBuffer, RandomSampler
    from utils import make_env

    progress_bar = tqdm(range(EPOCHS), desc=f'Training Epoch {counter}')
    for i, batch in enumerate(progress_bar):
        sampler = RandomSampler()
        envs = [make_env(config) for config in CONFIGS]
        while len(sampler.transition_pairs) < SAMPLE_SIZE:
            for env in envs:
                sampler.sample(env, MAX_SAMPLE_STEP)

        transition_buffer = TransitionBuffer(sampler.transition_pairs)
        dataloader = DataLoader(transition_buffer, batch_size=BATCH_SIZE, shuffle=True)
        for x0, a, x1 in dataloader:
            x0 = x0.to(device)
            a = a.to(device)
            x1 = x1.to(device)
            model.train_batch(x0, x1, a)

        encoder = model.phi
        for config, env in zip(CONFIGS, envs):
            env_path = config['env_file']
            env_name = env_path.split('/')[-1].split('.')[0]

        model.save(f"{session_name}/model_epoch_{counter}.pth", counter, performance)
        counter += 1
