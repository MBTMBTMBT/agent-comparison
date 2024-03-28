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

    session_name = "saved-models"
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

    sampler = RandomSampler()
    envs = [make_env(config) for config in CONFIGS]
    while len(sampler.transition_pairs) < SAMPLE_SIZE:
        for env in envs:
            sampler.sample(env, MAX_SAMPLE_STEP)

    transition_buffer = TransitionBuffer(sampler.transition_pairs)
