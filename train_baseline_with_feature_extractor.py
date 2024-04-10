from feature_wrapper import FeatureWrapper

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import math
    import torch
    from configurations import mix_sampling
    from feature_model import FeatureNet
    from utils import find_latest_checkpoint
    import os
    from env_sampler import TransitionBuffer, RandomSampler, SamplerWrapper
    from utils import make_env
    from stable_baselines3 import PPO
    from configurations import *
    from utils import *
    from functools import partial
    from stable_baselines3.common.env_util import DummyVecEnv

    TRAIN_CONFIGS = mix_sampling
    TEST_CONFIGS = mix_sampling

    # model configs
    NUM_ACTIONS = 4
    LATENT_DIMS = 8
    RECONSTRUCT_SIZE = (72, 72)
    RECONSTRUCT_SCALE = 2

    # sampler configs
    SAMPLE_SIZE = 16384
    SAMPLE_REPLAY_TIME = 4
    MAX_SAMPLE_STEP = 4096
    VEC_ENV_REPEAT_TIME = 1

    # train hyperparams
    WEIGHTS = {
        'inv': 1.0,
        'dis': 1.0,
        'dec': 0.1,
    }
    BATCH_SIZE = 256
    LR = 1e-4

    # train configs
    PRE_TRAIN_STEPS = SAMPLE_SIZE * SAMPLE_REPLAY_TIME * 50
    SAVE_FREQ = SAMPLE_SIZE * SAMPLE_REPLAY_TIME * 1

    EPOCHS = 1000
    NUM_EPOCH_STEPS = MAX_SAMPLE_STEP * 4

    # eval configs
    NUM_EVAL_EPISODES = 16
    EVAL_FREQ = MAX_SAMPLE_STEP * 1

    session_name = "learn_feature_reconstruct"
    feature_model_name = 'feature_model_step'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, img_size=RECONSTRUCT_SIZE,
                                   initial_scale_factor=RECONSTRUCT_SCALE, device=device).to(device)

    if not os.path.isdir(session_name):
        os.makedirs(session_name)
    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name)

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

    # init tensorboard writer
    tb_writer = SummaryWriter(log_dir=session_name)

    print("Start pre-training:")

    # Initialize the progress bar
    pbar = tqdm(total=MAX_SAMPLE_STEP, initial=feature_extractor_step_counter, desc="Pre-Train Progress")

    while feature_extractor_step_counter < MAX_SAMPLE_STEP:
        sampler = RandomSampler()
        envs = [make_env(config) for config in TRAIN_CONFIGS]
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
                loss_val, inv_loss_val, ratio_loss_val, pixel_loss_val = feature_extractor.train_batch(x0, x1, a)
                tb_writer.add_scalar('loss', loss_val, feature_extractor_step_counter)
                tb_writer.add_scalar('inv_loss', inv_loss_val, feature_extractor_step_counter)
                tb_writer.add_scalar('ratio_loss', ratio_loss_val, feature_extractor_step_counter)
                tb_writer.add_scalar('pixel_loss', pixel_loss_val, feature_extractor_step_counter)

                # Update the progress bar description with the latest loss values
                pbar.set_description(
                    f"Progress - Loss: {loss_val:.4f}, Inv: {inv_loss_val:.4f}, Ratio: {ratio_loss_val:.4f}, Pixel: {pixel_loss_val:.4f}")
                pbar.update(1)  # Assuming each batch represents a single step, adjust as necessary

                feature_extractor_step_counter += 1
                if feature_extractor_step_counter >= MAX_SAMPLE_STEP:
                    break
            if feature_extractor_step_counter >= MAX_SAMPLE_STEP:
                break

        if feature_extractor_step_counter % SAVE_FREQ == 0:
            _save_name = f"{session_name}/{feature_model_name}_{feature_extractor_step_counter}.pth"
            print(f"Saving model with name {_save_name}")
            feature_extractor.save(_save_name, epoch_counter, feature_extractor_step_counter, performance)
    pbar.close()

    _train_env_configurations = TRAIN_CONFIGS
    train_env_configurations = []
    for _ in range(VEC_ENV_REPEAT_TIME):
        train_env_configurations += _train_env_configurations

    env_fns = [partial(make_env, config, SamplerWrapper) for config in train_env_configurations]

    env = FeatureWrapper(DummyVecEnv(env_fns), feature_extractor.phi, device)

    # Check for the latest saved model
    model_path, epoch_counter, agent_num_steps = find_latest_checkpoint(session_name)

    # get callbacks
    step_counter_callback = StepCounterCallback()

    test_and_log_callback = TestAndLogCallback(
        trap_31_test,
        session_name,
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        start_num_steps=agent_num_steps,
        deterministic=False,
        render=False,
        verbose=1,
    )

