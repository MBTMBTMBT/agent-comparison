if __name__ == "__main__":
    from train_baseline import *
    from utils import *
    from simple_gridworld import ACTION_NAMES
    from functools import partial

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 256,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-two-rooms-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
        # {
        #     "env_type": "SimpleGridworld",
        #     "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
        #     "cell_size": None,
        #     "obs_size": None,
        #     "agent_position": None,
        #     "goal_position": None,
        #     "num_random_traps": 5,
        #     "make_random": True,
        #     "max_steps": 512,
        # },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-many-rooms-9.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 3,
            "make_random": True,
            "max_steps": 512,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-many-rooms-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 5,
            "make_random": True,
            "max_steps": 512,
        },
    ]

    test_env_configurations = [
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-empty-traps-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": (3, 3),
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-maze-traps-7.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 256,
        },
        {
            "env_type": "SimpleGridworld",
            "env_file": "envs/simple_grid/gridworld-corridors-traps-13.txt",
            "cell_size": None,
            "obs_size": None,
            "agent_position": None,
            "goal_position": None,
            "num_random_traps": 0,
            "make_random": True,
            "max_steps": 128,
        },
    ]

    env_fns = [partial(make_env, config) for config in train_env_configurations]

    env = DummyVecEnv(env_fns)

    # policy_kwargs = dict(
    #     features_extractor_class=FlexibleImageEncoder,
    #     features_extractor_kwargs=dict(features_dim=64),
    # )

    # dir names
    base_name = "simple-gridworld-ppo-prior"
    save_dir = "saved-models"

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the newest model based on the custom base name and directory
    newest_model_path, idx = find_newest_model(base_name=base_name, save_dir=save_dir)
    if newest_model_path:
        print(f"Loading model from {newest_model_path}")
        model = PPO.load(newest_model_path, env=env, verbose=1)
    else:
        print("Creating a new model")
        model = PPO("CnnPolicy", env, policy_kwargs={"normalize_images": False}, verbose=1)  # policy_kwargs=policy_kwargs,

    for i in range(idx+1, 100):
        model.learn(total_timesteps=100000, progress_bar=True)
        save_model(model, i, base_name, save_dir)

        for config in train_env_configurations + test_env_configurations:
            test_env = make_env(config)
            obs, _ = test_env.reset()
            terminated, truncated = False, False
            rendered, action, probs = None, None, None
            count = 0
            sum_reward = 0
            trajectory = []
            rewards = [0.0, ]
            while not (terminated or truncated):
                rendered = test_env.render(mode='rgb_array')
                action, _states = model.predict(obs, deterministic=False)
                dis = model.policy.get_distribution(obs.unsqueeze(0).to(torch.device(device)))
                probs = dis.distribution.probs
                probs = probs.to(torch.device('cpu')).squeeze()
                trajectory.append((rendered, action.item(), probs))
                obs, reward, terminated, truncated, info = test_env.step(action.item())
                rewards.append(reward)
                count += 1
                sum_reward += reward
                if count >= 256:
                    break

            print("Test:", i, f"Test on {config['env_file']} completed.", "Step:", count, "Reward:", sum_reward)
            if rendered is not None and action is not None and probs is not None:
                rendered = test_env.render(mode='rgb_array')
                trajectory.append((rendered, action.item(), probs))
            save_trajectory_as_gif(trajectory, rewards, ACTION_NAMES, filename=config["env_file"].split('/')[-1] + f"_trajectory_{i}.gif")
