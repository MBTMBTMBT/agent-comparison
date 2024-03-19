if __name__ == "__main__":
    from stable_baselines3 import PPO
    from configurations import *
    from utils import *
    from functools import partial
    from stable_baselines3.common.env_util import DummyVecEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # repeat the trained envs, this may help increase randomization
    rep = 2
    _train_env_configurations = train_env_configurations
    train_env_configurations = []
    for _ in range(rep):
        train_env_configurations += _train_env_configurations

    env_fns = [partial(make_env, config) for config in train_env_configurations]

    env = DummyVecEnv(env_fns)

    # dir names
    base_name = "simple-gridworld-ppo"
    save_dir = "saved-models"

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # get callbacks
    test_and_log_callback = TestAndLogCallback(
        test_env_configurations,
        base_name+"-log",
        n_eval_episodes=10,
        eval_freq=2500,
        deterministic=False,
        render=False,
        verbose=1,
    )

    model = PPO("CnnPolicy", env, policy_kwargs={"normalize_images": False}, verbose=1)
    model.learn(total_timesteps=5000000, callback=[test_and_log_callback], progress_bar=True)
    save_model(model, 0, base_name, save_dir)
