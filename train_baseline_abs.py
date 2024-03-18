if __name__ == "__main__":
    from stable_baselines3 import PPO
    from configurations import train_env_configurations, test_env_configurations
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
    base_name = "simple-gridworld-ppo-abs-half"
    save_dir = "saved-models"

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # get callbacks
    test_and_log_callback = TestAndLogCallback(
        test_env_configurations,
        base_name+"-log",
        n_eval_episodes=10,
        eval_freq=5000,
        deterministic=False,
        render=False,
        verbose=1,
    )

    update_env_callback = UpdateEnvCallback(
        train_env_configurations,
        num_clusters=45,
        update_env_freq=1000,
        update_agent_freq=20000,
        verbose=1,
        abs_rate=0.5,
    )

    model = PPO("CnnPolicy", env, policy_kwargs={"normalize_images": False}, verbose=1)
    model.learn(total_timesteps=3000000, callback=[test_and_log_callback, update_env_callback], progress_bar=True)
    save_model(model, 0, base_name, save_dir)
