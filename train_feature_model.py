if __name__ == '__main__':
    from configurations import maze13_sampling

    CONFIGS = maze13_sampling
    SAMPLE_SIZE = 4096
    MAX_SAMPLE_STEP = 4096

    from env_sampler import TransitionBuffer, RandomSampler
    from utils import make_env

    sampler = RandomSampler()
    envs = [make_env(config) for config in CONFIGS]
    while len(sampler.transition_pairs) < SAMPLE_SIZE:
        for env in envs:
            sampler.sample(env, MAX_SAMPLE_STEP)

    transition_buffer = TransitionBuffer(sampler.transition_pairs)
