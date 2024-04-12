train_env_configurations = [
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-empty-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": None,
    #     "goal_position": None,
    #     "num_random_traps": 3,
    #     "make_random": True,
    #     "max_steps": 128,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 32,
    },
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-maze-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": None,
    #     "goal_position": None,
    #     "num_random_traps": 3,
    #     "make_random": True,
    #     "max_steps": 128,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
    },
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-two-rooms-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": None,
    #     "goal_position": None,
    #     "num_random_traps": 3,
    #     "make_random": True,
    #     "max_steps": 128,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-many-rooms-9.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (7, 1),
        # "goal_position": (1, 7),
        "num_random_traps": 3,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-many-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 64,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 64,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (11, 1),
        # "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 32,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 32,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-rooms-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 384,
    },
]
test_env_configurations = [
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-empty-traps-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": (1, 3),
    #     "goal_position": (3, 3),
    #     "num_random_traps": 0,
    #     "make_random": False,
    #     "max_steps": 128,
    # },
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-maze-traps-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": (5, 1),
    #     "goal_position": (1, 5),
    #     "num_random_traps": 0,
    #     "make_random": False,
    #     "max_steps": 256,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (1, 1),
        "goal_position": (11, 1),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
    },
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-maze-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": (5, 1),
    #     "goal_position": (1, 5),
    #     "num_random_traps": 0,
    #     "make_random": False,
    #     "max_steps": 128,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
    },
    # {
    #     "env_type": "SimpleGridworld",
    #     "env_file": "envs/simple_grid/gridworld-two-rooms-7.txt",
    #     "cell_size": None,
    #     "obs_size": None,
    #     "agent_position": (5, 1),
    #     "goal_position": (1, 5),
    #     "num_random_traps": 0,
    #     "make_random": False,
    #     "max_steps": 128,
    # },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (2, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (11, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 32,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 32,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-rooms-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
        "num_clusters": 384,
    },
]

maze13_train = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
        "do_abs": True,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
        "do_abs": True,
    },
]

maze13_test = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
]

four_rooms_train = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
        "do_abs": True,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 48,
        "do_abs": True,
    },
]

four_rooms_test = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
]

rooms_31_train = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-rooms-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 333,
        "do_abs": True,
    },
]

rooms_31_test = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-rooms-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
]

trap_31_train = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 256,
        "do_abs": True,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 256,
        "do_abs": True,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-31.txt",
        "cell_size": None,
        "obs_size": None,
        # "agent_position": (29, 1),
        # "goal_position": (1, 29),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 2048,
        "num_clusters": 32,
        "do_abs": False,
    },
]

trap_31_test = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-31.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (29, 1),
        "goal_position": (1, 29),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 512,
        "do_abs": False,
    },
]

maze13_sampling = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": None,
        "goal_position": (2, 10),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 4096,
        "num_clusters": 48,
    },
]

maze13_sampling_rand = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
]

four_room13_sampling = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
]

mix_sampling = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 4096,
        "num_clusters": 48,
    },
]

mix_sampling_eval = [
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 1024,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 1024,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-empty-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 1024,
        "num_clusters": 48,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
        "cell_size": None,
        "obs_size": (96, 96),
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 1024,
        "num_clusters": 48,
    },
]
