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
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
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
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-13.txt",
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
        "env_file": "envs/simple_grid/gridworld-four-rooms-trap-at-doors-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
        "make_random": True,
        "max_steps": 512,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": None,
        "goal_position": None,
        "num_random_traps": 0,
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
        "agent_position": (1, 3),
        "goal_position": (3, 3),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-traps-7.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (5, 1),
        "goal_position": (1, 5),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 256,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-corridors-traps-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (1, 1),
        "goal_position": (11, 1),
        "num_random_traps": 0,
        "make_random": False,
        "max_steps": 256,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-7.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (5, 1),
        "goal_position": (1, 5),
        "num_random_traps": 3,
        "make_random": True,
        "max_steps": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-maze-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (1, 11),
        "num_random_traps": 5,
        "make_random": True,
        "max_steps": 512,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-two-rooms-7.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (5, 1),
        "goal_position": (1, 5),
        "num_random_traps": 3,
        "make_random": True,
        "max_steps": 128,
    },
    {
        "env_type": "SimpleGridworld",
        "env_file": "envs/simple_grid/gridworld-four-rooms-13.txt",
        "cell_size": None,
        "obs_size": None,
        "agent_position": (11, 1),
        "goal_position": (2, 11),
        "num_random_traps": 5,
        "make_random": True,
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
        "make_random": True,
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
        "make_random": True,
        "max_steps": 512,
    },
]
