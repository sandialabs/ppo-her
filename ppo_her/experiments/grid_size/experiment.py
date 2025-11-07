from ray import tune
import numpy as np
import os
from copy import deepcopy

''' ----- File Imports ----- '''
from ppo_her.base.env import GoalEnv
from ppo_her.base.get_model import get_model
from ppo_her.base.config import DEFAULT_CONFIG
from ppo_her.base.run import run
from ppo_her.base.utils import get_name
from ppo_her.base.utils import get_experiment_name
from ppo_her.base.experiment import train

NUM_SAMPLES = 16

if __name__ == '__main__':
    
    # Experiment name
    experiment_name = get_experiment_name(__file__)
    print('Running experiment: ' + str(experiment_name))

    # Configs
    config = deepcopy(DEFAULT_CONFIG)
    config['model_config']['use_her'] = tune.grid_search([True, False])
    config['total_timesteps'] = int(1000E3)
    config['env_config']['spawn_policy'] = tune.grid_search(['random', 'apart'])
    config['model_config']['type'] = 'ppo'
    config['model_config']['her_method'] = 'final'
    config['env_config']['prey_velocity'] = tune.grid_search(
        ['static', 'random', 'attract', 'repel', 'straight_away', 'random_direction']
    )

    # Number of times to run
    config['model_config']['experiment_number'] = tune.grid_search(np.arange(NUM_SAMPLES))

    # Grid Size
    config['env_config']['GRID_SIZE'] = tune.grid_search(np.arange(10, 51, 10))

    # Run
    run(
        train, num_samples=1,
        config=config, num_concurrent=None,
        num_nodes=None,
        experiment_name=experiment_name,
    )

