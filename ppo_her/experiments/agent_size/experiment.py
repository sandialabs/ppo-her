from ray import tune
import numpy as np
import os
from copy import deepcopy

''' ----- File Imports ----- '''
from ppo_her.base.experiment import train
from ppo_her.base.config import DEFAULT_CONFIG
from ppo_her.base.run import run
from ppo_her.base.utils import get_experiment_name

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
    config['env_config']['AGENT_AT_TARGET_DISTANCE'] = tune.grid_search([1, 0.5, 0.2, 0.1])

    # Number of times to run
    config['model_config']['experiment_number'] = tune.grid_search(np.arange(NUM_SAMPLES))

    # Run
    run(
        train, num_samples=1,
        config=config, num_concurrent=None,
        num_nodes=None,
        experiment_name=experiment_name,
    )

