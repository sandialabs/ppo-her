from ray import tune
import numpy as np
import os
from copy import deepcopy

''' ----- File Imports ----- '''
from ppo_her.base.experiment import train
from ppo_her.base.config import DEFAULT_CONFIG
from ppo_her.base.run import run
from ppo_her.base.utils import get_experiment_name

if __name__ == '__main__':
    
    # Constants
    NUM_SAMPLES = 4
    
    # Experiment name
    experiment_name = get_experiment_name(__file__)
    print('Running experiment: ' + str(experiment_name))


    # Configs
    config = deepcopy(DEFAULT_CONFIG)
    config['model_config']['use_her'] = True
    config['total_timesteps'] = int(1000E3)
    config['env_config']['spawn_policy'] = 'random'
    config['model_config']['type'] = 'sac'
    config['model_config']['her_method'] = tune.grid_search(['final', 'future', 'episode'])
    config['env_config']['prey_velocity'] = 'static'
    config['model_config']['num_her_samples'] = tune.grid_search([1, 2, 4, 8, 16])
    config['model_config']['learning_rate'] = tune.grid_search([3E-3, 3E-4, 3E-5])

    # Number of times to run
    config['model_config']['experiment_number'] = tune.grid_search(np.arange(NUM_SAMPLES))

    # Run
    run(
        train, num_samples=1,
        config=config, num_concurrent=None,
        num_nodes=None,
        experiment_name=experiment_name,
    )

