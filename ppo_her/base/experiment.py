import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from copy import deepcopy
import gymnasium
from ray import tune

''' ----- File Imports ----- '''
from ppo_her.base.env import GoalEnv
from ppo_her.base.get_model import get_model
from ppo_her.base.config import DEFAULT_CONFIG
from ppo_her.base.utils import get_name

def train(config):

    # Create the environment
    if config['env_type'] == 'predator_prey':
        env = GoalEnv(configs=config['env_config'])
    else:
        env = gymnasium.make(config['env_type'])

    experiment_name = get_name(config) 
    
    # Get Ray experiment number
    try:
        config['model_config']['ray_experiment'] = str(tune.get_trial_id())
    except:
        config['model_config']['ray_experiment'] = 'NoTune'

    # Create the model
    
    config['model_config']['experiment_name'] = experiment_name    
    config['model_config']['num_dims'] = config['env_config']['NUM_DIMS']
    model = get_model(config['model_config'], env)

    # Callback - evaluation and stop training early - only for SAC
    callback = None
    if config['model_config']['type'] == 'SAC':
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1)
        eval_callback = EvalCallback(deepcopy(env), callback_on_new_best=callback_on_best)
        callback = eval_callback

    # Train the model
    model.learn(total_timesteps=config['total_timesteps'], callback=callback)

    # We need to return some score for Ray.tune
    # We return a score of zero
    score = {}
    score['episode_reward_mean'] = 0
    return score

