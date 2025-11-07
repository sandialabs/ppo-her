import numpy as np

'''
    Default configuration files:
        ENV_CONFIG - parameters that affect the environment.  Most of these
            parameters are for the Predator-Prey environments and don't affect
            Fetch
        MODEL_CONFIG - parameters that affect the model, including HER
        DEFAULT_CONFIG - all parameters combined
'''

ENV_CONFIG = {
    'NUM_DIMS': 3, # number of physical dimensions 
    'GRID_SIZE': 10, # the size of the environment.  X and Y directions are +- GRID_SIZE, Z direction is 0-GRIDSIZE
    'AGENT_AT_TARGET_DISTANCE': 1, # the distance at which the agent is considered to be "at the target"
    'MAX_EPISODE_STEPS': 20, # the number of steps that the agent can take to reach the target
    'prey_velocity': 'static', # static, random, attract, repel, straight_away, random_direction
    'spawn_policy': 'random', # random, apart
}

MODEL_CONFIG = {
    'type': 'ppo', # ppo, sac
    'use_her': True, # True, False
    'her_method': 'final', # final, future, random, episode, maximum_entropy
    'her_action_on_stop': 'me', # me, stop
    'her_failure_method': 'random', # random, targeted
    'num_her_samples': 1, # This is "k" in the HER (Andrychowicz 2018) paper
    'learning_rate': 3E-4, # learning rate
    'experiment_number': 0, # When we repeat an experiment, each experiment gets its own number
    'stop_her_mode': 'reward', #'reward', 'steps' - stop HER if we reach a given reward or a given number of steps
    'stop_her': np.infty, # the threshold at which we stop HER.  Works with stop_her_mode. 
    'success_fraction': 0.5, #[0, 1] - the fraction of successes that we are aiming for when using HER.  Used mostly for maximum entropy HER
    'stop_experiment_on_success': True, # Stop running experiments when the average reward reaches 1
    'target_kl': 0.05, # StableBaselines3 target_kl
    'ent_coef': 0.00, # StableBaselines3 ent_coef
    'policy_kwargs': None, # StableBaselines3 policy_kwargs
    'only_her_if_move': False, # Only add to HER buffer if the achieved goal changes.  Used for MEHER experiments
    'l2_coeff': 0, # StableBaselines3 l2_coeff
    'n_steps': 2048, # For PPO ONLY, the number of steps to take before training
    }

DEFAULT_CONFIG = {
    'env_config': ENV_CONFIG,
    'model_config': MODEL_CONFIG,
    'total_timesteps': int(1000E3), # Total timesteps to train for
    'env_type': 'predator_prey', # predator_prey, Fetch<fetch_type><version> - the environment to train on
}
