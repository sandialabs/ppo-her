from stable_baselines3 import HerReplayBuffer
from ppo_her.base.custom_sac import CustomSAC
from ppo_her.base.custom_ppo import CustomPPO
from os.path import join, abspath


'''
    Return the RL model, given a config
    Inputs:
        config (dict) - the model configuation.  See config.py for an example
        env (gymnasium.Env) - the RL environment
    Returns:
        model (stable_baselines3) - the RL model
'''
def get_model(config, env):

    tensorboard_log = join(
        '../sb3_results',
        config['experiment_name'],
        config['ray_experiment'],
    )
    tensorboard_log = abspath(tensorboard_log)
    print(tensorboard_log)

    if config['policy_kwargs'] is not None:
        policy_kwargs = config['policy_kwargs']
    else:
        policy_kwargs = {}
        policy_kwargs['net_arch'] = [64, 64]
    policy_kwargs['net_arch'] = [512, 512, 512]

    if config['type'].lower() == 'sac':

        policy_setup = {
            'stop_her_mode': config['stop_her_mode'],
            'stop_her': config['stop_her'],
        }

        if config['use_her']:
            replay_buffer_class=HerReplayBuffer
            replay_buffer_kwargs=dict(
                n_sampled_goal=config['num_her_samples'],
                goal_selection_strategy=config['her_method'],
                copy_info_dict=True,
            )
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None

        batch_size_dict = {
            'Push': 2048,
            'Slide': 2048,
            'PickAndPlace': 1024,
            'Reach': 256
        }

        batch_size = 256
        for key, value in batch_size_dict.items():
            if key in str(type(env.unwrapped)):
                batch_size = value
                break

        if 'Reach' in str(type(env.unwrapped)):
            tau = 0.005
        else:
            tau = 0.05

        model = CustomSAC(
            'MultiInputPolicy',
            env,
            tensorboard_log=tensorboard_log,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            learning_rate=config['learning_rate'],
            policy_kwargs=policy_kwargs,
            policy_setup=policy_setup,
            buffer_size=int(1E6),
            batch_size=batch_size,
            gamma=0.95,
            tau=tau,
            verbose=2,
        )
    elif config['type'].lower() == 'ppo':
    
        policy_setup = {
            'num_dims': config['num_dims'],
            'use_her': config['use_her'],
            'num_her_samples': config['num_her_samples'],
            'her_action_on_stop': config['her_action_on_stop'],
            'her_failure_method': config['her_failure_method'],
            'her_method': config['her_method'],
            'reward_info': None,
            'stop_her_mode': config['stop_her_mode'],
            'stop_her': config['stop_her'],
            'success_fraction': config['success_fraction'],
            'stop_experiment_on_success': config['stop_experiment_on_success'],
            'only_her_if_move': config['only_her_if_move'],
        }

        try:
            n_steps=config['n_steps']
        except:
            print('Could not get n_steps')
            n_steps = None

        policy_kwargs['optimizer_kwargs'] = {'weight_decay': config['l2_coeff']}
        model = CustomPPO(
            'MultiInputPolicy',
            env,
            policy_setup,
            tensorboard_log=tensorboard_log,
            clip_range_vf=0.2,
            learning_rate=config['learning_rate'],
            policy_kwargs=policy_kwargs,
            target_kl=config['target_kl'],
            ent_coef=config['ent_coef'],
            verbose=0,
            n_steps=n_steps,
        )

    return model
