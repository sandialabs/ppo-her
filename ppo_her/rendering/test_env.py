from ppo_her.base.env import GoalEnv
from ppo_her.base.config import ENV_CONFIG
from ppo_her.rendering.render import Render3d
from pprint import pprint



def test_velocity(prey_velocity, spawn_policy):
    '''
    Test that the experiment works so that we do not error out in parallel
    
    Parameters:
    For all parameters, see env for documentation
    prey_velocity (str) - the policy for prey actions
    spawn_policy (str) - the policy for spawning the prey and predator
    '''
    env_config = ENV_CONFIG
    env_config['prey_velocity'] = prey_velocity
    env_config['spawn_policy'] = spawn_policy
    
    env = GoalEnv(env_config)
    obs, _ = env.reset()
    done = False
    observations = [dict(obs)]
    while not done:
        action = env.action_space.sample()[0]
        obs, rew, terminate, truncate, info = env.step(action)
        pprint(obs)
        done = (terminate | truncate)
        observations.append(dict(obs))

    r = Render3d(observations, gif_name=(prey_velocity + '_' + spawn_policy + '.gif'))




if __name__ == '__main__':
    for pv in ['static', 'random', 'attract', 'repel', 'straight_away', 'random_direction']:
        for sp in ['apart', 'random']:
            test_velocity(pv, sp)
