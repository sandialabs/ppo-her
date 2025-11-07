from gymnasium import Env, spaces
import numpy as np
from ppo_her.base.config import ENV_CONFIG
from copy import deepcopy


class GoalEnv(Env):
    '''
    The Predator-Prey environment

    Parameters:
    config (dict): parameters for setting up the environment containing the following fields:
        NUM_DIMS (int) - number of physical dimensions 
        GRID_SIZE (float) - the size of the environment.  X and Y directions are +- GRID_SIZE, Z direction is 0-GRIDSIZE
        AGENT_AT_TARGET_DISTANCE (float) - the distance at which the agent is considered to be "at the target"
        MAX_EPISODE_STEPS (int) - the number of steps that the agent can take to reach the target
        prey_velocity (str) - 
            static - the prey does not move
            random - the prey takes a random walk
            attract - the prey moves towards (but not directly towards) the predator
            repel - the prey moves away (but not always directly away) from the predator
            straight_away - the prey always moves directly away from the predator
            random_direction - the prey chooses a direction, at random, at the beginning of the episode
                and moves in that direction for the rest of the episode
        spawn_policy (str) - how the initial positions are determined
            random - prey and predator initial positions are drawn from uniform distributions
            apart - prey and predator positions are kept constant
    '''

    def __init__(self, configs=None):

        # Store the configs
        if configs is None:
            configs = ENV_CONFIG

        assert isinstance(configs, dict), 'Configs should be a dict'
        for key, value in configs.items():
            self.__dict__[key] = value

        self.define_spaces()

    def define_spaces(self):
        min_action = np.ones(self.NUM_DIMS) * -1
        max_action = np.ones(self.NUM_DIMS)
        self.action_space = spaces.Box(min_action, max_action, dtype=np.float32)

        min_observation = -1 * self.GRID_SIZE * np.ones(self.NUM_DIMS)
        max_observation = self.GRID_SIZE * np.ones(self.NUM_DIMS)
        desired_goal = spaces.Box(min_observation, max_observation, dtype=np.float32)
        achieved_goal = spaces.Box(min_observation, max_observation, dtype=np.float32)
        observation = spaces.Box(min_observation, max_observation, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'desired_goal': desired_goal,
            'achieved_goal': achieved_goal,
            'observation': observation,
        })

    def compute_reward(self, achieved_goal, desired_goal, info):
        num_steps = np.shape(desired_goal)[0]
        rewards = np.zeros(num_steps)
        for num, (ag, dg, i) in enumerate(zip(achieved_goal, desired_goal, info)):
            rewards[num], _ = self._compute_reward(ag, dg, i)
        
        return rewards

    
    def _compute_reward(self, achieved_goal, desired_goal, info):
        
        at_target = self.get_at_target(desired_goal, achieved_goal)
        
        # "Sparse"rewards - 1 if at target, 0 if not at target, -1 if fail (timeout)
        if (at_target):
            done = True
            reward = 1
        elif (info['episode_steps'] > self.MAX_EPISODE_STEPS):
            done = True
            reward = -1            
        else:
            done = False
            reward = 0

        return reward, done


    def step(self, action):    
        # Increment episode steps    
        self.episode_steps += 1
        info = {
            'episode_steps': self.episode_steps
        }

        # Step the predator
        self.observation['achieved_goal'] = np.clip(
            self.observation['achieved_goal'] + action,
            -1 * self.GRID_SIZE,
            self.GRID_SIZE,
        )

        # See if the predator caught the prey
        reward, done = self._compute_reward(
            self.observation['achieved_goal'],
            self.observation['desired_goal'],
            info
        )
        terminate = truncate = done

        # Step the prey
        self.observation['desired_goal'] = self.observation['desired_goal'] + self.observation['observation']
        self.observation['desired_goal'] = np.clip(
            self.observation['desired_goal'],
            -1 * self.GRID_SIZE,
            self.GRID_SIZE,
        )
        
        prey_velocity = self.get_prey_velocity(self.observation, is_first_step=False)
        self.observation['observation'] = prey_velocity

        return self.observation, reward, terminate, truncate, info

    def get_prey_velocity(self, observation, is_first_step):
        '''
        This implements the Prey policy (static, attract, repel, etc...)
        '''


        max_predator_speed = np.linalg.norm(np.ones(self.NUM_DIMS))
        max_prey_speed = max_predator_speed / 2
        prey_pred_direction = observation['achieved_goal'] - observation['desired_goal']

        def get_orthogonal_vector(vector_in):
            # Initialize a random vector
            vector_out = np.random.uniform(0, 1, size=self.NUM_DIMS)

            # Find where the input vector is non-zero to avoid divide by infinity
            is_non_zero = np.where(vector_in != -0)[0]
            if len(is_non_zero) > 0:
                chosen_location = np.random.choice(is_non_zero)
            else:
                # If our velocity vector is zero, we choose an orthogonal vector at random
                # and assign a random value to vector_in
                vector_in = np.random.uniform(0.01, 1, self.NUM_DIMS)
                chosen_location = np.random.choice(self.NUM_DIMS)

            # Create the ortogonal vector
            vector_out[chosen_location] = -1 * np.dot(
                vector_in[:-1], vector_out[:-1]) / vector_in[chosen_location]
            
            # Make sure that it has the same magnitude
            vec_in_mag = np.linalg.norm(vector_in)
            vec_out_mag = np.linalg.norm(vector_out)
            vector_out = vec_in_mag * vector_out / vec_out_mag


            if np.any(np.abs(vector_out) == np.infty):
                raise ValueError('Infinite values detected!')
            return vector_out

        if self.prey_velocity == 'static':
            # The prey remains still
            prey_velocity = np.zeros(self.NUM_DIMS)
            return prey_velocity
        elif self.prey_velocity == 'random':
            # The prey moves with random speed and direction
            prey_speed = np.random.uniform(0, max_prey_speed)
            prey_direction = np.random.uniform(-1, 1, self.NUM_DIMS)
        elif self.prey_velocity == 'attract':
            # The prey moves towards the predator at max velocity
            # So that we don't move directly towards the predator, at noise between
            # APPROXIMATELY 10 and 90 degrees
            prey_speed = max_prey_speed
            orthogonal_vector = get_orthogonal_vector(prey_pred_direction)

            # Mix the two orthogonal vectors
            mixing_factor = np.random.uniform(0.1, 1)
            prey_direction = mixing_factor * orthogonal_vector + (1 - mixing_factor) * prey_pred_direction
            
        elif self.prey_velocity == 'repel':
            # The prey moves away from the predator, but not in a prefectly straight line
            prey_speed = max_prey_speed
            orthogonal_vector = get_orthogonal_vector(-1 * prey_pred_direction)

            # Mix the two orthogonal vectors
            mixing_factor = np.random.uniform(0, 1)
            prey_direction = mixing_factor * orthogonal_vector + (1 - mixing_factor) * prey_pred_direction * -1
        elif self.prey_velocity == 'straight_away':
            # The prey moves away from the predator in a straight line
            prey_speed = max_prey_speed
            prey_direction = -1 * prey_pred_direction
        elif self.prey_velocity == 'random_direction':
            # The prey moves in a random direction, which may be towards or away from the predator
            if is_first_step:
                prey_speed = max_prey_speed
                prey_direction = np.random.uniform(-1, 1, self.NUM_DIMS)
            else:
                prey_speed = np.linalg.norm(observation['observation'])
                prey_direction = np.copy(observation['observation'])
        else:
            raise ValueError('Unknown prey velocity type: ' + str(self.prey_velocity))
        
        if np.all(prey_direction == 0):
            # Avoid divide-by-zero error
            prey_velocity = prey_direction
        else:
            prey_velocity = prey_speed * prey_direction / np.linalg.norm(prey_direction)

        
        if np.any(np.isnan(prey_velocity)):
            ValueError('Prey Velocity cannot be NaN!')

        return prey_velocity
    
    def get_at_target(self, prey_location, pred_location):
        # Determine if the Predator and Prey positions are sufficiently similar to call it a collision
        at_target = np.linalg.norm(prey_location - pred_location) <= self.AGENT_AT_TARGET_DISTANCE
        return at_target
        
    def get_initial_locations(self):
        observation = self.observation_space.sample()

        # Spawn the Predator and Prey into the environment at random, non-overlapping locations
        if self.spawn_policy == 'random':

            # Ensure that the two positions do not overlap
            at_target = self.get_at_target(
                observation['desired_goal'], observation['achieved_goal'])
            while at_target:
                observation = self.observation_space.sample()
                at_target = self.get_at_target(
                    observation['desired_goal'], observation['achieved_goal'])
                
            pred_location = observation['achieved_goal']
            prey_location = observation['desired_goal']

        # Spawn the predator and prey in the environment at set locations
        elif self.spawn_policy == 'apart':
            pred_location = np.zeros(self.NUM_DIMS)
            prey_location = np.zeros(self.NUM_DIMS)
            prey_location[-1] = self.GRID_SIZE / 2

        else:
            raise ValueError('Unknown spawn policy: ' + str(self.spawn_policy))
        
        return pred_location, prey_location

    def reset(self, seed=None, options=None):
        self.episode_steps = 0
        
        # Set the initial positions
        self.observation = self.observation_space.sample()
        pred_location, prey_location = self.get_initial_locations()
        self.observation['desired_goal'] = prey_location
        self.observation['achieved_goal'] = pred_location

        # Right now, we do not use the observation, but we need something to be
        # compatible with SB3.  Give 0 so that we do not have uncorrelated noise
        prey_velocity = self.get_prey_velocity(self.observation, is_first_step=True)
        self.observation['observation'] = prey_velocity

        info = {}
        return self.observation, info


if __name__ == '__main__':
    '''
    Debugging code to ensure that changes to environment don't cause immediate crashes
    '''
    from ppo_her.base.config import ENV_CONFIG

    for pv in ['static', 'random', 'attract', 'repel', 'straight_away', 'random_direction']:
        for sp in ['random', 'apart']:
            env_config = deepcopy(ENV_CONFIG)
            env_config['prey_velocity'] = pv
            env_config['spawn_policy'] = sp

            env = GoalEnv(env_config)
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, rew, terminate, truncate, info = env.step(action)
                done = (terminate | truncate)
            print(pv + ' | ' + sp + ' | ' + 'Done!')






