from stable_baselines3 import PPO
import time
import sys
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.buffers import DictRolloutBuffer
import numpy as np
import torch as th
from copy import deepcopy

def get_dict_idxs(dict_in, indices):
    """
    Extracts specific elements from the values of a dictionary based on the provided indices.

    Parameters:
    dict_in (dict): A dictionary where each value is an array.
    indices (np.ndarray or list): An array or list of indices to select elements from the arrays in the dictionary.
                                  If the indices are of boolean type, the function converts them to integer indices.

    Returns:
    new_dict: A new dictionary containing the same keys as dict_in, but with values that are NumPy arrays
          containing only the elements specified by the indices.
    """
    new_dict = {}
    if indices.dtype == 'bool':
        indices = np.where(indices)[0]
    for key, value in dict_in.items():
        new_dict[key] = np.take(value, indices, axis=0)
    return new_dict

class CustomPPO(PPO):
    '''
    Version of StableBaselines3 PPO that allows for HER.

    Parameters:
    Parameters that are unchanged from StableBaselines are not listed.
    policy_setup (dict): parameters for setting up HER, containing the following fields:
        'num_dims' (int) - the number of physical dimensions
        'use_her' (bool) - whether we should use HER or not
        'num_her_samples' (int) - the number of HER samples to add to the buffer
        'her_action_on_stop' (str) - when we reach the threshold performance, we modify HER.
            We can either:
                'stop' or 
                'me' - switch to maximum entropy
        'her_failure_method' (str) - for MEHER, we need to determine how we add, failurs to the buffer:
            'random' - choose goals at random
            'targeted' - choose goals that are near failed goal
        'her_method' (str) - the method that we use to generate HER samples (see Andrychowicz 2018):
            'final' - defined in Andrychowicz 2018
            'future' - defined in Andrychowicz 2018
            'random' - defined in Andrychowicz 2018
            'episode' - defined in Andrychowicz 2018
            'maximum_entropy' - try to maximize the entropy by "balancing" successes and failures
        'reward_info' (dict) - information used to calculate the rewards.  This should be None.
        'stop_her_mode' (str) - The method that we use to determine when we should turn HER off.  Set stop_her to
            infinity to disable
                'reward' - stop when we reach a reward threshold
                'steps' - stop when we reach a certain number of steps
        'stop_her' (float) - the threshold at which we stop using HER.  The threshold is applied to the mean
            reward or the number of steps, depending on the value of stop_her_mode
        'success_fraction' (float) - for MEHER, the target proportion of successes in the HER buffer.  Range [0, 1]
        'stop_experiment_on_success' (bool) - if we reach a mean reward of 1, stops experiment if True
        'only_her_if_move' (bool) - if True, we only add HER transitions for trajectories that cause the achieved goal to move

    '''
    def __init__(
        self,
        policy,
        env,
        policy_setup,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        # Set the HER-related values
        for key, value in policy_setup.items():
            self.__dict__[key] = value

        # If the HER method is final, the num_her_samples parameter is arbitrary.  Set to 1
        if self.her_method == 'final':
            self.num_her_samples = 1
        else:
            self.num_her_samples = policy_setup['num_her_samples']

        # Check that HER methods are recognized
        valid_her_methods = ['final', 'future', 'episode', 'random', 'maximum_entropy', 'targeted']
        assert self.her_method in valid_her_methods, 'Unknown HER method:' + str(self.her_method)
        self.reward_timestep = None

    '''
    Version of PPO.learn() that let's us use HER
    '''
    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="OnPolicyAlgorithm",
        reset_num_timesteps=True,
        progress_bar=False
        ):

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        original_buffer_size = self.rollout_buffer.buffer_size

        while self.num_timesteps < total_timesteps:

            self.rollout_buffer.buffer_size = original_buffer_size
            self.rollout_buffer.reset()

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

                    # Get success for Fetch only
                    try:
                        # This is just a test to see if we are in a Predator-Prey Env
                        num_dims = self.env.unwrapped.get_attr('NUM_DIMS')
                    except:
                        max_steps = self.env.envs[0].env._saved_kwargs['max_episode_steps']
                        successes = np.abs([eib['r'] for eib in self.ep_info_buffer if eib['l'] == max_steps])
                        successes = np.mean(successes < max_steps)
                        self.logger.record("rollout/ep_success", successes)

                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            # Check for early stopping.  Note that this is currently hard-coded to stop if the
            # mean_episode_reward == 1
            mean_episode_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            if self.stop_experiment_on_success:
                if mean_episode_reward == 1:
                    if self.reward_timestep is None:
                        self.reward_timestep = self.num_timesteps
            
            # This code actually breaks if we are successful enough.  1.3 * the number of timesteps
            # ensures that we have enough time after the success to see the trends
            if self.reward_timestep is not None:
                if self.num_timesteps > (1.3 * self.reward_timestep):
                    break

            # Determine if we should stop using HER
            self.random_her = False
            take_her_action = False
            if (self.stop_her_mode == 'steps') and (self.num_timesteps > self.stop_her):
                take_her_action = True
            if (self.stop_her_mode == 'reward') and (mean_episode_reward > self.stop_her):
                take_her_action = True
           
            # Determine what to do if we are "stopping" HER
            if take_her_action:
                if self.her_action_on_stop == 'me':
                    self.random_her = True
                elif self.her_action_on_stop == 'stop':
                    self.use_her = False
                else:
                    raise ValueError('Unknown her action: ' + str(self.her_action_on_stop))

            # Implement HER
            if self.use_her:
                self.her()

            self.train()


        callback.on_training_end()

        return self

    
    def her(self):

        # Create function to get log_probs, values
        # This comes from SB3 PPO policy forward, but we keep the actions
        # that we had already chosen
        def get_vlp(obs, actions):
            with th.no_grad():
                features = self.policy.extract_features(obs)
                if self.policy.share_features_extractor:
                    latent_pi, latent_vf = self.policy.mlp_extractor(features)
                else:
                    pi_features, vf_features = features
                    latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
                    latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
                # Evaluate the values for the given observations
                values = self.policy.value_net(latent_vf)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                log_prob = distribution.log_prob(actions)
            return values, log_prob
        
        # Determine where the episodes begin
        episodes = np.cumsum(self.rollout_buffer.episode_starts)
        max_episode = int(np.max(episodes))

        # Determine how big the new buffer needs to be.  We will implement HER by creating a new
        # buffer with only HER transitions and concatnating it (below) with the initial samples
        if self.her_method == 'final':
            buffer_size = self.rollout_buffer.buffer_size
        else:
            buffer_size = self.rollout_buffer.buffer_size * int(self.num_her_samples)

        # Initialize the HER Buffer
        new_buffer = DictRolloutBuffer(
            buffer_size,
            self.rollout_buffer.observation_space,
            self.rollout_buffer.action_space,
            device=self.rollout_buffer.device,
            gae_lambda=self.rollout_buffer.gae_lambda,
            gamma=self.rollout_buffer.gamma,
            n_envs=self.rollout_buffer.n_envs,
        )

        # Determine the number of final/random rewards to use
        rewards = self.rollout_buffer.rewards
        episode_ends = np.where(self.rollout_buffer.episode_starts)[0][1:] - 1
        mean_success = (np.mean(rewards[episode_ends]) + 1) / 2
        
        # The desired fraction success D, can be achieved by taking an average of
        # the achived success A and the needed success N:
        # D = (N + A) / 2
        # N = 2D - A
        # This will be clipped to deal with situations where the desired number cannot
        # be achieved
        fraction_success_needed = np.clip(2 * self.success_fraction - mean_success, 0, 1)

        # Fix an off-by-one air that occurs when using only_her_if_move
        if self.only_her_if_move:
            max_index = max_episode
        else:
            max_index = max_episode + 1

        # Main HER implementation
        for episode in range(max_index):

            # Get data for the current episode
            is_episode = (episodes == episode)
            num_steps = np.sum(is_episode)
            if num_steps == 0:
                continue

            actions = self.rollout_buffer.actions[is_episode, :]

            observations = get_dict_idxs(self.rollout_buffer.observations, is_episode)

            # Determine if we should use HER
            if self.only_her_if_move:
                achieved_goal = observations['achieved_goal'].squeeze()
                ag_diff = achieved_goal - achieved_goal[0, :]
                ag_mag = np.linalg.norm(ag_diff, axis=1)
                mag_max = np.max(ag_mag)
                DISTANCE_THRESHOLD = 1E-2
                above_threshold = mag_max > DISTANCE_THRESHOLD
                if not above_threshold:
                    continue
            
            # Determine if we should switch HER methods - typically because we are running HER
            # for a limited time or until we achieve a certain level of success
            her_method = self.her_method
            if self.random_her:
                her_method = np.random.choice(
                    [self.her_failure_method, her_method],
                    p=[1 - self.success_fraction, self.success_fraction]
                )

            if (self.her_method == 'maximum_entropy') or (self.her_method == 'targeted'):
                her_method = np.random.choice(
                    [self.her_failure_method, 'final'],
                    p=[1 - fraction_success_needed, fraction_success_needed]
                )

            # Create the HER transitions
            for step_num in range(num_steps):
                for _ in range(self.num_her_samples):

                    if her_method == 'final':
                        chosen_step = -1
                    elif her_method == 'episode':
                        chosen_step = np.random.choice(num_steps)
                    elif her_method == 'future':
                        chosen_step = np.random.choice(np.arange(step_num, num_steps))

                    if her_method == 'random':
                        goal_obs = dict(self.env.envs[0].env.observation_space.sample())
                    elif her_method == 'targeted':
                        A_SMALL_FACTOR = 1.1
                        
                        num_tries = 100
                        intersects = True
                     
                        critical_distance = self.env.envs[0].env.AGENT_AT_TARGET_DISTANCE
                        distance = critical_distance * A_SMALL_FACTOR
                        while (num_tries > 0) and intersects:
                            direction = np.random.uniform(-1, 1, size=self.num_dims)
                            scaled_change = distance * direction / np.linalg.norm(direction)
                            new_location = get_dict_idxs(observations, np.asarray([-1]))['achieved_goal'] + scaled_change
                            traj_distances = np.linalg.norm(observations['achieved_goal'].squeeze() - new_location.reshape(1, -1), axis=1)
                            
                            num_tries -= 1
                            if np.all(traj_distances > critical_distance):
                                intersects = False

                      
                        goal_obs = get_dict_idxs(observations, np.asarray([-1]))
                        goal_obs['desired_goal'] = new_location.reshape(1, 1, -1)

                    else:
                        goal_obs = get_dict_idxs(observations, np.asarray([chosen_step]))

                    # Alias the step observation and action
                    obs = get_dict_idxs(observations, np.asarray([step_num]))
                    act = actions[step_num, :]
                    
                    # Create the new observation
                    new_obs = deepcopy(obs)
                    new_obs['desired_goal'] = goal_obs['achieved_goal'].reshape([1, 1, -1])
                    np_obs = deepcopy(new_obs)
                    cpu_obs = deepcopy(new_obs)
                    for key, value in new_obs.items():
                        new_obs[key] = th.tensor(value).to(self.device)
                        cpu_obs[key] = th.tensor(value).cpu()

                    # This code isn't used right now.  This creates information that may be necessary
                    # to calculate the rewards using the prescribed env.compute_rewards() function
                    reward_info = {'episode_steps': step_num}
                    if self.reward_info is not None:
                        raise NotImplementedError('The code cannot handle additional reward information right now.')
                        for ri in self.reward_info:
                            reward_info[ri] = self.rollout_buffer.__dict__[ri][is_episode, :][step_num, :]

                    # Calculate value, log_prob, rewards, dones
                    value, log_prob = get_vlp(new_obs, th.tensor(act).to(self.device))
                    try:
                        reward, done = self.env.envs[0].env._compute_reward(
                            np_obs['achieved_goal'],
                            np_obs['desired_goal'],
                            reward_info,
                        )
                    except:
                        reward = self.env.envs[0].env.compute_reward(
                            np.asarray([np_obs['achieved_goal']]),
                            np.asarray([np_obs['desired_goal']]),
                            np.asarray([reward_info]),
                        )
                        max_steps = self.env.envs[0].env._saved_kwargs['max_episode_steps']
                        done = step_num == (max_steps - 1)                        
                    
                    done = done * 1 # Convert to float

                    # Add to the new buffer
                    new_buffer.add(
                        cpu_obs,  # type: ignore[arg-type]
                        th.tensor([act]),
                        th.tensor([reward]),
                        th.tensor([(step_num == 0)]),  # type: ignore[arg-type]
                        th.tensor([value.cpu()]),
                        th.tensor([log_prob.cpu()]),
                    )

        new_buffer.compute_returns_and_advantage(last_values=th.tensor([value.cpu()]), dones=np.asarray([done]))

        # Remove the episodes where the achieved goal did not change
        field_names = ['observations', 'actions', 'rewards', 'advantages', 'returns', 'episode_starts', 'values', 'log_probs']
        if self.only_her_if_move:
            for name in field_names:                    
                if isinstance(new_buffer.__dict__[name], dict):
                    for key, value in new_buffer.__dict__[name].items():
                        new_buffer.__dict__[name][key] = new_buffer.__dict__[name][key][:new_buffer.pos, ...]
                else:
                    new_buffer.__dict__[name] = new_buffer.__dict__[name][:new_buffer.pos, ...]
            new_buffer.buffer_size = new_buffer.pos
        assert new_buffer.pos == new_buffer.buffer_size, 'We do not have all the data!'

        # Concatenate the buffers (original + HER)       
        for name in field_names:
            new_data = new_buffer.__dict__[name]
            
            if isinstance(new_data, dict):
                for key, value in new_data.items():
                    self.rollout_buffer.__dict__[name][key] = np.concatenate(
                        (self.rollout_buffer.__dict__[name][key], new_data[key]),
                        axis=0,
                    )
            else:
                self.rollout_buffer.__dict__[name] = np.concatenate(
                    (self.rollout_buffer.__dict__[name], new_data),
                    axis=0,
                )

        # If it is maximum entropy, adjust the success fraction
        num_to_remove = 0
        remove_to = 0
        if (self.her_method == 'maximum_entropy') or (self.her_method == 'targeted'):

            # Keep track of how successful we were before implementing MEHER
            # This is mostly for debugging
            rewards = self.rollout_buffer.rewards
            num_steps = len(rewards)
            episode_starts = np.where(self.rollout_buffer.episode_starts)[0]
            episode_ends = episode_starts[1:] - 1
            episode_rewards = rewards[episode_ends]
            mean_success = np.sum(episode_rewards >= 0) / len(episode_rewards)
            print('-----------')
            print(mean_success)

            # Determine how many to keep

            # We have to define success as >= 0, because when we create
            # random targets, we can be successful before the end of the reach,
            # which results in a reward of 0
            s = np.sum(episode_rewards >= 0)
            f = np.sum(episode_rewards < 0)
            t = s + f
            d = self.success_fraction
            A_SMALL_NUMBER = 1E-8

            if d != 1.0:
                n = int((d * t + f - t) / (d - 1 + A_SMALL_NUMBER))
                remove_from_success = True
            else:
                n = -1

            if n < 0:
                remove_from_success = False
                n = int((d * t + f - t) / (d))
            num_to_remove = n

            # Determine the episodes that we should remove to reach the MEHER success_faction
            if remove_from_success:
                successful_episodes = np.where(episode_rewards >= 0)[0]
                chosen_episodes = np.random.choice(
                    successful_episodes, num_to_remove, replace=False
                )
            else:
                failure_episodes = np.where(episode_rewards < 0)[0]
                chosen_episodes = np.random.choice(
                    failure_episodes, num_to_remove, replace=False
                )
            drop_starts = episode_starts[chosen_episodes]
            drop_ends = episode_ends[chosen_episodes]

            keep_idx = np.ones(num_steps)
            for st, en in zip(drop_starts, drop_ends):
                keep_idx[np.arange(st, en + 1)] = 0
            remove_to = np.sum(keep_idx == 0)
            
            keep_idx = np.where(keep_idx)[0]

            # Remove the necessary episodes
            for name in field_names:                    
                if isinstance(self.rollout_buffer.__dict__[name], dict):
                    for key, value in self.rollout_buffer.__dict__[name].items():
                        self.rollout_buffer.__dict__[name][key] = self.rollout_buffer.__dict__[name][key][keep_idx, ...]
                else:
                    self.rollout_buffer.__dict__[name] = self.rollout_buffer.__dict__[name][keep_idx, ...]

        # Update buffer parameters
        self.rollout_buffer.buffer_size += (new_buffer.buffer_size - remove_to)
        self.rollout_buffer.pos = self.rollout_buffer.buffer_size
        self.rollout_buffer.full = True
