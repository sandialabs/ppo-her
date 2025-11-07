from stable_baselines3.sac import SAC
import numpy as np


class CustomSAC(SAC):
    '''
    Version of StableBaselines3 SAC that allows for early stopping.

    Parameters:
    Parameters that are unchanged from StableBaselines are not listed.
    policy_setup (dict): parameters for setting up HER, containing the following fields:
        'stop_her_mode' (str) - The method that we use to determine when we should turn HER off.  Set stop_her to
            infinity to disable
                'reward' - stop when we reach a reward threshold
                'steps' - stop when we reach a certain number of steps
        'stop_her' (float) - the threshold at which we stop using HER.  The threshold is applied to the mean
            reward or the number of steps, depending on the value of stop_her_mode

    '''

    def __init__(
        self,
        policy,
        env,
        learning_rate,
        buffer_size=1_000_000,  # 1e6
        learning_starts= 100,
        batch_size= 256,
        tau= 0.005,
        gamma= 0.99,
        train_freq= (1, "step"),
        gradient_steps= 1,
        action_noise = None,
        replay_buffer_class = None,
        replay_buffer_kwargs= None,
        optimize_memory_usage = False,
        policy_kwargs= None,
        stats_window_size= 100,
        tensorboard_log= None,
        verbose= 0,
        device = "auto",
        seed= None,
        use_sde= False,
        sde_sample_freq = -1,
        use_sde_at_warmup= False,
        policy_setup=None,
    ):
        
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size=buffer_size,  # 1e6
            learning_starts= learning_starts,
            batch_size= batch_size,
            tau= tau,
            gamma= gamma,
            train_freq= train_freq,
            gradient_steps= gradient_steps,
            action_noise = action_noise,
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs= replay_buffer_kwargs,
            optimize_memory_usage = optimize_memory_usage,
            policy_kwargs= policy_kwargs,
            stats_window_size= stats_window_size,
            tensorboard_log= tensorboard_log,
            verbose= verbose,
            device = device,
            seed= seed,
            use_sde= use_sde,
            sde_sample_freq = sde_sample_freq,
            use_sde_at_warmup= use_sde_at_warmup,
        )

        if policy_setup is not None:
            for key, value in policy_setup.items():
                self.__dict__[key] = value

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=4,
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        self.reward_timestep = None


        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

                # Check if we can terminate b/c we were successful
                WINDOW_SIZE = 100
                non_zero_rewards = self.replay_buffer.rewards[self.replay_buffer.rewards != 0]
                if len(non_zero_rewards) > WINDOW_SIZE:
                    non_zero_rewards = non_zero_rewards[-WINDOW_SIZE:]
                    mean_return = np.mean(non_zero_rewards)

                    # Quit once we get to 1.3 * num_timesteps after we reach 100% success
                    if mean_return >= 1:
                        if self.reward_timestep is None:
                            self.reward_timestep = self.num_timesteps

                    if self.reward_timestep is not None:
                        if self.num_timesteps > (1.3 * self.reward_timestep):
                            break

                    # Determine if we should turn HER off
                    if self.stop_her_mode == 'reward':
                        if mean_return >= self.stop_her:
                            self.replay_buffer.her_ratio = 0
                            self.replay_buffer.n_sampled_goal = 0


        callback.on_training_end()

        return self