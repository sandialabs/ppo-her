# from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray import air
from ppo_her.base.utils import get_name, get_folder

def run(train, num_samples=2, config=None, num_concurrent=None, num_nodes=None, experiment_name=None):
    '''
        Run a ray.tune parallel experiment

        Parameters:
            train (function) - the function to run in parallel
            num_samples (int) - the number of repetitions for each experiment to run
                We will typically want this to be 1.  We increase the number of experiments
                by chaning the experiment_number parameter in the config so that each experiment
                is logged seperately
            config (dict) - the config.  See config.py for docs
            num_concurrent (int) - max number of experiments to run in parallel
            num_nodes (int) - the number of slurm nodes to run on
            experiment_name (str) - the name of the experiment - for logging
    '''

    # Input handling
    if config is None:
        config = {}

    if num_nodes is None:
        num_nodes = 1

    if num_concurrent is None:
        num_concurrent = num_samples
    
    # Create the ray "trainable"
    # trainable_with_gpu = tune.with_resources(train, {"gpu": (num_nodes/num_concurrent)})
    trainable_with_cpu = tune.with_resources(train, {"cpu": 1})
    if experiment_name is None:
        experiment_name = get_name(config)
    else:
        experiment_name = str(experiment_name)
    
    local_dir = get_folder('ray_results')

    # Run the experiments in parallel
    tuner = tune.Tuner(
        #trainable_with_gpu,
        trainable_with_cpu,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric='episode_reward_mean',
            mode='max',
        ),
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir=local_dir,
        ),
    )

    # Log the results
    print('Saving results to: ' + local_dir +  ' | ' + experiment_name)
    results = tuner.fit()
