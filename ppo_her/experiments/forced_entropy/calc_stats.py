from ppo_her.processing.calc_stats import StatCalc
from ppo_her.base.utils import get_folder, get_experiment_name
from os.path import join

if __name__ == '__main__':
    experiment_name = get_experiment_name(__file__)

    x_mapping = {
        'step': 'step',
        'time': 'wall_time',
    }

    extra_columns = [{'Type': 'PPO'}, {'Type': 'SAC'}]

    for mode in ['step', 'time']:
        file_location = [
            join(get_folder('summary_data'), experiment_name, mode + '.json'),
        ]
        x_axis = x_mapping[mode]
        
        sc = StatCalc(file_location, x_axis, experiment_name, mode, extra_columns=extra_columns)






