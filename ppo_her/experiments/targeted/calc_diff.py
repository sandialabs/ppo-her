from ppo_her.base.utils import get_folder, get_experiment_name
from os.path import join

from ppo_her.processing.calc_diff import Compare

class CustomCompare(Compare):

    def define_constants(self):
        self.null_mapping = {
            'use_h': True,
        }
        self.rename_dict = {
            'spawn_policy': 'Spawn Method',
            'prey_velocity': 'Prey Policy',
        }
        self.names = ['time_to_learn_steps', 'time_to_learn_time', 'max_median_reward', 'iqr']
        self.desirable = ['low', 'low', 'high', 'low']
        self.division_factor = [1000, 60, 1, 1]

        self.condition_columns = ['success_fraction']

    def rename_columns(self, table):
        return table


if __name__ == '__main__':
    folder = get_folder('stats')
    experiment_name = get_experiment_name(__file__)
    stat_path = join(folder, experiment_name)

    c = CustomCompare(stat_path, experiment_name)







