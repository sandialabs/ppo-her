from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

EXPERIMENT_NAME = [get_experiment_name(__file__), 'ppo-her']
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|use_h = False|type = PPO': 'PPO',
    '|use_h = True|type = PPO': 'PPO-HER',
    '|use_h = False|type = SAC': 'SAC',
    '|use_h = True|type = SAC': 'SAC-HER',
}
COLUMN_TYPE = 'prey_velocity'
ROW_TYPE = 'spawn_policy'

if __name__ == '__main__':
    p = PlotOnly(
        experiment_name=EXPERIMENT_NAME,
        column_type=COLUMN_TYPE,
        row_type=ROW_TYPE,
        use_time_axis=USE_TIME_AXIS,
        legend_mapping=LEGEND_MAPPING,
        experiment_values = [{'type': 'SAC'}, {'type': 'PPO'}],
    )
