from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

EXPERIMENT_NAME = get_experiment_name(__file__)
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|her_m = episode|num_h = 1': 'Episode, 1',
    '|her_m = episode|num_h = 2': 'Episode, 2',
    '|her_m = episode|num_h = 4': 'Episode, 4',
    '|her_m = episode|num_h = 8': 'Episode, 8',
    '|her_m = future|num_h = 1': 'Future, 1',
    '|her_m = future|num_h = 2': 'Future, 2',
    '|her_m = future|num_h = 4': 'Future, 4',
    '|her_m = future|num_h = 8': 'Future, 8',
    '|her_m = final|num_h = 1': 'Final',
    '|her_m = final|num_h = 2': None,
    '|her_m = final|num_h = 4': None,
    '|her_m = final|num_h = 8': None,
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
    )
