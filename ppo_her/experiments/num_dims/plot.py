from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

EXPERIMENT_NAME = get_experiment_name(__file__)
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|NUM_D = 2': '2',
    '|NUM_D = 3': '3',
    '|NUM_D = 4': '4',
    '|NUM_D = 5': '5',
    '|NUM_D = 6': '6',
}
COLUMN_TYPE = 'prey_velocity'
ROW_TYPE = 'spawn_policy'
LEGEND_TITLE = 'Number of Physical Dimensions'

if __name__ == '__main__':
    p = PlotOnly(
        experiment_name=EXPERIMENT_NAME,
        column_type=COLUMN_TYPE,
        row_type=ROW_TYPE,
        use_time_axis=USE_TIME_AXIS,
        legend_mapping=LEGEND_MAPPING,
        legend_title=LEGEND_TITLE,
        large_bottom=True,
    )
