from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

EXPERIMENT_NAME = get_experiment_name(__file__)
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|succe = 0.0': '0.0',
    '|succe = 0.1': '0.1',
    '|succe = 0.2': '0.2',
    '|succe = 0.3': '0.3',
    '|succe = 0.4': '0.4',
    '|succe = 0.5': '0.5',
    '|succe = 0.6': '0.6',
    '|succe = 0.7': '0.7',
    '|succe = 0.8': '0.8',
    '|succe = 0.9': '0.9',
    '|succe = 1.0': '1.0',
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
        legend_title='% Success',
        large_bottom=True,
    )
