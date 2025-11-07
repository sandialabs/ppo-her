from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

current_exp = get_experiment_name(__file__)

EXPERIMENT_NAME = [current_exp]
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    #100, 10, 20, 50
    '|GRID_ = 10|use_h = True' : '10 | PPO-HER',
    '|GRID_ = 20|use_h = True' : '20 | PPO-HER',
    '|GRID_ = 30|use_h = True' : '30 | PPO-HER',
    '|GRID_ = 40|use_h = True' : '40 | PPO-HER',
    '|GRID_ = 50|use_h = True' : '50 | PPO-HER',
    '|GRID_ = 10|use_h = False' : '10 | PPO',
    '|GRID_ = 20|use_h = False' : '20 | PPO',
    '|GRID_ = 30|use_h = False' : '30 | PPO',
    '|GRID_ = 40|use_h = False' : '40 | PPO',
    '|GRID_ = 50|use_h = False' : '50 | PPO',
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
        large_bottom=True,
    )
