from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

current_exp = get_experiment_name(__file__)
other_exp = 'ppo-her'

EXPERIMENT_NAME = [current_exp, other_exp]
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|use_h = True|learn = 0.003': 'PPO-HER, LR=3E-3',
    '|use_h = True|learn = 0.0003': 'PPO-HER, LR=3E-4',
    '|use_h = True|learn = 3e-05': 'PPO-HER, LR=3E-5',
    '|use_h = False|learn = 0.003': 'PPO, LR=3E-3',
    '|use_h = False|learn = 0.0003': 'PPO, LR=3E-4',
    '|use_h = False|learn = 3e-05': 'PPO, LR=3E-5',
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
