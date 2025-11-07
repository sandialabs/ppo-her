from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder

EXPERIMENT_NAME = [get_experiment_name(__file__)]
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|use_h = False|type = PPO': 'PPO',
    '|use_h = True|type = PPO': 'PPO-HER',
}
COLUMN_TYPE = 'prey_velocity'
ROW_TYPE = 'spawn_policy'
DEFAULT_KEYS = {}

if __name__ == '__main__':
    p = PlotOnly(
        experiment_name=EXPERIMENT_NAME,
        column_type=COLUMN_TYPE,
        row_type=ROW_TYPE,
        use_time_axis=USE_TIME_AXIS,
        legend_mapping=LEGEND_MAPPING,
        default_values=None,
        experiment_values = [{'type': 'PPO'}, {'type': 'SAC'}],
    )
