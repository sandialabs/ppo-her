from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join, abspath
import os

EXPERIMENT_NAME = [get_experiment_name(__file__)]
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|type = None|use_her = False': 'PPO',
    '|type = None|use_her = None': 'PPO-Her',
    # '|type = sac|use_her = False': 'SAC',
    # '|type = sac|use_her = None': 'SAC-HER',
}

# LEGEND_MAPPING = None
COLUMN_TYPE = 'env_type'
ROW_TYPE = 'None'
DEFAULT_KEYS = {}

class CustomPlotter(PlotOnly):

    def __init__(
            self,
            experiment_name,
            use_time_axis,
            column_type,
            row_type,
            legend_mapping=None,
            legend_title=None,
            large_bottom=False,
            default_values=None,
            experiment_values=None,
            columns_to_exclude=None,
            save_dir = None,
        ):

        if not isinstance(experiment_name, list):
            experiment_name = [experiment_name]
        self.experiment_name = experiment_name

        self.use_time_axis = use_time_axis
        self.row_type = row_type
        self.column_type = column_type
        self.legend_mapping = legend_mapping
        self.legend_title = legend_title
        self.large_bottom = large_bottom
        self.columns_to_exclude = columns_to_exclude

        self.default_values = default_values
        self.experiment_values = experiment_values
        self.save_dir = save_dir

        # Redefine standard columsn to include summary variables 
        self.standard_columns = ['step', 'tag', 'value', 'wall_time', 'low', 'high', 'med', 'num_reps']
        self.HER_NAME = 'use_h'
        self.PPO_NAME = 'type'

        self.y_label = 'Fraction Success'
        self.y_lim = [-0.1, 1.1]
        self.fig_size = (11, 3)

        # Setup variables
        x_fxn = {
            'step': [self.define_step_labels],
            'time': [self.define_time_labels],
        }
        json_dir = get_folder('summary_data')
        for mode in ['step', 'time']:
            for fxn in x_fxn[mode]:
                fxn()

            file_name = []
            for exp_name in experiment_name:
                file_name.append(os.path.join(json_dir, exp_name, mode + '.json'))
            self.load_data(file_name=file_name)
            self.plot_data(mode=mode)



if __name__ == '__main__':
    p = CustomPlotter(
        experiment_name=EXPERIMENT_NAME,
        column_type=COLUMN_TYPE,
        row_type=ROW_TYPE,
        use_time_axis=USE_TIME_AXIS,
        legend_mapping=LEGEND_MAPPING,
        default_values={'type': 'ppo', 'use_her': True},
    )
