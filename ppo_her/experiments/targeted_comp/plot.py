from ppo_her.processing.plot_only import PlotOnly
from ppo_her.base.utils import get_experiment_name, get_folder
from os.path import join

EXPERIMENT_NAME = ['targeted', 'maximum_entropy', 'stop_her_reward', 'ppo-her']
experiment_folder = get_folder('experiments')
USE_TIME_AXIS = False
LEGEND_MAPPING = {
    '|succe = 0.5|meher = False|use_h = False': None, #'PPO',
    '|succe = 0.5|meher = False|use_h = True': 'PPO-HER',

    '|succe = 0.0|meher = me|use_h = True': None,
    '|succe = 0.1|meher = me|use_h = True': None,
    '|succe = 0.2|meher = me|use_h = True': None,
    '|succe = 0.3|meher = me|use_h = True': None,
    '|succe = 0.4|meher = me|use_h = True': None,
    '|succe = 0.5|meher = me|use_h = True': None,
    '|succe = 0.6|meher = me|use_h = True': 'MEHER 0.6',
    '|succe = 0.7|meher = me|use_h = True': None,
    '|succe = 0.8|meher = me|use_h = True': None,
    '|succe = 0.9|meher = me|use_h = True': None,
    '|succe = 1.0|meher = me|use_h = True': None,

    '|succe = 0.0|meher = Targeted|use_h = True': None,
    '|succe = 0.1|meher = Targeted|use_h = True': None,
    '|succe = 0.2|meher = Targeted|use_h = True': None,
    '|succe = 0.3|meher = Targeted|use_h = True': None,
    '|succe = 0.4|meher = Targeted|use_h = True': None,
    '|succe = 0.5|meher = Targeted|use_h = True': None,
    '|succe = 0.6|meher = Targeted|use_h = True': 'Targeted MEHER 0.6',
    '|succe = 0.7|meher = Targeted|use_h = True': None,
    '|succe = 0.8|meher = Targeted|use_h = True': None,
    '|succe = 0.9|meher = Targeted|use_h = True': None,
    '|succe = 1.0|meher = Targeted|use_h = True': None,
    
    '|succe = 0.5|meher = True|use_h = True': 'PPO-HER-2-PPO',
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
        experiment_values = [
            {'meher': 'Targeted'},
            {'meher': 'me'},
            {'meher': 'True'},
            {'meher': 'False'}],
        save_dir=get_experiment_name(__file__),
    )
