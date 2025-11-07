from ppo_her.processing.base import BaseProcessor
from ppo_her.processing.base_x_formatter import XFormatter
from ppo_her.base.utils import get_experiment_name, get_folder

FOLDER_NAME = get_folder('ray_results') 
MAX_STEP = int(0)
EXPERIMENT_NAME = get_experiment_name(__file__)
USE_TIME_AXIS = True
FIELDS_TO_IGNORE = ['MAX_E']

class Processor(BaseProcessor, XFormatter):
    pass

if __name__ == '__main__':
    p = Processor(
        folder_name=FOLDER_NAME,
        max_step=MAX_STEP,
        experiment_name=EXPERIMENT_NAME,
        use_time_axis=USE_TIME_AXIS,
        fields_to_ignore=FIELDS_TO_IGNORE,
    )
