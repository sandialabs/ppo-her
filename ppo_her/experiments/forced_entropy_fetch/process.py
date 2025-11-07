from ppo_her.processing.base import BaseProcessor
from ppo_her.processing.base_x_formatter import XFormatter
from ppo_her.base.utils import get_experiment_name, get_folder
import pandas as pd
import numpy as np
from copy import deepcopy

FOLDER_NAME = get_folder('ray_results')
MAX_STEP = int(0)
EXPERIMENT_NAME = get_experiment_name(__file__)
USE_TIME_AXIS = True


class CustomXFormatter(XFormatter):

    def group_by(self):
        self.standard_columns = ['step', 'tag', 'value', 'wall_time']
        self.extra_columns = {'dir_name', 'wall_time', 'ep_success', 'success_rate'}

        self.row_type = 'None'
        self.column_type = 'env_type'



class Processor(BaseProcessor, CustomXFormatter):

    def process_data(self):
        
        print('Processing')

        # Only save the episode returns
        data = self.sr[0].scalars
        if self.num_folders > 1:
            for sr in self.sr[1:]:
                data = pd.concat((data, sr.scalars), axis=0)

        data = data.loc[np.logical_or(data.tag == 'rollout/ep_success', data.tag == 'rollout/success_rate'), :]
       
        # Do not use partial data
        completed_data = np.unique(data.loc[data.step >= self.max_step, 'dir_name'])
        self.data = data.loc[data.dir_name.isin(completed_data), :]

        # Create a function to seperate the value from the parameter name
        def get_condition_value(cell):
            return cell[-1]

        def get_first_element(cell):
            return cell[0]

        # Parse the condition string into seperate variables
        conditions = self.data.dir_name
        split_conditions = conditions.str.split(pat='|', n=-1, expand=True)
        sc_data = {}
        new_names = {}
        columns_to_remove = []
        last_column = 0
        for column in split_conditions:
            try:
                sc = split_conditions[column].str.split(pat='/', n=-1, expand=True)[0]
                sc = sc.str.split(pat='-', n=-1, expand=True)

                unique_titles = pd.unique(sc[0])
                for ut in unique_titles:

                    data = sc.iloc[:, 1]

                    temp_data = deepcopy(data)
                    temp_data[sc[0] != ut] = 'None'

                    if ut not in list(sc_data.keys()):
                        sc_data[ut] = temp_data
                    else:
                        # We haven't hit this condition yet, but it should be caused by an offset in the data
                        nones = np.where(sc_data[ut] == 'None')[0]
                        sc_data[ut].iloc[nones] = temp_data.iloc[nones]
            except:
                columns_to_remove.append(column)
        split_conditions = pd.DataFrame(sc_data)

        # Remove variables that are all the same
        no_change = []
        for column in split_conditions:
            unique_vals = pd.unique(split_conditions[column])
            if len(unique_vals) == 1:
                no_change.append(column)
        split_conditions = split_conditions.drop(no_change, axis=1)

        # Concatenate the columns
        self.data = pd.concat((self.data, split_conditions), axis=1)

if __name__ == '__main__':
    p = Processor(
        folder_name=FOLDER_NAME,
        max_step=MAX_STEP,
        experiment_name=EXPERIMENT_NAME,
        use_time_axis=USE_TIME_AXIS,
    )
