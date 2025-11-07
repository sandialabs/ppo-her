from tbparse import SummaryReader
from os.path import join, abspath
import numpy as np
from abc import ABC
import pandas as pd
from copy import deepcopy
from itertools import product
import os
import json
from warnings import warn
from flatten_dict import flatten

from ppo_her.base.utils import get_folder

class BaseProcessor(ABC):

    '''
        Abstract base class for data processors
        Inputs:
            folder_name (str) - the parent folder for the data
            max_step (int) - trials with less than max_step steps will not be included
            experiment_name (str) - the folder within folder_name that contains
                experiment-specific results
            use_time_axis (bool) - if True, plot the data on a time axis in addition
                to the default step axis
            fields_to_ignore (list) - columns to ignore when making groupings
    '''

    def __init__(
            self,
            folder_name,
            max_step,
            experiment_name,
            use_time_axis,
            fields_to_ignore=None,
        ):

        self.folder_name = abspath(folder_name)
        self.max_step = max_step
        self.experiment_name = experiment_name
        self.use_time_axis = use_time_axis

        if fields_to_ignore is None:
            self.fields_to_ignore = []
        else:
            self.fields_to_ignore = fields_to_ignore

        self.define_step_labels()
        self.group_by()
        self.load_data()
        self.process_data()
        self.summarize_data(mode='step')

        if self.use_time_axis:
            self.define_time_labels()
            self.summarize_data(mode='time')      

    def load_data(self):

        # Initialize variables
        self.extra_columns = {'dir_name', 'wall_time'}
        if not isinstance(self.experiment_name, list):
            self.experiment_name = [self.experiment_name]
        self.num_folders = len(self.experiment_name)
        self.sr = []
        self.experiment_dicts = []
        

        for experiment_name in self.experiment_name:
            experiment_dicts = {}

            # Load the tensorboard data for each experiment
            results_dir = join(self.folder_name, experiment_name, 'sb3_results')
            print('Loading: ' + str(results_dir))
            self.sr.append(SummaryReader(results_dir, extra_columns=self.extra_columns))

            # Load the experiment configs
            tensorboard_directory = join(self.folder_name, experiment_name)
            ray_folders = [entry.name for entry in os.scandir(tensorboard_directory) if entry.is_dir()]
            for folder in ray_folders:

                if 'train' not in folder:
                    continue

                key = folder.split('_')[1:3]
                key = '_'.join(key)

                config_name = join(tensorboard_directory, folder, 'params.json')
                with open(config_name, 'r') as file:
                    config = json.load(file)
                flat_config = flatten(config, reducer='dot')
                new_dict = {}
                for key2, value2 in flat_config.items():
                    try:
                        new_key = key2.split('.')[1]
                    except:
                        new_key = key2
                    new_dict[new_key] = value2
                experiment_dicts[key] = new_dict
        self.experiment_dicts.append(experiment_dicts)
        
        print('All data loaded!')

    def process_data(self):
        
        print('Processing')

        # Combine the configs with the tensorboard data
        def get_experiment_name(df):
            return df.split('/')[1]

        def match_config(df, config):
            return pd.DataFrame(config[df], index=[0])

        srs = []
        for sr, config in zip(self.sr, self.experiment_dicts):
            data = sr.scalars
            data = data.loc[data.tag == 'rollout/ep_rew_mean', :].reset_index(drop=True)
            data.loc[:, 'experiment_name'] = data.loc[:, 'dir_name'].apply(get_experiment_name)

            new_data = data.loc[:, 'experiment_name'].apply(match_config, args=(config,))
            new_data = pd.concat(new_data.tolist()).reset_index(drop=True)
            data = pd.concat([data, new_data], axis=1)
            srs.append(data)
        data = pd.concat(srs)
       
        # Do not use partial data
        completed_data = np.unique(data.loc[data.step >= self.max_step, 'dir_name'])
        self.data = data.loc[data.dir_name.isin(completed_data), :]

        # Remove columns that do not change
        no_change = []
        for column in self.data.columns:
            unique_vals = pd.unique(self.data[column])
            if len(unique_vals) == 1:
                no_change.append(column)
        self.data = self.data.drop(no_change, axis=1)


    def summarize_data(self, mode):

        print('Summarizing')

        # Initialize for summarizing data
        all_summary_data = []

        # Get the number of sub-plots required
        if self.row_type in self.data.columns:
            unique_rows = np.unique(self.data[self.row_type])
            num_rows = len(unique_rows)
        else:
            unique_rows = np.asarray(['None'])
            num_rows = 1

        if self.column_type in self.data.columns:
            unique_columns = np.unique(self.data[self.column_type])
            num_columns = len(unique_columns)
        else:
            unique_columns = np.asarray(['None'])
            num_columns = 1

        # Figure out which columns are non-standard
        non_standard_columns = []
        for col in self.data.columns:
            if col in self.standard_columns:
                continue
            if col == self.row_type:
                continue
            if col == self.column_type:
                continue
            # Don't do anything with the experiment number
            if 'exper' in col:
                continue
            if 'dir_name' in col:
                continue
            if 'OnPolicyAlgorithm' in col:
                continue
            if col in self.fields_to_ignore:
                continue
            non_standard_columns.append(col)

        # Find all possible combinations of non-standard columns
        unique_values = []
        for nsc in non_standard_columns:
            unique_values.append(pd.unique(self.data[nsc]))
        possible_values = list(product(*unique_values))
        num_possible_values = len(possible_values)

        # Summarize each instance
        for col_num, col in enumerate(unique_columns):
            for row_num, row in enumerate(unique_rows):

                # Get the data that corresponds to the columns and the rows
                x_max = 0
                
                sub_data = deepcopy(self.data)
                if self.column_type in self.data.columns:
                    sub_data = sub_data.loc[sub_data[self.column_type] == col, :]
                if self.row_type in self.data.columns:
                    sub_data = sub_data.loc[sub_data[self.row_type] == row, :]
                

                # Extract out data for a single line
                for pv in possible_values:
                    sub_sub_data = sub_data.copy()
                    
                    # Extract out the part of the sub_data the corresponds to a single condition (line)
                    # and create the legend entry
                    current_summary_data = {}
                    for col_title, val in zip(non_standard_columns, pv):
                        sub_sub_data = sub_sub_data.loc[
                            (sub_sub_data[col_title] == val), :
                        ]
                        current_summary_data[col_title] = val
                    current_summary_data[self.column_type] = col
                    current_summary_data[self.row_type] = row

                    # Ensure that some data exists
                    if len(sub_sub_data) == 0:
                        warn('No data found for: ' + str(non_standard_columns) + ' = ' + str(pv))
                    
                    # Get the right units for the x-axis
                    if mode == 'step':
                        sub_sub_data = self.change_x_axis_step(sub_sub_data)
                    else:
                        sub_sub_data = self.change_x_axis_time(sub_sub_data)

                    # Remove extraneous columns
                    sub_sub_data = sub_sub_data.loc[:, ['value', self.x_data]]

                    # Calculate the statistics
                    summary_data = sub_sub_data.groupby(self.x_data).quantile([0.25, 0.5, 0.75])
                    summary_data = summary_data.reset_index()

                    low = summary_data.loc[summary_data.level_1 == 0.25, [self.x_data, 'value']]
                    high = summary_data.loc[summary_data.level_1 == 0.75, [self.x_data, 'value']]
                    med = summary_data.loc[summary_data.level_1 == 0.50, [self.x_data, 'value']]

                    # Collect summary data
                    current_summary_data['num_reps'] = sub_sub_data.groupby(self.x_data).count().values.squeeze()
                    current_summary_data['low'] = low.loc[:, 'value'].values
                    current_summary_data['high'] = high.loc[:, 'value'].values
                    current_summary_data['med'] = med.loc[:, 'value'].values
                    current_summary_data[self.x_data] = med[self.x_data].values

                    all_summary_data.append(current_summary_data)

        # Make directories if they do not exist
        json_dir = get_folder('summary_data')
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)
        experiment_dir = os.path.join(json_dir, self.experiment_name[0])
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        
        # Save the summary data
        data_name = join(experiment_dir, mode + '.json')
        print('Saving data to: ' + data_name)
        all_summary_data = pd.DataFrame(all_summary_data)
        all_summary_data.to_json(data_name)

