from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ppo_her.processing.base_x_formatter import XFormatter
from ppo_her.base.utils import get_folder
from ppo_her.base.config import DEFAULT_CONFIG
from flatdict import FlatDict
from warnings import warn
from ppo_her.processing.colors import get_colors
from copy import deepcopy

class PlotOnly(XFormatter):

    '''
        Plot data that has already been processed by a processing.base.BaseProcessor
        Inputs:
            experiment_name (str) - the folder within folder_name that contains
                experiment-specific results
            use_time_axis (bool) - if True, plot the data on a time axis in addition
                to the default step axis
            column_type (str) - the dataframe column to make the plot column
            row_type (str) - the dataframe column to make the plot row
            legend_mapping (dict) - mapping between the legend text and the text that should be displayed
            legend_title (str) - the title for the legend
            large_bottom (bool) - add space for multi-row legends
            default_values (dict) - the values to use when defaults cannot be found.
                Keys = column names.  Values = default values
            experiment_values (list of dicts) - experiment-specific columns that we need to give
                to properly seperate values
            columns_to_exclude (list of strings) - columns that should not be considered
            save_dir (str) - folder to save to - overwrites default
    '''

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

        # Input handling
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

        # Define plot limits and labels
        self.y_label = 'Episode Return'
        self.y_lim = [-1.1, 1.1]
        self.fig_size = (11, 6)

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

    def load_data(self, file_name):
        print('Loading: ' + str(file_name))
        data = []
        for num, fn in enumerate(file_name):
            new_data = pd.read_json(fn)
            if self.experiment_values is not None:
                for key, value in self.experiment_values[num].items():
                    new_data.loc[:, key] = value
            data.append(new_data)
        self.data = pd.concat(data)

        # Find and replace nans with default values
        if self.data.isnull().values.any():
            null_cols = self.data.isnull().any(axis=0)
            null_cols = np.asarray(null_cols[null_cols.values].index)

            config = FlatDict(deepcopy(DEFAULT_CONFIG))

            keys = np.asarray(config.keys())
            keys2 = np.asarray([key.split(':')[-1] for key in keys])
            if self.default_values is not None:
                keys3 = list(self.default_values.keys())
                keys4 = np.asarray([key.split(':')[-1] for key in keys3])
            for nc in null_cols:
                
                num_chars = len(nc)
                if self.default_values is not None:
                    default_keys = np.asarray([key[:num_chars] for key in keys4])
                if (self.default_values is not None) and (nc in default_keys.tolist()):
                    temp_keys = default_keys
                else:
                    temp_keys = np.asarray([key[:num_chars] for key in keys2])
                    warn('Cannot find all values of: ' + str(nc) + '. Replacing some with default values')

                # Determine which key to use
                key_location = np.where(temp_keys == nc)[0]
                if len(key_location) > 1:
                    warn('Multiple keys found.  Using first')
                elif len(key_location) == 0:
                    warn('Key not found in default configs: ' + str(nc))
                    print('Removing Column: ' + str(nc))
                    self.data = self.data.drop(nc, axis=1)
                else:
                    key_location = key_location[0]
                    default_key = keys[key_location]
                    default_value = config[default_key]

                    self.data.loc[self.data[nc].isnull(), nc] = default_value
        
    
    def plot_data(self, mode):

        # Get the number of sub-plots required
        if self.row_type != 'None':
            unique_rows = np.unique(self.data[self.row_type])
            num_rows = len(unique_rows)
        else:
            unique_rows = ['None']
            num_rows = 1
        
        if self.column_type != 'None':
            unique_columns = np.unique(self.data[self.column_type])
            num_columns = len(unique_columns)
        else:
            unique_columns = ['None']
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
            if 'exper_' in col:
                continue
            if 'dir_name' in col:
                continue
            if 'OnPolicyAlgorithm' in col:
                continue
            if self.columns_to_exclude is not None:
                if col in self.columns_to_exclude:
                    continue
            non_standard_columns.append(col)

        # Find all existing combinations of non-standard columns
        unique_data = self.data.loc[:, non_standard_columns]
        values = unique_data.values
        tuple_values = [tuple(val) for val in values]
        possible_values = []
        for tv in tuple_values:
            if tv not in possible_values:
                possible_values.append(tv)
        num_possible_values = len(possible_values)
        
        colors = get_colors(num_possible_values)
        color_mapping = {}
        if self.legend_mapping is not None:
            current_num = 0
            for value in self.legend_mapping.values():
                if value is not None:
                    color_mapping[value] = colors[current_num]
                    current_num += 1

        current_color = 0
        pv_to_label = {}

        # Create the figure
        fig, ax = plt.subplots(num_rows, num_columns, figsize=self.fig_size)
        if ax.ndim == 1:
            if num_rows == 1:
                ax = np.expand_dims(ax, 0)
            else:
                ax = np.expand_dims(ax, -1)
        
        # Plot each instance
        for col_num, col in enumerate(unique_columns):
            for row_num, row in enumerate(unique_rows):

                ax_max = 0

                # Get the data that corresponds to the columns and the rows
                if (num_rows > 1) or (num_columns > 1):
                    current_axis = ax[row_num, col_num]
                else:
                    current_axis = ax
                x_max = 0
                sub_data = self.data.loc[
                        (self.data[self.row_type] == row)
                        & (self.data[self.column_type] == col), :
                    ]
                

                # Extract out data for a single line
                for pv in possible_values:
                    full_label = ''
                    label = None
                    sub_sub_data = sub_data.copy()
                    
                    # Extract out the part of the sub_data the corresponds to a single condition (line)
                    # and create the legend entry
                    for col_title, val in zip(non_standard_columns, pv):
                        sub_sub_data = sub_sub_data.loc[
                            (sub_sub_data[col_title] == val), :
                        ]
                        full_label += ('|' + str(col_title) + ' = ' + str(val))
                        if (row_num < len(unique_rows) - 1) or (col_num < len(unique_columns) - 1):
                            label = '_nolabel_'
                            pv_to_label[pv] = full_label
                        else:
                            label = full_label                       
                                                
                    # Change legends based on mapping
                    if (self.legend_mapping is not None):
                        if (label in list(self.legend_mapping.keys())):
                            label = self.legend_mapping[label]

                        if (full_label in list(self.legend_mapping.keys())) and (self.legend_mapping[full_label] is None):
                            continue
                    
                    # Ensure that some data exists
                    if len(sub_sub_data) == 0:
                        continue

                    if len(sub_sub_data.loc[:, 'low'].values[0]) == 0:
                        continue

                    # Setup the colors
                    if (pv not in list(color_mapping.keys())) and (full_label not in list(color_mapping.keys())):
                        color_mapping[pv] = colors[current_color]
                        current_color += 1

                    low = np.asarray(sub_sub_data.loc[:, 'low'].values[0])
                    high = np.asarray(sub_sub_data.loc[:, 'high'].values[0])
                    med = np.asarray(sub_sub_data.loc[:, 'med'].values[0])

                    if mode == 'time':
                        x_axis_name = 'wall_time'
                    else:
                        x_axis_name = mode
                    x_axis = np.asarray(sub_sub_data.loc[:, x_axis_name].values[0])

                    # Determine how to scale the x-axis
                    is_at_max = np.where(med == np.max(med))[0][0]
                    if is_at_max > 0:
                        ax_max = np.max([ax_max, x_axis[is_at_max]])
                    else:
                        ax_max = np.max([ax_max, x_axis[-1]])

                    # Don't plot redundant data
                    is_at_max_1 = np.where(med == 1)[0]
                    if len(is_at_max_1) > 0:
                        is_at_max_1 = int(np.ceil(is_at_max_1[0] * 1.2))
                        if is_at_max_1 > len(med):
                            is_at_max_1 = -1
                    else:
                        is_at_max_1 = -1

                    # Determine the color to use
                    try:
                        color = color_mapping[self.legend_mapping[pv_to_label[pv]]]
                    except:
                        color = color_mapping[pv]

                    current_axis.fill_between(self.x_scale * x_axis[:is_at_max_1], low[:is_at_max_1], high[:is_at_max_1], color=color, alpha=0.3, label='_nolabel_')
                    current_axis.plot(self.x_scale *  x_axis[:is_at_max_1], med[:is_at_max_1], color=color, label=label)

                col_mapping = {
                    'attract': 'Attract',
                    'away': 'Straight\nAway',
                    'directio': 'Random\nDirection',
                    'random': 'Random',
                    'repel': 'Repel',
                    'static': 'Static',
                }
                row_mapping = {
                    'apart': 'Spawn Apart',
                    'random': 'Spawn Random'
                }

                # Only give axis labels to boundary plots
                if col_num == 0:
                    rm = row_mapping.get(row)
                    if rm is None:
                        ylabel = self.y_label 
                    else:
                        ylabel = str(rm) + '\n' + self.y_label
                    current_axis.set_ylabel(ylabel)
                else:
                    current_axis.set_yticks([])
                
                if (row_num == num_rows - 1) and (col_num == 0):
                    current_axis.set_xlabel(self.xlabel)

                

                if row_num == 0:
                    col_title = col_mapping.get(col)
                    if col_title is None:
                        col_title = col
                    current_axis.set_title(col_title)
                current_axis.set_ylim(self.y_lim)
                current_axis.set_xlim([-1 * 0.05 * ax_max * self.x_scale, ax_max * self.x_scale * 1.2])
       
        # Determine the number of columns to use for the legend
        MAX_NUM_COLS = 6
        if current_color > MAX_NUM_COLS:
            n_cols = MAX_NUM_COLS
            
            # Balance the rows
            num_rows_needed = np.ceil(current_color / MAX_NUM_COLS)
            n_cols = int(np.ceil(current_color / num_rows_needed))

        else:
            num_rows_needed = 1
            n_cols = current_color

        if num_rows_needed > 1:
            self.large_bottom = True

        if self.large_bottom:
            fig.subplots_adjust(bottom=0.15)

        # Correct ordering in legend
        if self.legend_mapping is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            inverse_dict = {}
            for num, value in enumerate(self.legend_mapping.values()):
                inverse_dict[value] = num

            order = []
            other_order = []
            for num, label in enumerate(labels):
                if label in list(inverse_dict.keys()):
                    order.append(inverse_dict[label])
                else:
                    other_order.append(num)
            order.extend(other_order)
            order = np.argsort(order)
            handles = [handles[idx] for idx in order]
        else:
            handles = None

        fig.legend(handles=handles, ncol=n_cols, title=self.legend_title, loc='outside lower right')

        # Save the figures
        image_dir = get_folder('images')
        experiment_name = self.experiment_name[0]
        if self.save_dir is None:
            save_dir = join(image_dir, experiment_name)
        else:
            save_dir = join(image_dir, self.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = join(save_dir, self.save_name)
        print('Saving to: ' + save_name)
        plt.tight_layout()
        plt.savefig(save_name + '.png')
        plt.savefig(save_name + '.svg')

