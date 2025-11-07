from ppo_her.base.utils import get_folder, get_experiment_name
from os.path import join, exists
import os
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from ppo_her.processing.bar_plot import BarPlot

class Compare():

    '''
        Compare 2 conditions
        This creates bar plots

        Parameters:
        stat_path (str) - the path to where the statistics are stored
        experiment_name (str) - the name of the experiment where the conditions were created
    '''
    def __init__(self, stat_path, experiment_name):
        self.stat_path = stat_path
        self.experiment_name = experiment_name
        self.define_constants()
        self.load_data()
        self.collect_data()
        self.organize_data()
        self.plot_data()
        self.save_data()

    def define_constants(self):
        # Set default values if they don't exist
        self.null_mapping = {
            'use_h': True,
        }

        # Rename columns if they don't have the right names
        self.rename_dict = {
            'spawn_policy': 'Spawn Method',
            'prey_velocity': 'Prey Policy',
        }

        # Names of the metrics to measure
        self.names = ['time_to_learn_steps', 'time_to_learn_time', 'max_median_reward', 'iqr']

        # State if we want low or high values for each metric
        self.desirable = ['low', 'low', 'high', 'low']

        # Set scaling factor to display data in a meaningful way
        self.division_factor = [1000, 60, 1, 1]

        # Determine which column we should use to split conditions
        self.condition_columns = ['prey_velocity', 'Type']


    def load_data(self):

        modes = ['step', 'time']
        data = []
        for mode in modes:
            file_name = mode + '_stats.csv'
            full_path = join(self.stat_path, file_name)
            data.append(pd.read_csv(full_path))
        stats = data[0]
        stats = stats.rename({'time_to_learn': 'time_to_learn_steps'}, axis=1)
        stats['time_to_learn_time'] = data[1].loc[:, 'time_to_learn']

        # Set default values        
        for column in stats:
            if column in list(self.null_mapping.keys()):
                stats.loc[stats[column].isnull(), column] = self.null_mapping[column]   

        
        self.stats = stats

    def collect_data(self):   
        
        stats = self.stats

        # Collect the necessary data
        time_steps = []
        time_seconds = []
        reward = []
        iqr = []

        base_columns = list(self.rename_dict.keys())
        base_columns.extend(self.condition_columns)
        base_columns + ['max_median_reward']

        # Rewards
        stats.loc[:, 'max_median_reward'] = (stats.max_median_reward + 1) / 2
        reward_group = stats.loc[:, base_columns + ['max_median_reward']]
        reward.append(reward_group)

        # Time steps
        step_group = stats.loc[:,  base_columns + ['time_to_learn_steps']]
        time_steps.append(step_group)

        # Time seconds
        time_group = stats.loc[:,  base_columns + ['time_to_learn_time']]
        time_seconds.append(time_group)

        # IQR
        iqr_group = stats.loc[:,  base_columns + ['iqr']]
        iqr.append(iqr_group)

        tables = [time_steps, time_seconds, reward, iqr]
        self.tables = tables        

    def rename_columns(self, table):
        table.loc[table.loc[:, 'use_h'] == True, 'use_h'] = '-HER'
        table.loc[table.loc[:, 'use_h'] == False, 'use_h'] = ''
        return table
    
    def move_conditions_to_columns(self, table, table_num):

        #List all experimental conditions
        names = self.names
        unique_vals = []
        for cc in self.condition_columns:
            unique_vals.append(pd.unique(table.loc[:, cc]))

        # Get a list of all experimental condition combinations
        conditions = list(product(*unique_vals))

        col_names = []
        for condition in conditions:

            # Create the new column
            condition_for_name = [str(cond) for cond in condition]
            col_name = ' '.join(condition_for_name)
            col_names.append(col_name)
            table[col_name] = np.nan

            # Determine which data should be moved to it
            should_fill = np.zeros(len(table))
            for name, val in zip(self.condition_columns, condition):
                is_condition = (table.loc[:, name] == val)
                should_fill += is_condition

            table.loc[should_fill.values == 1, col_name] = table.loc[should_fill.values == 1, names[table_num]]

        table = table.drop([names[table_num]] + self.condition_columns, axis=1)
        
        return table, col_names


    def organize_data(self):
        tables = self.tables
        self.plot_tables = []

        # Concatenate
        desirable = self.desirable
        division_factor = self.division_factor
        num_tables = len(tables)
        for table_num in range(num_tables):

            # Concatnate all conditions
            table = tables[table_num]
            table = pd.concat(table, axis=0)

            # Rename conditions for column headings
            table = self.rename_columns(table)
            
            # Move condition values to columns
            table, col_names = self.move_conditions_to_columns(table, table_num)

            # Group the varaibles and take nan_max to remove nans
            all_groups = []
            grouped = table.groupby(['spawn_policy', 'prey_velocity'])
            for name, group in grouped:
                group = group.reset_index()
                group.loc[0, col_names] = np.nanmax(group.loc[:, col_names], axis=0)
                all_groups.append(group.loc[0, :])
                
            # Remove nans
            table = pd.DataFrame(all_groups).reset_index().drop(['index', 'level_0'], axis=1)
            table.loc[:, col_names] = table.loc[:, col_names] / division_factor[table_num]  

            # Plot the bars
            self.plot_tables.append(deepcopy(table))

            # Determine if high or low is desirable
            if desirable[table_num] == 'low':
                arg_fxn = np.nanargmin
            else:
                arg_fxn = np.nanargmax

            # Bold the desireable values
            desire_values = arg_fxn(table.loc[:, col_names], axis=1)
            desire_values = np.asarray([col_names[dv] for dv in desire_values])
            for num, dv in enumerate(desire_values):
                table.loc[num, dv] = '\\textbf{' + "{:.2f}".format(table.loc[num, dv]) + '}'

            rename_dict = self.rename_dict
            table = table.rename(rename_dict, axis=1)
            table = table.set_index(['Spawn Method', 'Prey Policy'])

            tables[table_num] = table

        self.tables = tables

    def plot_data(self):

        # Create a custom metric
        time_table = self.plot_tables[0].set_index(['spawn_policy', 'prey_velocity'])
        perf_table = self.plot_tables[2].set_index(['spawn_policy', 'prey_velocity'])
        print(time_table)
        print(perf_table)
        all_columns = list(time_table.columns)
        
        time_table.loc[:, all_columns] = 1 - (time_table.loc[:, all_columns] / np.max(time_table.loc[:, all_columns], axis=1).values.reshape(-1, 1))
        custom_metric = perf_table * time_table
        custom_metric = custom_metric.reset_index()
        self.plot_tables.append(custom_metric)
        self.names.append('custom_metric')
        print(custom_metric)

        names = {
            'max_median_reward': 'Maximum Median Return',
            'time_to_learn_steps': 'Time to Learn (steps)',
            'custom_metric': '|Return| * |Time|',
        }
        
        for num, table in enumerate(self.plot_tables):
            table = table.set_index(['spawn_policy', 'prey_velocity'])
            plot_name = self.names[num]

            datas = ['max_median_reward', 'time_to_learn_steps', 'custom_metric']          
            if plot_name in datas:
                bp = BarPlot(
                    table, plot_name=plot_name,
                    experiment_name=self.experiment_name,
                    ylabel=names[plot_name]
                )


    def save_data(self):
        tables = self.tables

        for table_num, table in enumerate(tables):
            file_name = get_folder('tex')
            if not exists(file_name):
                os.mkdir(file_name)

            file_name = join(file_name, self.experiment_name)
            if not exists(file_name):
                os.mkdir(file_name)

            file_name = join(file_name, self.names[table_num] + '.tex')
            table.to_latex(file_name, float_format='%.2f')


if __name__ == '__main__':
    folder = get_folder('stats')
    experiment_name = get_experiment_name(__file__)
    stat_path = join(folder, experiment_name)

    c = Compare(stat_path, experiment_name)







