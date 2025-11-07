from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import os
from ppo_her.base.utils import get_folder
from ppo_her.processing.colors import get_colors


class BarPlot():

    '''
        Plot data that has already been processed by a calc_diff.py as bar plots
        Inputs:
            data (pandas) - the data to plot
            experiment_name (str) - folder to store the data
            plot_name (str) - the name of the file to be stored
            large_bottom (bool) - add padding to bottom of figure
            ylabel (str) - ylabel
    '''

    def __init__(
            self,
            data,
            experiment_name=None,
            plot_name=None,
            large_bottom=False,
            ylabel=None,
        ):

        self.data = data
        self.experiment_name = experiment_name
        if plot_name is None:
            self.plot_name = 'test'
        else:
            self.plot_name = plot_name
        self.large_bottom = large_bottom

        self.BAR_START = 0
        self.BAR_END = 1
        self.xlabel = '% Success'
        self.ylabel = ylabel

        self.plot_data()
    
    def plot_data(self):

        data = self.data
        idx = np.asarray(data.index)
        split_idx = np.asarray([[te for te in ie] for ie in idx])

        # Get the number of sub-plots required
        unique_rows = np.unique(split_idx[:, 0])
        unique_columns = np.unique(split_idx[:, 1])
        num_rows = len(unique_rows)
        num_columns = len(unique_columns)

        # Create the figure
        fig, ax = plt.subplots(num_rows, num_columns, figsize=(11, 6))
        max_val = np.max(data.values) * 1.1
        min_val = np.min(data.values)
        if min_val > 0:
            min_val = 0

        # Map from short names to full names
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
        
        # Plot each instance
        for col_num, col in enumerate(unique_columns):
            for row_num, row in enumerate(unique_rows):

                current_axis = ax[row_num, col_num]
                
                # Build the index to get the data
                idx = (row, col)
                sub_data = data.loc[idx]
                num_bars = len(sub_data)

                # Multiply by 2 because we start at -1
                heights = sub_data.values.reshape(1, -1)
                if self.experiment_name in ['max_median_reward']:
                    heights = heights * 2
                full_x = np.linspace(self.BAR_START, self.BAR_END, num_bars)
                width = (self.BAR_END - self.BAR_START) / (num_bars + 1)

                bar_colors = get_colors(num_bars)

                if self.experiment_name in ['max_median_reward']:
                    bottom = -1
                else:
                    bottom = 0

                current_axis.bar(
                    full_x.squeeze(),
                    heights.squeeze(),
                    width=width,
                    color=bar_colors,
                    bottom=bottom,
                    label=np.arange(num_bars))

                # Only give axis labels to boundary plots
                if col_num == 0:
                    current_axis.set_ylabel(row_mapping[row] + '\n' + str(self.ylabel))
                else:
                    current_axis.set_yticks([])
                
                if (row_num == num_rows - 1) and (col_num == 0):
                    current_axis.set_xlabel(self.xlabel)

                if (row_num == num_rows - 1) and (col_num == 0):
                    current_axis.set_xlabel(self.xlabel)

                if row_num == 0:
                    if col in list(col_mapping.keys()):
                        col_title = col_mapping[col]
                    current_axis.set_title(col_title)

                if self.experiment_name == 'max_median_reward':
                    current_axis.set_ylim([-1.1, 1.1])
                else:
                    current_axis.set_ylim([min_val, max_val])
       
        # Determine the number of columns to use for the legend
        MAX_NUM_COLS = 6
        if num_bars > MAX_NUM_COLS:
            # Balance the rows
            num_rows_needed = np.ceil(num_bars / MAX_NUM_COLS)

        else:
            num_rows_needed = 1

        if num_rows_needed > 1:
            self.large_bottom = True

        if self.large_bottom:
            fig.subplots_adjust(bottom=0.15)

        # Save the figures
        image_dir = get_folder('images')
        experiment_name = self.experiment_name
        save_dir = join(image_dir, experiment_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = join(save_dir, self.plot_name)
        print('Saving to: ' + save_name)
        plt.savefig(save_name + '.png')
        plt.savefig(save_name + '.svg')

