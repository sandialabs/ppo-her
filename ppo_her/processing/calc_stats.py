import pandas as pd
from ppo_her.base.utils import get_folder
from os.path import join, exists
import os
import numpy as np

class StatCalc():

    '''
        Calculate the maximum median reward and time to learning
        Inputs:
            file_location (str) - where the data is to summarize
            x_axis (str) - the name of the column containing the x-axis data
            experiment_name (str) - the name of the experiment
            mode (str) - the name of the x_axis
            extra_columns (list of dicts) - list of extra columns to add when loading multiple folders
    '''
    def __init__(self, file_location, x_axis, experiment_name, mode, extra_columns=None):
        self.file_location = file_location
        self.x_axis = x_axis
        self.experiment_name = experiment_name
        self.mode = mode
        self.extra_columns = extra_columns

        self.TIME_TO_LEARN_THRESHOLD = 0.95 # 95% of reward range

        self.load_data()
        self.summarize_data()
        self.save_stats()

    def load_data(self):
        if not isinstance(self.file_location, list):
            file_location = [self.file_location]
        else:
            file_location = self.file_location
        
        self.data = []
        for num, fl in enumerate(file_location):
            data = pd.read_json(fl)
            if self.extra_columns is not None:
                for key, value in self.extra_columns[num].items():
                    data.loc[:, key] = value
            self.data.append(data)
        if len(self.data) > 1:
            self.data = pd.concat(self.data)
        else:
            self.data = self.data[0]

    def summarize_data(self):
        max_median_rewards = []
        time_to_learn = []
        iqr = []

        for index, data in self.data.iterrows():
            
            # Find the maximum median reward
            if len(data.med) == 0:
                max_median_rewards.append(np.nan)
                time_to_learn.append(np.nan)
                iqr.append(np.nan)

                continue
            max_med_reward = np.max(data.med)
            max_median_rewards.append(max_med_reward)

            # Determine where the time to learn should be
            scaled_reward = (max_med_reward + 1) / 2 # Scale to [0, 1]
            scaled_threshold = self.TIME_TO_LEARN_THRESHOLD * scaled_reward # Get 95% of the value
            reward_threshold = (scaled_threshold * 2) - 1 # Scale back to [-1, 1]

            # Determine where the first instance
            time_to_learn_idx = np.where(data.med >= reward_threshold)[0][0]
            time_to_learn.append(data[self.x_axis][time_to_learn_idx])

            # Determine the variability
            max_loc = np.where(data.med == max_med_reward)[0][-1]
            iqr.append(data.high[max_loc] - data.low[max_loc])


        self.data.loc[:, 'time_to_learn'] = time_to_learn
        self.data.loc[:, 'max_median_reward'] = max_median_rewards
        self.data.loc[:, 'iqr'] = iqr


    def save_stats(self):
        folder = get_folder('stats')
        if not exists(folder):
            os.mkdir(folder)
        
        sub_folder = join(folder, self.experiment_name)
        if not exists(sub_folder):
            os.mkdir(sub_folder)

        file_name = join(sub_folder, str(self.mode) + '_stats.csv')
        print('Saving stats to: ' + str(file_name))
        self.data.to_csv(file_name)





