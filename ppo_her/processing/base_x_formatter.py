import numpy as np
from abc import ABC

class XFormatter(ABC):
    '''
        Format the x-axis of a plot for the Predator-Prey environments
    '''

    def define_step_labels(self):
        self.x_data = 'step'
        self.xlabel = 'Time (1000 steps)'
        self.save_name = 'steps'
        self.format_x_axis = False
        self.her_multiplier = 0.5
        self.x_scale = 1 / 1000

    def define_time_labels(self):
        self.x_data = 'wall_time'
        self.xlabel = 'Time (Minutes)'
        self.save_name = 'time'
        self.format_x_axis = True
        self.her_multiplier = 1
        self.x_scale = 1

    def change_x_axis_step(self, data):
        data = data.loc[:, ['value', self.x_data]]
        return data
 
    def change_x_axis_time(self, data):
        unique_dirs = np.unique(data.dir_name)
        
        # Make sure that time starts at 0 for every run  
        for un in unique_dirs:
            data.loc[data.dir_name == un, self.x_data] = (
                data.loc[data.dir_name == un, self.x_data]
                - np.min(data.loc[data.dir_name == un, self.x_data])
            )

        # Create the time bins 
        max_loc = np.max(data.loc[:, self.x_data])
        bins = np.linspace(0, max_loc, 100)
        SEC_PER_MIN = 60
        
        # Seperate into time bins
        data.loc[:, self.x_data] = bins[np.digitize(data.loc[:, self.x_data], bins=bins) - 1] / SEC_PER_MIN
        data = data.loc[:, ['value', self.x_data]]
        return data
    
    def group_by(self):
        self.standard_columns = ['step', 'tag', 'value', 'wall_time']

        self.row_type = 'spawn_policy'
        self.column_type = 'prey_velocity'


class SACXFormatter(XFormatter):
    
    '''
        Format the x-axis of a plot for the Predator-Prey environments
        Specialized for SAC, instead of PPO
            Differs in terms of bin size
    '''

    def change_x_axis_step(self, data):
        print('Using SAC X Formatter')

        # Create the time bins 
        max_loc = np.max(data.loc[:, self.x_data])
        bins = np.linspace(0, max_loc, 100)
        
        # Seperate into time bins
        data.loc[:, self.x_data] = bins[np.digitize(data.loc[:, self.x_data], bins=bins) - 1]
        data = data.loc[:, ['value', self.x_data]]
        return data
