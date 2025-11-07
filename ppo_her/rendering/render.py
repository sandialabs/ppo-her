import matplotlib.pyplot as plt
import numpy as np
import imageio
from os import remove, listdir
from tqdm import tqdm
from ppo_her.rendering.sphere import draw_sphere


class Render3d():

    '''
        Render an epoch of the Predator-Prey environment as a gif in 3d
        An example is provided at the bottom of the file

        Inputs:
            observations (list of dicts) - the observations of all agents at each timestep.
            num_dims (int) - the number of physical dimensions (2D, 3D, etc...)
            grid_size (int) - the size of the workspace in arbitrary units.  The
                workspace is assumed to be a hypercube
            gif_name (str) - the name to use when saving the gif file
            prey_radius (float) - the radius of the prey, in workspace arbitrary units
            base_radius (float) - the radius of the base and the predator
            reward (float) - the return achieved for the episode.  This is used to label the episode
                as either a success (reward > 0) or failure (reward < 0)
    '''

    def __init__(
        self, observations, num_dims=3, grid_size=10,
        gif_name=None, prey_radius=1,
        base_radius=1, reward=None
        ):
        
        # Input handling
        self.observations = self.get_observations(observations)
        self.num_dims = num_dims
        self.grid_size = grid_size

        # Get the names
        self.names = list(self.observations.keys())

        if gif_name is None:
            self.gif_name='./episode.gif'
        else:
            self.gif_name = str(gif_name)

        # Initialize the plot 
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        def get_agent_color(agent):
            if 'prey' in agent:
                color='b'
                radius = prey_radius
            elif 'base' in agent:
                color = 'k'
                radius = base_radius
            else:
                color='r'
                radius = prey_radius

            return color, radius

        count = 0
        print('Creating *.pngs:')

        # Create a *.png file for each step
        for obs_num in range(np.shape(self.observations['prey'])[0]):

            # Clear the figure
            ax.clear()

            # Plot hidden spheres in corners to show whole workspace
            zero_z = np.ones(3)
            zero_z[-1] = 0
            draw_sphere(ax, 1E-6, -11 * zero_z, 'b')
            draw_sphere(ax, 1E-6, 11 * np.ones(3), 'b')

            # Plot each agent
            for agent_num, agent in enumerate(self.names):
                color, radius = get_agent_color(agent)

                # Get the position
                pos = self.observations[agent][obs_num, :]
                if len(pos) == 2:
                    # Make 3D, if necessary
                    pos = np.append(pos, [0])

                draw_sphere(ax, radius, pos, color)

                # Plot lines
                for dim in range(self.num_dims):
                    pos2 = np.copy(pos)
                    if (dim == 0) | (dim == 2):
                        pos2[dim] = -1 * self.grid_size
                    elif dim == 1:
                        pos2[dim] = self.grid_size
                    
                    ax.plot(
                        [pos[0], pos2[0]],
                        [pos[1], pos2[1]],
                        [pos[2], pos2[2]],
                        linewidth=0.5,
                        color=color,
                    )
            
            result_string = ''
            if obs_num == (np.shape(self.observations['prey'])[0] - 1):
                if reward is not None:
                    if reward > 0:
                        result_string = ' - Success'
                    elif reward < 0:
                        result_string = ' - Failure'

            # Title and save
            plt.title('Step #' + str(obs_num) + result_string)
            plt.savefig('_' + str(obs_num) + '.png')
            count += 1      
        
        # Create gif by combining all *.pngs
        self.create_gif(count)

    def get_observations(self, observations):
        '''
            Get the predator and prey observations from the observations

            Parameters:
            observations (list of dicts) - the observations of all agents at each timestep.
        '''

        # Initialize
        pred_positions = np.asarray([obs['achieved_goal'] for obs in observations])
        prey_positions = np.asarray([obs['desired_goal'] for obs in observations])
        num_dims = np.shape(prey_positions)[1]

        # Prey positions
        observations = {
            'prey': prey_positions,
        }

        # Predator positions
        num_preds = int(np.shape(pred_positions)[1] / num_dims)
        for pred_num in range(num_preds):
            observations['predator' + str(pred_num)] = pred_positions[
                :, (num_dims * pred_num):(num_dims * (pred_num + 1))
            ]

        return observations
    
    def create_pngs(self):

        # Update
        count = 1
        print('Creating *.pngs:')
        for num in range(np.shape(self.observations['prey'])[0]):

            for key, val in self.observations.items():
                pos = val[num, :]
                self.agents[key].set(center=pos)
            plt.title('Step #' + str(num))
            plt.savefig(str(count) + '.png')
            count += 1
        return count - 1

    def create_gif(self, count):
        # Create gif
        file_names = ['_' + str(i) + '.png' for i in range(count)]

        if len(file_names) == 0:
            print('No movements - predator spawned on prey')
            return

        with imageio.get_writer(self.gif_name, mode='I', duration=0.01) as gif_writer:

            print('Creating *.gif:')
            # Every movement
            for file_name in tqdm(file_names):
                image = imageio.imread(file_name)
                gif_writer.append_data(image)

            # Hold at end for a while
            for i in range(10):
                image = imageio.imread(file_name)
                gif_writer.append_data(image)


        # Remove png
        while len(file_names) > 0:
            for file in file_names:
                remove(file)
            file_names = listdir('./')
            file_names = [f for f in file_names if '_*.png' in f]

        # Notify user
        print('GIF saved!')


if __name__ == '__main__':
    NUM_OBS = 10
    NUM_FEATURES = 6
    observations = []
    for _ in range(NUM_OBS):
        obs = {
            'achieved_goal': np.random.random(size=(3,)),
            'desired_goal': np.random.random(size=(3,)),
            'observation': np.random.random(size=(3,)),
        }
        observations.append(obs)

    renderer = Render3d(
        observations, num_dims=3, grid_size=10,
        use_set_function=False, gif_name=None, prey_radius=1, base_radius=1
    )