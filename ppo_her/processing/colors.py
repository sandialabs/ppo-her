import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

'''
    Get unique colors for each condition
    Parameters:
        num_colors_needed (int) - the number of conditions
    Returns:
        bar_colors (list) - the colors
'''
def get_colors(num_colors_needed):
    # Get default colors
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Add colors as necessary
    if len(cmap) < num_colors_needed:
        num_repeats_needed = int(np.ceil(num_colors_needed / len(cmap))) - 1
        existing_colors = deepcopy(cmap)
        for repeat in range(num_repeats_needed):
            new_colors = [colors.hex2color(color) for color in existing_colors]
            new_colors = [np.asarray(color) / (repeat + 2) for color in new_colors]
            new_colors = [colors.rgb2hex(color) for color in new_colors]
            cmap.extend(new_colors)

    bar_colors = [cmap[n] for n in range(num_colors_needed)]
    return bar_colors


# Test
if __name__ == '__main__':
    bc = get_colors(24)
    print(bc)