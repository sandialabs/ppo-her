
import matplotlib.pyplot as plt
import numpy as np

'''
    Reference:
    https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
'''

'''
    Draw a sphere
    Inputs:
        ax (matplotlib axis) - the axis to draw the sphere on
        size (float) - the radius of the sphere
        pos (list) - the x,y,z coordinates of the sphere center
        color (char) - the color of the sphere

'''
def draw_sphere(ax, size, pos, color):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = size * np.cos(u)*np.sin(v) + pos[0]
    y = size * np.sin(u)*np.sin(v) + pos[1]
    z = size * np.cos(v) + pos[2]
    ax.plot_wireframe(x, y, z, color=color, linewidth=0.1)
    ax.set_box_aspect(np.asarray((np.ptp(x), np.ptp(y), np.ptp(z))) * 10)
  

if __name__ == '__main__':
    grid_size = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    poses = [
        (5, 5, 5),
        (10, 10, 10),
        (0, 0, 0),
    ]
    colors = [
        'g',
        'r',
        'b',
     ]
    sizes = [        
        0.1,
        0.0000001,
        0.0000001,
     ]
    for pos, color, radius in zip(poses, colors, sizes):
        draw_sphere(ax, radius, pos, color)
        plt.savefig('./sphere.png')
