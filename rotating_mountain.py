import os

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import mlab
from PIL import Image


def save_image(output_folder, frame, max_frame_digits, bg_color):
    file_name = os.path.join(output_folder, 'frame_%s.png' %
                             str(frame).zfill(max_frame_digits))
    plt.savefig(file_name, format='png', facecolor=bg_color)


def create_animation_frames():
    # Settings
    elev, azim = 30, 80  # Initial camera position

    # Create output directory
    gif_frames_output_folder = 'gif_frames_rotating_mountain'
    if not os.path.exists(gif_frames_output_folder):
        os.makedirs(gif_frames_output_folder)

    # Initialize figure
    bg_color = '#000000'
    ax = plt.figure(figsize=(5, 5), facecolor=bg_color,
                    frameon=False).add_subplot(111, projection='3d')
    ax.patch.set_facecolor(bg_color)

    # Open elevation map
    z = np.asarray(Image.open(
        'rotating_mountain_elevation_map.jpg').convert('L'))
    im_h, im_w = z.shape[:2]
    x, y = np.meshgrid(np.arange(im_w), np.arange(im_h))

    # Interpolation (this is only needed if you want the surface to be plotted
    #                to have different dimensions as the elevation map image)
    n = 200
    xi = np.linspace(0, im_w, num=n)
    yi = np.linspace(0, im_h, num=n)
    x_interp, y_interp = np.meshgrid(xi, yi)
    z_interp = mlab.griddata(x.ravel(), y.ravel(), z.ravel(), xi, yi)
    z_interp[np.isnan(z_interp)] = z_interp.min()

    # Create frames for the animation
    frames_grow = 40
    frames_steady = 150
    frames_total = frames_grow * 2 + frames_steady
    for frame in np.arange(frames_total):
        print("Rotating mountain: creating frame %s / %s..." % (
              frame, frames_total))
        if frame < frames_grow:
            # Mountain growing
            z_now = z_interp * float(frame+1)/frames_grow
        elif frame > frames_grow + frames_steady:
            # Mountain shrinking
            z_now = z_interp * (1 - (float(frame - frames_grow - frames_steady)
                                     / frames_grow))
        else:
            # Mountain steady
            z_now = z_interp
        surf = ax.plot_surface(x_interp, y_interp, z_now, rstride=1, cstride=1,
                               color=bg_color, cmap='magma', linewidth=0.7,
                               antialiased=True, shade=False, alpha=1.0,
                               vmin=z_interp.min(), vmax=z_interp.max())
        surf.set_edgecolors(surf.to_rgba(surf._A))
        surf.set_facecolors((0, 0, 0, 0))
        ax.axis('off')
        plt.show()
        # Position the surface nicely in the center of the figure
        ax.set_zlim((z_interp.min() - 4 * z_interp.std(),
                     z_interp.max() + 2 * z_interp.std()))
        ax.set_xlim((x_interp.min() + 0.8 * x_interp.std(),
                     x_interp.max() - 0.8 * x_interp.std()))
        ax.set_ylim((y_interp.min() + 0.8 * y_interp.std(),
                     y_interp.max() - 0.8 * y_interp.std()))
        # Rotate camera 360 degrees during the loop
        if frame > 0:
            ax.view_init(elev=elev,
                         azim=azim + (frame / float(frames_total) * 360))
        save_image(gif_frames_output_folder, frame, 3, bg_color)
        # Remove last drawn surface
        surf.remove()

    return gif_frames_output_folder
