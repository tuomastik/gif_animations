import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


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
    z = Image.open('mountain_elevation_map.jpg').convert('L')

    # Interpolation (this is only needed if you want the surface to be plotted
    #                to have different dimensions as the elevation map image)
    n = 200
    z_interp = np.asarray(z.resize(size=(n, n), resample=Image.NEAREST))
    x_interp, y_interp = np.meshgrid(np.linspace(0, z.width, num=n),
                                     np.linspace(0, z.height, num=n))

    surf = ax.plot_surface(x_interp, y_interp, z_interp, rstride=1, cstride=1,
                           color=bg_color, cmap='magma', linewidth=0.7,
                           antialiased=True, shade=False, alpha=1.0,
                           vmin=z_interp.min(), vmax=z_interp.max())
    surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors((0, 0, 0, 0))
    ax.axis('off')

    # Position the surface nicely in the center of the figure
    ax.set_zlim((z_interp.min() - 4 * z_interp.std(),
                 z_interp.max() + 2 * z_interp.std()))
    ax.set_xlim((x_interp.min() + 0.8 * x_interp.std(),
                 x_interp.max() - 0.8 * x_interp.std()))
    ax.set_ylim((y_interp.min() + 0.8 * y_interp.std(),
                 y_interp.max() - 0.8 * y_interp.std()))

    # Create frames for the animation
    print("Creating frames for mountain GIF...")
    frames_total = 230
    for frame in tqdm(range(frames_total)):
        # Rotate camera 360 degrees during the loop
        ax.view_init(elev=elev, azim=azim + (frame / frames_total * 360))
        save_image(gif_frames_output_folder, frame, 3, bg_color)

    return gif_frames_output_folder
