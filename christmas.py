import os

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from tqdm import tqdm

from utils import create_3d_figure


def draw_tree(ax):
    drawn_shapes = []
    height_of_lowest_triangle = 25

    # Triangle lower corners
    a = np.array([1, -1, 0])
    b = np.array([1, 1, 0])
    c = np.array([-1, 1, 0])
    d = np.array([-1, -1, 0])
    # Triangle top
    e = np.array([0, 0, height_of_lowest_triangle])
    triangles1 = np.array([[a, b, e],
                           [b, c, e],
                           [c, d, e],
                           [d, a, e]], dtype=float)
    bottom1 = np.array([[a, b, c, d]])

    triangles2 = np.array(triangles1)
    triangles2[:, :, :2] *= 0.8
    triangles2[:, :2, 2] += (0.5 * height_of_lowest_triangle)
    triangles2[:, 2, 2] += (0.4 * height_of_lowest_triangle)
    bottom2 = np.array([np.unique(np.vstack(triangles2[:, :2]), axis=0)])[:, [0, 1, 3, 2]]

    triangles3 = np.array(triangles2)
    triangles3[:, :, :2] *= 0.6
    triangles3[:, :2, 2] += (0.6 * height_of_lowest_triangle)
    triangles3[:, 2, 2] += (0.3 * height_of_lowest_triangle)
    bottom3 = np.array([np.unique(np.vstack(triangles3[:, :2]), axis=0)])[:, [0, 1, 3, 2]]

    trunk_low = -8
    trunk_width = 0.15
    a = np.array([trunk_width, -trunk_width, trunk_low])
    a2 = np.array([trunk_width, -trunk_width, 0])
    b = np.array([trunk_width, trunk_width, trunk_low])
    b2 = np.array([trunk_width, trunk_width, 0])
    c = np.array([-trunk_width, trunk_width, trunk_low])
    c2 = np.array([-trunk_width, trunk_width, 0])
    d = np.array([-trunk_width, -trunk_width, trunk_low])
    d2 = np.array([-trunk_width, -trunk_width, 0])
    trunk = np.array([
        # Bottom
        [a, b, c, d],
        # Sides
        [a, a2, b2, b],
        [b, b2, c2, c],
        [c, c2, d2, d],
        [d, d2, a2, a],
        # Top
        [a2, b2, c2, d2]
    ])

    # Draw the tree
    for shape, color, alpha in zip(
            [trunk, triangles1, bottom1, triangles2, bottom2, triangles3, bottom3],
            ["brown"] + ["#378b29"] * 6,
            [0.4] + [0.3] * 6):
        pc = Poly3DCollection(shape, linewidths=1, alpha=alpha)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        ax.add_collection3d(pc)
        drawn_shapes.append(pc)
    return drawn_shapes


def draw_top_light(ax):
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta) * 0.2
    y = np.sin(phi) * np.sin(theta) * 0.2
    z = np.cos(phi) * 2 + 44
    drawn_shapes = ax.plot_surface(x, y, z, color="gold", rcount=6, alpha=1, ccount=50, zorder=999)
    return [drawn_shapes]


def draw_ribbon(ax, figsize):
    drawn_shapes = []
    # Draw ribbon
    n_points = 1000
    theta = np.linspace(np.pi, 10 * np.pi, n_points)
    z = np.linspace(0, 70, n_points)
    r = z * 0.04
    z = z[:-int(0.4 * n_points)][::-1]
    x = (r * np.sin(theta))[:-int(0.4 * n_points)]
    y = (r * np.cos(theta))[:-int(0.4 * n_points)]
    drawn_shapes += ax.plot(x, y, z, color="white", alpha=1, zorder=-999, lw=figsize[0] / 5)

    # Draw ribbon lights
    np.random.seed(1)
    for i in np.random.choice(range(len(z)), size=30):
        drawn_shapes.append(ax.scatter(x[i], y[i], z[i], marker="*", color="white",
                                       # https://en.wikipedia.org/wiki/Arithmetic_progression
                                       s=10 + 40 * (figsize[0] / 5 - 1)))
    return drawn_shapes


def draw_text(ax, figsize):
    ax.text2D(-0.05, -0.075, "Happy Holidays!", color="white",
              # https://fonts.google.com/specimen/Pacifico
              fontproperties=fm.FontProperties(fname="Pacifico-Regular.ttf",
                                               size=figsize[0] / 5 * 25))
    ax.text2D(-0.064, -0.09, "Warm greetings, Artific Intelligence", color="white",
              # https://fonts.google.com/specimen/Montserrat
              fontproperties=fm.FontProperties(fname="Montserrat-Light.ttf",
                                               size=figsize[0] / 5 * 13))


def create_animation_frames(gif_frames_output_folder: str) -> None:
    figsize = (5, 5)
    assert(figsize[0] == figsize[1])
    elev, azim = 24, 25
    fig, ax = create_3d_figure(figsize=figsize,
                               background_color="#131919",
                               xlim=(-2, 2),
                               ylim=(-2, 2),
                               zlim=(-7, 31),
                               elev=elev,
                               azim=azim)

    draw_tree(ax)
    draw_top_light(ax)
    draw_ribbon(ax, figsize)
    draw_text(ax, figsize)

    print("Creating frames of the christmas GIF...")
    total_frames = 360
    for frame in tqdm(range(total_frames)):

        if frame > 1:
            # Rotate camera
            ax.view_init(elev=elev, azim=azim + (frame / total_frames * 360))

        # Save
        file_name = os.path.join(gif_frames_output_folder, 'frame_%s.png' % str(frame).zfill(3))
        plt.savefig(file_name, facecolor=fig.get_facecolor())
