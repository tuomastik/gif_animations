import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from tqdm import tqdm

from utils import create_3d_figure


def create_animation_frames(gif_frames_output_folder: str) -> None:
    rs = np.random.RandomState(1)
    elev, azim, dist = 90, 80, 10  # Initial camera position

    # Generate random bivariate dataset
    x, y = rs.randn(2, 50)
    x_orig, y_orig = x, y

    # elev & dist during animation: steady - change - steady - change
    frames_tot = 360
    frames_elev_change = 51
    frames_elev_steady = (frames_tot - 2*frames_elev_change) / 2
    assert(frames_tot == 2*frames_elev_steady + 2*frames_elev_change)

    print("Creating frames of the KDE GIF...")
    for i in tqdm(range(1, frames_tot, 1)):

        if i < frames_elev_steady:
            small_random_number = np.random.randn(1)[0]*0.005
            elev = elev + small_random_number  # Needed for camera to refresh
        elif i < frames_elev_steady + frames_elev_change:
            elev -= 1
            dist -= 0.015
        elif i < 2*frames_elev_steady + frames_elev_change:
            elev, dist = elev, dist
        elif i < 2*frames_elev_steady + 2*frames_elev_change:
            elev += 1
            dist += 0.015

        f, ax = create_3d_figure(figsize=(5, 5),
                                 xlim=(-2.8, 1.6),
                                 ylim=(-1.6, 1.6),
                                 zlim=None,
                                 background_color="#131919",
                                 elev=elev,
                                 azim=azim + (i / frames_tot * 2*360),
                                 dist=dist)

        # For fast rendering, keep gridsize small
        cm = sns.cubehelix_palette(start=i/frames_tot*3, light=1, as_cmap=True)
        sns.kdeplot(x, y, cmap=cm, shade=True, cut=1, ax=ax, gridsize=200)
        colors = [c._facecolors[0] for c in ax.get_children() if
                  isinstance(c, art3d.Poly3DCollection)]
        brightest_color = colors[np.argmax([np.sum(c) for c in colors])]
        ax.patch.set_facecolor(brightest_color)
        f.set_facecolor(brightest_color)
        # plt.show()
        plt.draw()
        file_name = os.path.join(gif_frames_output_folder,
                                 "frame_%s.png" % str(i).zfill(4))
        plt.savefig(file_name, format="png", dpi=300)
        plt.close("all")

        if i == frames_tot - 2:
            x, y = x_orig, y_orig
