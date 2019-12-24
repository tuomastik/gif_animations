# This animation is inspired by this video: https://youtu.be/XUA8UREROYE
# Also, see: https://en.wikipedia.org/wiki/Diffusion-limited_aggregation

import os
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from tqdm import tqdm

from utils import create_2d_figure, rotate_point


class Particle:
    def __init__(self,
                 x: float,
                 y: float):
        self.x = x
        self.y = y

    def update(self):
        self.y -= 1
        spread = 2
        self.x += np.random.uniform(-spread, spread)

    def rotate(self, angle_deg: float, center_x: float, center_y: float):
        self.x, self.y = rotate_point(x=self.x, y=self.y,
                                      center_x=center_x, center_y=center_y,
                                      angle_deg=angle_deg)
        return self

    def finished(self):
        return self.y < 0

    def intersects(self, points):
        result = False
        for p in points:
            euclidean_dist = np.linalg.norm(
                np.array((self.x, self.y)) -
                np.array((p.x, p.y)))
            if euclidean_dist < 1.1:
                result = True
                break
        return result


def save_image(fig, output_folder, frame, max_frame_digits):
    file_name = os.path.join(output_folder, 'frame_%s.png' %
                             str(frame).zfill(max_frame_digits))
    plt.savefig(file_name, format='png', facecolor=fig.get_facecolor())


def create_animation_frames(gif_frames_output_folder: str) -> None:
    np.random.seed(1)

    fig, ax = create_2d_figure(figsize=(5, 5),
                               background_color="#131919",
                               xlim=(-45, 45),
                               ylim=(-45, 45))

    colormap = cm.get_cmap('Blues')
    norm = colors.Normalize(vmin=0, vmax=40)

    snowflake = []

    for frame in tqdm(range(321)):

        p = Particle(x=0, y=40)
        while not p.finished() and not p.intersects(snowflake):
            p.update()

        points = [p,
                  copy(p).rotate(1 * 60, 0, 0),
                  copy(p).rotate(2 * 60, 0, 0),
                  copy(p).rotate(3 * 60, 0, 0),
                  copy(p).rotate(4 * 60, 0, 0),
                  copy(p).rotate(5 * 60, 0, 0)]

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        ax.scatter(x, y, marker="D", s=5,
                   c=x.size * [colormap(norm(np.sqrt((x[0] ** 2) + (y[0] ** 2))))])
        snowflake += points

        if frame % 4 == 0:
            save_image(fig, gif_frames_output_folder, frame, 3)
