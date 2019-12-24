from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def rotate_point(x: float,
                 y: float,
                 center_x: float,
                 center_y: float,
                 angle_deg: float):
    # Rotate point (x, y) around center point (center_x, center_y)
    # https://en.wikipedia.org/wiki/Rotation_matrix
    angle_radians = angle_deg / 180.0 * np.pi
    sine, cosine = np.sin(angle_radians), np.cos(angle_radians)
    new_x, new_y = np.dot(np.array([x, y]) - np.array([center_x, center_y]),
                          [[cosine, sine],
                           [-sine, cosine]]) + np.array([center_x, center_y])
    return new_x, new_y


def create_2d_figure(figsize: Tuple[float, float],
                     background_color: str,
                     xlim: Tuple[float, float],
                     ylim: Tuple[float, float]):
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    # ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax


def create_3d_figure(figsize: Tuple[float, float],
                     background_color: str,
                     xlim: Tuple[float, float] = None,
                     ylim: Tuple[float, float] = None,
                     zlim: Tuple[float, float] = None,
                     elev: float = None,
                     azim: float = None,
                     dist: float = None,
                     is_tight_layout: bool = False,
                     proj_type: str = "persp"):
    fig = plt.figure(figsize=figsize, facecolor=background_color,
                     frameon=False)
    # ax = fig.add_subplot(111, projection="3d", proj_type=proj_type)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type=proj_type)
    ax.patch.set_facecolor(background_color)
    ax.axis('off')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    if elev is not None:
        ax.elev = elev
    if azim is not None:
        ax.azim = azim
    if dist is not None:
        ax.dist = dist

    if is_tight_layout:
        plt.tight_layout()
    return fig, ax
