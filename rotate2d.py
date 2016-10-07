import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm


def create_figure(min_val, max_val):
    background_color = '#131919'
    fig = plt.figure(figsize=(5, 5), facecolor=background_color)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))
    return fig, ax


def rotate_point(x, y, center_point, angle_deg):
    # Rotate point (x, y) around center_point
    # https://en.wikipedia.org/wiki/Rotation_matrix
    angle_radians = angle_deg / 180.0 * np.pi
    sine, cosine = np.sin(angle_radians), np.cos(angle_radians)
    new_x = (x-center_point) * cosine - (y-center_point) * sine + center_point
    new_y = (x-center_point) * sine + (y-center_point) * cosine + center_point
    return new_x, new_y


def get_line_equation(x0, y0, x1, y1):
    # ax + by + c = 0
    a = (y1 - y0) / np.float(x1 - x0)
    b = -1
    c = -a * x0 - b * y0
    return a, b, c


def get_closest_point_on_line(x, y, a, b, c):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    x, y = ((np.float(b) * (b * x - a * y) - a * c) / (a ** 2 + b ** 2),
            (np.float(a) * (-b * x + a * y) - b * c) / (a ** 2 + b ** 2))
    return x, y


def draw_projections(ax, x, y, a, b, c):
    drawn_points, coords, distances = [], [], []
    for x_point, y_point in zip(x, y):
        x_on_line, y_on_line = get_closest_point_on_line(
            x_point, y_point, a, b, c)
        # Calculate Euclidean distance with Pythagorean theorem
        distances.append(np.sqrt((x_point-x_on_line)**2 +
                                 (y_point-y_on_line)**2))
        coords.append([[x_point, x_on_line], [y_point, y_on_line]])

    min_distance, max_distance = np.min(distances), np.float(np.max(distances))
    # Get values for colormap
    colormap = cm.get_cmap('YlGnBu_r')
    norm = colors.Normalize(vmax=max_distance, vmin=min_distance)

    for i, (c, d, x_point, y_point) in enumerate(zip(coords, distances, x, y)):
        # Draw projection
        line_proj, = ax.plot(*c, linewidth=(2.3-d/max_distance)**2,
                             color=colormap(norm(d)), alpha=1.0, zorder=i)
        # Draw point
        point = ax.scatter(x_point, y_point, color=colormap(norm(d)),
                           s=60-d/max_distance*30, marker='o', zorder=i)
        # Append
        drawn_points += [line_proj, point]
    return drawn_points


def run_animation(ims_output_folder):

    # Settings
    n_data_points = 20
    min_val, max_val = 20, 80
    val_range = max_val - min_val
    np.random.seed(123)

    # Create data points
    x = np.random.randint(low=min_val, high=max_val, size=n_data_points)
    y = np.random.randint(low=min_val, high=max_val, size=n_data_points)

    # Create starting and ending point of the rotating line
    extra_length = 0.1
    x0, y0 = [min_val - extra_length * val_range] * 2
    x1, y1 = [max_val + extra_length * val_range] * 2
    center = min_val + np.round((max_val - min_val) / 2.0)

    # Create figure
    fig, ax = create_figure(min_val=x0, max_val=x1)

    drawn_shapes = None
    max_degrees = 361
    max_digits = len(str(max_degrees))
    for frame, _ in enumerate(np.arange(1, 361)):

        # Rotate points
        x0, y0 = rotate_point(x=x0, y=y0, center_point=center, angle_deg=1)
        x1, y1 = rotate_point(x=x1, y=y1, center_point=center, angle_deg=1)

        if drawn_shapes is not None:
            # Remove last drawn shapes if they exist
            for s in drawn_shapes:
                s.remove()

        # Get line equation
        a, b, c = get_line_equation(x0, y0, x1, y1)
        # Draw perpendicular projections
        drawn_shapes = draw_projections(ax, x, y, a, b, c)
        # Plot rotating line
        line, = ax.plot([x0, x1], [y0, y1], linewidth=4.5, color='magenta',
                        zorder=999)
        drawn_shapes.append(line)

        file_name = os.path.join(ims_output_folder,
                                 'frame_%s.png' % str(frame).zfill(max_digits))
        plt.savefig(file_name, format='png', facecolor=fig.get_facecolor())
