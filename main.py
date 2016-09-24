import os

import numpy as np
import matplotlib.pyplot as plt


def create_figure(min_val, max_val):
    background_color = '#131919'
    fig = plt.figure(figsize=(5, 5), facecolor=background_color)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.patch.set_facecolor(background_color)
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))
    fig.tight_layout()
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
    lines = []
    for x_point, y_point in zip(x, y):
        x_on_line, y_on_line = get_closest_point_on_line(
            x_point, y_point, a, b, c)
        line_proj, = ax.plot([x_point, x_on_line], [y_point, y_on_line],
                             linewidth=1, color='cyan', alpha=0.8)
        lines.append(line_proj)
    return lines


def rotate_2d(ims_output_folder):

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

    # Visualize points
    ax.scatter(x, y, color='cyan', s=50, marker='^')

    line, projected_lines = None, None
    max_degrees = 361
    max_digits = len(str(max_degrees))
    for frame, _ in enumerate(np.arange(1, 361)):

        # Rotate points
        x0, y0 = rotate_point(x=x0, y=y0, center_point=center, angle_deg=1)
        x1, y1 = rotate_point(x=x1, y=y1, center_point=center, angle_deg=1)

        if line is not None and projected_lines is not None:
            # Remove last drawn lines if exist
            lines = projected_lines + [line]
            for l in lines:
                l.remove()

        # Plot rotating line
        line, = ax.plot([x0, x1], [y0, y1], linewidth=3, color='magenta')
        # Get line equation
        a, b, c = get_line_equation(x0, y0, x1, y1)
        # Draw perpendicular projections
        projected_lines = draw_projections(ax, x, y, a, b, c)

        # plt.draw()
        # plt.show()

        # plt.tight_layout()
        file_name = os.path.join(ims_output_folder,
                                 'frame_%s.png' % str(frame).zfill(max_digits))
        plt.savefig(file_name, format='png', facecolor=fig.get_facecolor())


def create_gif(ims_input_folder, gif_output_name, delay=2):
    # ImageMagick
    os.system(("{path_to_convert} -delay {delay} "
               "{ims_folder}/*png {gif_name}").format(
        path_to_convert="/opt/local/bin/convert", delay=delay,
        ims_folder=ims_input_folder, gif_name=gif_output_name))


if __name__ == '__main__':
    plt.close("all")

    # Create output folder
    gif_frames_output_folder = 'gif_frames_2d'
    if not os.path.exists(gif_frames_output_folder):
        os.makedirs(gif_frames_output_folder)

    rotate_2d(ims_output_folder=gif_frames_output_folder)
    create_gif(ims_input_folder=gif_frames_output_folder,
               gif_output_name='rotating_projections_2d.gif', delay=3)
