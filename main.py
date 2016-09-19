import numpy as np
import matplotlib.pyplot as plt


def create_figure(min_val, max_val):
    background_color = '#131919'
    fig = plt.figure(figsize=(10, 10), facecolor=background_color)
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


def rotate_2d():

    # Settings
    n_data_points = 20
    min_val, max_val = 20, 80
    val_range = max_val - min_val

    # Create data points
    x = np.random.randint(low=min_val, high=max_val, size=n_data_points)
    y = np.random.randint(low=min_val, high=max_val, size=n_data_points)

    # Create figure
    fig, ax = create_figure(min_val, max_val)

    # Visualize points
    ax.scatter(x, y, color='cyan', s=50, marker='^')

    # Plot line ax + by + c = 0
    x0, y0 = [min_val - 0.1 * val_range] * 2
    x1, y1 = [max_val + 0.1 * val_range] * 2
    center = min_val + np.round((max_val - min_val) / 2.0)

    line, projected_lines = None, None
    for _ in np.arange(1, 361):

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

        plt.draw()
        plt.show()

    # plt.tight_layout()
    # plt.savefig('rotate_2d.png', format='png',
    #             facecolor=fig.get_facecolor(), edgecolor='none')


if __name__ == '__main__':
    plt.close("all")
    rotate_2d()
