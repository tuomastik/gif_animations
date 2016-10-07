import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class Plane:
    def __init__(self, point, normal, xx=None, yy=None, zz=None, d=None):
        self.point = np.array(point).astype(float)  # [x_0, y_0, z_0]
        self.normal = np.array(normal).astype(float)  # [a, b, c]
        self.xx, self.yy, self.zz, self.d = xx, yy, zz, d

    @staticmethod
    def _rotate_array_elements(array, n):
        """ Rotate elements of array n times """
        start_part, end_part = array[:-n], array[-n:]
        if isinstance(array, list):
            return end_part + start_part
        elif isinstance(array, np.ndarray):
            return np.concatenate((end_part, start_part))

    def calculate_coords(self, min_val, max_val):

        # Calculate d = -(a*x_0 + b*y_0 + c*z_0)
        self.d = (-self.point * self.normal).sum()

        # Rotate dimensions if needed to avoid division by zero
        norm = self.normal
        times_rotated = 0
        while norm[2] == 0:
            times_rotated += 1
            norm = self._rotate_array_elements(norm, 1)

        coordinate_matrix_1, coordinate_matrix_2 = np.meshgrid(
            [min_val, max_val], [min_val, max_val])

        coordinate_matrix_3 = 1. * (
            -norm[0] * coordinate_matrix_1 -
            norm[1] * coordinate_matrix_2 - self.d) / norm[2]

        self.xx, self.yy, self.zz = self._rotate_array_elements(
            [coordinate_matrix_1, coordinate_matrix_2, coordinate_matrix_3],
            times_rotated)


def create_figure():
    background_color = '#131919'
    fig = plt.figure(figsize=(10, 10), facecolor=background_color,
                     frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor(background_color)
    ax.axis('off')
    return fig, ax


def run_animation():

    min_val, max_val = 20, 80  # For x, y and z
    val_range = max_val - min_val
    center = min_val + val_range / 2.0

    # Define the 6 square faces of cube using
    # point-normal form of the equation of a plane.
    # Points are in the centers of square faces.
    square_faces = [Plane(point=[center, center, min_val],  # xy-plane 1
                          normal=[0, 0, 1]),
                    Plane(point=[center, center, max_val],  # xy-plane 2
                          normal=[0, 0, 1]),
                    Plane(point=[center, min_val, center],  # xz-plane 1
                          normal=[0, 1, 0]),
                    Plane(point=[center, max_val, center],  # xz-plane 2
                          normal=[0, 1, 0]),
                    Plane(point=[min_val, center, center],  # yz-plane 1
                          normal=[1, 0, 0]),
                    Plane(point=[max_val, center, center],  # yz-plane 2
                          normal=[1, 0, 0])]

    fig, ax = create_figure()
    for square_face, c in zip(square_faces, ['r', 'g', 'b', 'c', 'm', 'y']):
        square_face.calculate_coords(min_val, max_val)
        ax.plot_surface(square_face.xx, square_face.yy, square_face.zz,
                        color=c, alpha=0.7)
