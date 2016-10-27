import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
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

    @staticmethod
    def _rotate_point(x, y, z, x_rot, y_rot, z_rot, center):
        # Degrees to radians
        x_rot = x_rot / 180.0 * np.pi
        y_rot = y_rot / 180.0 * np.pi
        z_rot = z_rot / 180.0 * np.pi
        # Pre-calculate cosines and sines
        cos_x, sin_x = np.cos(x_rot), np.sin(x_rot)
        cos_y, sin_y = np.cos(y_rot), np.sin(y_rot)
        cos_z, sin_z = np.cos(z_rot), np.sin(z_rot)
        # Build rotation matrix
        a11 = cos_y * cos_z
        a12 = cos_x * sin_z + sin_x * sin_y * cos_z
        a13 = sin_x * sin_z - cos_x * sin_y * cos_z
        a21 = -cos_y * sin_z
        a22 = cos_x * cos_z - sin_x * sin_y * sin_z
        a23 = sin_x * cos_z + cos_x * sin_y * sin_z
        a31 = sin_y
        a32 = -sin_x * cos_y
        a33 = cos_x * cos_y
        # Matrix multiplication
        return np.dot(np.array([x, y, z]) - center,
                      [[a11, a12, a13],
                       [a21, a22, a23],
                       [a31, a32, a33]]) + center

    def rotate(self, x_rot, y_rot, z_rot, center):
        for (ix, x), (_, y), (_, z) in zip(np.ndenumerate(self.xx),
                                           np.ndenumerate(self.yy),
                                           np.ndenumerate(self.zz)):
            self.xx[ix], self.yy[ix], self.zz[ix] = self._rotate_point(
                x, y, z, x_rot, y_rot, z_rot, center)


def create_figure():
    background_color = '#131919'
    fig = plt.figure(figsize=(10, 10), facecolor=background_color,
                     frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor(background_color)
    ax.axis('off')
    return fig, ax


def disable_perspective():
    # Source: http://tinyurl.com/jthka7l
    def orthogonal_proj(zfront, zback):
        a = (zfront + zback) / (zfront - zback)
        b = -2 * (zfront * zback) / (zfront - zback)
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, a, b],
                         [0, 0, -0.0001, zback]])
    proj3d.persp_transformation = orthogonal_proj


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

    drawn_squares = []
    fig, ax = create_figure()
    # disable_perspective()
    # ax.view_init(elev=35.3, azim=45)
    for i in np.arange(1, 360, 40):
        # if drawn_squares:
        #     [square.remove() for square in drawn_squares]
        #     drawn_squares = []
        for face, c in zip(square_faces, ['r', 'g', 'b', 'c', 'm', 'y']):
            if i == 1:
                face.calculate_coords(min_val, max_val)
            face.rotate(i, i, i, center=center)
            drawn_squares.append(
                ax.plot_surface(face.xx, face.yy, face.zz, color=c, alpha=0.7))
