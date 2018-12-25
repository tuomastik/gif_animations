import os

import numpy as np
from mpl_toolkits.mplot3d import proj3d, art3d
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm


class Plane:
    def __init__(self, point, normal, xx=None, yy=None, zz=None, d=None,
                 face_colors=None):
        self.point = np.array(point).astype(float)  # [x_0, y_0, z_0]
        self.normal = np.array(normal).astype(float)  # [a, b, c]
        self.xx, self.yy, self.zz, self.d = xx, yy, zz, d
        self.face_colors = face_colors

    @staticmethod
    def _rotate_array_elements(array, n):
        """ Rotate elements of array n times clockwise """
        start_part, end_part = array[:-n], array[-n:]
        if isinstance(array, list):
            return end_part + start_part
        elif isinstance(array, np.ndarray):
            return np.concatenate((end_part, start_part))

    def calculate_coords(self, min_val, max_val, grid_size=5.0):
        # Calculate d = -(a*x_0 + b*y_0 + c*z_0)
        self.d = (-self.point * self.normal).sum()

        # Rotate dimensions if needed to avoid division by zero
        norm = self.normal
        times_rotated = 0
        while norm[2] == 0:
            times_rotated += 1
            norm = self._rotate_array_elements(norm, 1)

        step = np.abs(max_val-min_val)/np.float(grid_size)
        coordinate_matrix_1, coordinate_matrix_2 = np.array(np.meshgrid(
            np.arange(min_val, max_val+step, step),
            np.arange(min_val, max_val+step, step))).astype(float)
        coordinate_matrix_3 = 1. * (
            -norm[0] * coordinate_matrix_1 -
            norm[1] * coordinate_matrix_2 - self.d) / norm[2]

        self.xx, self.yy, self.zz = self._rotate_array_elements(
            [coordinate_matrix_1, coordinate_matrix_2, coordinate_matrix_3],
            times_rotated)

    @staticmethod
    def _rotate_point_around_xyz(x, y, z, x_rot, y_rot, z_rot, center):
        # Degrees to radians
        x_rot = x_rot / 180.0 * np.pi
        y_rot = y_rot / 180.0 * np.pi
        z_rot = z_rot / 180.0 * np.pi
        # Pre-calculate cosines and sines
        cos_x, sin_x = np.cos(x_rot), np.sin(x_rot)
        cos_y, sin_y = np.cos(y_rot), np.sin(y_rot)
        cos_z, sin_z = np.cos(z_rot), np.sin(z_rot)
        # Build rotation matrix
        # http://tinyurl.com/zpgddsa
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

    def rotate_around_xyz(self, x_rot, y_rot, z_rot, center):
        for (ix, x), (_, y), (_, z) in zip(np.ndenumerate(self.xx),
                                           np.ndenumerate(self.yy),
                                           np.ndenumerate(self.zz)):
            self.xx[ix], self.yy[ix], self.zz[ix] = \
                self._rotate_point_around_xyz(
                    x, y, z, x_rot, y_rot, z_rot, center)

    @staticmethod
    def _rotate_point_around_arbitrary_axis(x, y, z, ux, uy, uz, angle):
        """
        point = x, y, z
        unit vector of an arbitrary axis = ux, uy, uz
        """
        # Degrees to radians
        angle = angle / 180.0 * np.pi
        # Pre-calculate cosine and sine
        cos, sin = np.cos(angle), np.sin(angle)
        # Build rotation matrix
        # http://tinyurl.com/ka74357
        a11 = cos + (ux ** 2) * (1.0 - cos)
        a12 = ux * uy * (1.0 - cos) - uz * sin
        a13 = ux * uz * (1.0 - cos) + uy * sin
        a21 = uy * ux * (1.0 - cos) + uz * sin
        a22 = cos + (uy ** 2) * (1.0 - cos)
        a23 = uy * uz * (1.0 - cos) - ux * sin
        a31 = uz * ux * (1.0 - cos) - uy * sin
        a32 = uz * uy * (1.0 - cos) + ux * sin
        a33 = cos + (uz ** 2) * (1.0 - cos)
        # Matrix multiplication
        return np.dot(np.array([x, y, z]),
                      [[a11, a12, a13],
                       [a21, a22, a23],
                       [a31, a32, a33]])

    def rotate_around_arbitrary_axis(self, ux, uy, uz, angle):
        for (ix, x), (_, y), (_, z) in zip(np.ndenumerate(self.xx),
                                           np.ndenumerate(self.yy),
                                           np.ndenumerate(self.zz)):
            self.xx[ix], self.yy[ix], self.zz[ix] = \
                self._rotate_point_around_arbitrary_axis(
                    x, y, z, ux, uy, uz, angle)


def get_cube(min_val, max_val, center, grid_size):
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
    for face in square_faces:
        face.calculate_coords(min_val, max_val, grid_size=grid_size)
    return square_faces


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


def create_figure(min_val, max_val, elev, azim):
    fig = plt.figure(figsize=(5, 5), facecolor='#131919',
                     frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor('#131919')
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))
    ax.set_zlim((min_val, max_val))
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)
    disable_perspective()
    plt.tight_layout()
    return fig, ax


def colomapped_wireframe(plane, ax, stride, norm, colormap='hsv'):
    # Source: http://stackoverflow.com/a/24958192/5524090
    wire = ax.plot_wireframe(
        X=plane.xx, Y=plane.yy, Z=plane.zz, rstride=stride, cstride=stride)

    # Retrieve data from internal storage of plot_wireframe, then delete it
    nx, ny, _ = np.shape(wire._segments3d)
    wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
    wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
    wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
    wire.remove()

    # Create data for a LineCollection
    wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
    wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
    wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
    to_delete = np.arange(0, nx * ny, ny)
    wire_x1 = np.delete(wire_x1, to_delete, axis=1)
    wire_y1 = np.delete(wire_y1, to_delete, axis=1)
    wire_z1 = np.delete(wire_z1, to_delete, axis=1)
    scalars = np.delete(wire_z, to_delete)

    segs = [list(zip(xl, yl, zl)) for xl, yl, zl in
            zip(wire_x1.T, wire_y1.T, wire_z1.T)]

    # Plot the wireframe by a line3DCollection
    my_wire = art3d.Line3DCollection(segs, cmap=colormap, norm=norm,
                                     linewidths=2.0)
    my_wire.set_array(scalars)
    ax.add_collection(my_wire)
    return my_wire


def draw_cubes(ax, face_big, face_small, norm_small):
    return [
        ax.plot_wireframe(X=face_big.xx, Y=face_big.yy, Z=face_big.zz,
                          rstride=1, cstride=1, color='#465563', alpha=0.5,
                          linewidth=1.0),
        ax.scatter(face_big.xx, face_big.yy, face_big.zz, s=30,
                   c='#465563', lw=0, alpha=0.5),
        colomapped_wireframe(plane=face_small, ax=ax, stride=50,
                             norm=norm_small, colormap='cool')]


def save_image(fig, output_folder, frame, max_frame_digits):
    file_name = os.path.join(output_folder, 'frame_%s.png' %
                             str(frame).zfill(max_frame_digits))
    plt.savefig(file_name, format='png', facecolor=fig.get_facecolor())


def ease_out_quad(t, b, c, d):
    # Source: http://gizma.com/easing/
    # t = current time or frame
    # b = start value
    # c = change in value
    # d = duration in time or frames
    t /= d
    return -c * t*(t-2) + b


def ease_in_quad(t, b, c, d):
    # Source: http://gizma.com/easing/
    t /= d
    return c*t*t + b


def create_animation_frames():
    # Settings
    big_cube_min, big_cube_max = 20, 80  # For x, y, z
    center = big_cube_min + (big_cube_max - big_cube_min) / 2.0
    small_cube_min, small_cube_max = (big_cube_min + (center-big_cube_min)*0.5,
                                      big_cube_max - (center-big_cube_min)*0.5)
    elev, azim = 35.3, 45  # Initial camera position

    # Create output directory
    gif_frames_output_folder = 'gif_frames_cubes'
    if not os.path.exists(gif_frames_output_folder):
        os.makedirs(gif_frames_output_folder)

    # Initialize cubes
    big_cube = get_cube(big_cube_min, big_cube_max, center, grid_size=2.0)
    small_cube = get_cube(small_cube_min, small_cube_max, center,
                          grid_size=50.0)

    # Initialize figure
    fig, ax = create_figure(big_cube_min, big_cube_max, elev, azim)

    # Initialize color normalizer
    norm_small = colors.Normalize(vmin=small_cube_min, vmax=small_cube_max)

    print("Creating frames of the cubes GIF...")

    # Rotate cubes standing on their corners
    max_frames_rot1 = 61
    max_frames_rot2 = 254
    for frame in tqdm(range(max_frames_rot1)):
        drawn_shapes = []
        for face_big, face_small in zip(big_cube, small_cube):
            if frame > 0:
                face_small.rotate_around_arbitrary_axis(
                    ux=1/np.sqrt(3), uy=1/np.sqrt(3), uz=1/np.sqrt(3), angle=2)
                face_big.rotate_around_arbitrary_axis(
                    ux=1/np.sqrt(3), uy=1/np.sqrt(3), uz=1/np.sqrt(3), angle=2)
            drawn_shapes += draw_cubes(ax, face_big, face_small, norm_small)
        save_image(fig, gif_frames_output_folder, frame, 3)
        # Remove last drawn shapes
        [shape.remove() for shape in drawn_shapes]

    # Rotate small cube around x and z axis
    frames_accelerate = 70
    for frame in tqdm(np.arange(1, max_frames_rot2, 1)):
        drawn_shapes = []
        for face_big, face_small in zip(big_cube, small_cube):
            drawn_shapes += draw_cubes(ax, face_big, face_small, norm_small)

            if frame <= frames_accelerate:
                # Start rotating slowly in the beginning
                face_small.rotate_around_xyz(
                    ease_in_quad(float(frame)+1, 0, 2, frames_accelerate), 0.0,
                    ease_in_quad(float(frame)+1, 0, -1, frames_accelerate),
                    center=center)
            elif max_frames_rot2-frame <= frames_accelerate:
                # Slow down rotating in the end
                face_small.rotate_around_xyz(
                    ease_out_quad(
                        frames_accelerate-float(max_frames_rot2-frame)+1,
                        2, -2, frames_accelerate),
                    0.0,
                    ease_out_quad(
                        frames_accelerate-float(max_frames_rot2-frame)+1,
                        -1, 1, frames_accelerate),
                    center=center)
            else:
                # Rotation with steady speed
                face_small.rotate_around_xyz(2.0, 0.0, -1.0, center=center)

        # Rotate camera 360 degrees during the loop
        if frame > 0:
            ax.view_init(elev=elev,
                         azim=azim + (frame / float(max_frames_rot2) * 360))

        save_image(fig, gif_frames_output_folder, max_frames_rot1 + frame, 3)
        # Remove last drawn shapes
        [shape.remove() for shape in drawn_shapes]

    return gif_frames_output_folder
