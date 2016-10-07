import os

import subprocess
from sys import platform as _platform

import rotate2d


def get_imagemagick_path():
    if _platform == "linux" or _platform == "linux2":
        # Linux
        raise(Exception("I don't know where ImageMagick is installed :("))
    elif _platform == "darwin":
        # macOS (OS X)
        return os.path.join(os.sep, "opt", "local", "bin", "convert")
    elif _platform == "win32":
        # Windows
        return '"' + os.path.join('c:', os.sep, 'Program Files',
                                  'ImageMagick-7.0.3-Q16', 'convert.exe') + '"'


def create_gif(ims_input_folder, gif_output_name, delay=2):
    # Create GIF with ImageMagick
    subprocess.call(
        "{path_to_convert} -delay {delay} "
        "{ims_folder}/*png {gif_name}".format(
            path_to_convert=get_imagemagick_path(), delay=delay,
            ims_folder=ims_input_folder, gif_name=gif_output_name), shell=True)


if __name__ == '__main__':

    # Create output folder
    gif_frames_output_folder = 'gif_frames_2d'
    if not os.path.exists(gif_frames_output_folder):
        os.makedirs(gif_frames_output_folder)

    rotate2d.run_animation(ims_output_folder=gif_frames_output_folder)
    create_gif(ims_input_folder=gif_frames_output_folder,
               gif_output_name='rotating_projections_2d.gif', delay=3)
