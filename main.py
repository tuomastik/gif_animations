import os

import subprocess
from sys import platform as _platform

import rotating_projections
import rotating_cubes
import rotating_mountain


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
    print("Creating gif...")
    subprocess.call(
        "{path_to_convert} -delay {delay} "
        "{ims_folder}/*png {gif_name}".format(
            path_to_convert=get_imagemagick_path(), delay=delay,
            ims_folder=ims_input_folder, gif_name=gif_output_name), shell=True)


if __name__ == '__main__':
    frames_folder = rotating_projections.create_animation_frames()
    create_gif(ims_input_folder=frames_folder,
               gif_output_name='rotating_projections.gif', delay=3)

    frames_folder = rotating_cubes.create_animation_frames()
    create_gif(ims_input_folder=frames_folder,
               gif_output_name='rotating_cubes.gif', delay=3)

    frames_folder = rotating_mountain.create_animation_frames()
    create_gif(ims_input_folder=frames_folder,
               gif_output_name='rotating_mountain.gif', delay=4)
