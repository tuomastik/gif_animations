import os
import subprocess
from pprint import pprint
from sys import platform as _platform

import projections
import cubes
import mountain
import kde
import christmas


def get_imagemagick_path(binary="convert"):
    if _platform == "linux" or _platform == "linux2":
        return os.path.join(os.path.sep, "usr", "bin", binary)
    elif _platform == "darwin":  # macOS (OS X)
        return os.path.join(os.path.sep, "opt", "local", "bin", binary)
    elif _platform == "win32":  # Windows
        return '"' + os.path.join('c:', os.path.sep, 'Program Files',
                                  'ImageMagick-7.0.8-Q16', binary+".exe") + '"'


def convert_images_format(ims_folder, format_from="svg", format_to="png",
                          resolution="500x500", create_new_ims=True,
                          quality="100"):
    print("Converting %ss to %ss..." % (format_from.upper(),
                                        format_to.upper()))
    subprocess.call(
        "{path_to_mogrify} -format {format_to} "
        "-quality {quality} {resizing} *.{format_from}".format(
            path_to_mogrify=get_imagemagick_path(binary="mogrify"),
            format_to=format_to, quality=quality,
            resizing="" if resolution is None else "-{option} {res}".format(
                option="size" if create_new_ims else "resize", res=resolution),
            format_from=format_from), shell=True, cwd=ims_folder)


def create_gif(ims_input_folder, gif_output_name, delay=2, ext="png"):
    # Create GIF with ImageMagick
    print(f"Stitching together a GIF from the frames "
          "in the directory: {ims_input_folder}...")
    subprocess.call(
        "{path_to_convert} -delay {delay} "
        "{ims_folder}/*{ext} {gif_name}".format(
            path_to_convert=get_imagemagick_path(), delay=delay, ext=ext,
            ims_folder=ims_input_folder, gif_name=gif_output_name), shell=True)
    print(f"The following GIF was created successfully: {gif_output_name}")


if __name__ == '__main__':
    gifs = {module.__name__: module for module
            in [projections, cubes, mountain, kde, christmas]}

    print("Which GIF do you want to create?")
    pprint({i: g for i, g in enumerate(gifs.keys())})
    gif_id = int(input("Give the number of the GIF: "))
    gif_module = gifs[list(gifs.keys())[gif_id]]
    gif_name = gif_module.__name__

    frames_folder = gif_module.create_animation_frames()

    if gif_name == kde.__name__:
        convert_images_format(frames_folder, format_from="png", format_to="png",
                              resolution="500x500", create_new_ims=False)

    gif_creation_options = {
        projections.__name__: {"delay": 3},
        cubes.__name__: {"delay": 3},
        mountain.__name__: {"delay": 4},
        kde.__name__: {"delay": 4},
        christmas.__name__: {"delay": 2},
    }
    create_gif(ims_input_folder=frames_folder,
               gif_output_name=gif_name + ".gif",
               **gif_creation_options[gif_name])
