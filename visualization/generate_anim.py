import os

import imageio
from moviepy.editor import ImageSequenceClip


def generate_gif(folder, name):
    # List PNG files
    png_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    png_files.sort()  # Sort the files if necessary

    # Create GIF file name
    gif_file = f'{folder}/{name}.gif'

    # Create GIF from PNGs
    with imageio.get_writer(gif_file, mode='I', duration=0.01) as writer:  # Set duration as needed
        for png_file in png_files:
            image_path = os.path.join(folder, png_file)
            with imageio.get_reader(image_path) as reader:
                for frame in reader:
                    writer.append_data(frame)

    print(f'GIF created: {gif_file}')


def generate_mp4(folder, name):
    # List PNG files
    png_files = [f for f in os.listdir(folder) if f.endswith('.png') if int(f[0]) < 2]
    png_files.sort()  # Sort the files if necessary

    # Create list of image paths
    image_paths = [os.path.join(folder, png_file) for png_file in png_files]

    # Create video clip from PNG images
    clip = ImageSequenceClip(image_paths, fps=100)  # Set frames per second (fps) as needed

    # Output file name
    output_file = f'{folder}/{name}.mp4'

    # Write the video file
    clip.write_videofile(output_file, codec='libx264')

    print(f'Video clip created: {output_file}')


if __name__ == '__main__':
    who = 'full'
    path = os.path.join(os.path.dirname(__file__), f"../FINAL/images/{who}")
    generate_mp4(path, name=who)

