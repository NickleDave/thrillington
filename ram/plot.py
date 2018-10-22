import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


def _denormalize(locations, img_height, img_width):
    """helper function that converts locations from
    normalized co-ordinates"""
    for glimpse in range(locations.shape[0]):
        locations[glimpse, :, 0] = ((locations[glimpse, :, 0] + 1) / 2) * img_height
        locations[glimpse, :, 1] = ((locations[glimpse, :, 1] + 1) / 2) * img_width
    return locations


def bounding_box(x, y, x_size, y_size, color='w'):
    x = int(x - (x_size / 2))
    y = int(y - (y_size / 2))
    rect = patches.Rectangle(
        (x, y), x_size, y_size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def plot(epoch, patch_size=(8, 8), glimpses_filename=None, locations_filename=None,
         images_filename=None, examples_dir=None, plot_dir=None):
    """plot glimpses + locations"""
    if glimpses_filename is None:
        glimpses_filename = f"glimpses_epoch_{epoch}"
    if locations_filename is None:
        locations_filename = f"locations_epoch_{epoch}"
    if images_filename is None:
        images_filename = f"images_epoch_{epoch}"

    if examples_dir:
        glimpses_filename = os.path.join(examples_dir, glimpses_filename)
        locations_filename = os.path.join(examples_dir, locations_filename)
        images_filename = os.path.join(examples_dir, images_filename)

    glimpses = np.load(glimpses_filename)
    locations = np.load(locations_filename)
    images = np.load(images_filename)

    # grab useful params
    num_anims = locations.shape[0]
    num_cols = images.shape[0]

    # denormalize coordinates
    coords = _denormalize(locations=locations,
                          img_height=images.shape[1],
                          img_width=images.shape[2])

    figure, axes = plt.subplots(nrows=1, ncols=num_cols)

    # plot base image
    for j, ax in enumerate(axes.flat):
        ax.imshow(images[j].squeeze(), cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def update_data(i):
        color = 'r'
        co = coords[i]
        for j, ax in enumerate(axes.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(
                c[0], c[1], patch_size[0], patch_size[1], color
            )
            ax.add_patch(rect)

    # animate
    anim = animation.FuncAnimation(
        figure, update_data, frames=num_anims, interval=500, repeat=True
    )

    # save as mp4
    name = 'epoch_{}.mp4'.format(epoch)
    if plot_dir:
        name = os.path.join(plot_dir, name)
    anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])

