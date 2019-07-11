import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import style

from .utils import denormalize_img


def bounding_box(x, y, x_size, y_size, color='w'):
    x = int(x - (x_size / 2))
    y = int(y - (y_size / 2))
    rect = patches.Rectangle(
        (x, y), x_size, y_size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def examples(fixations, images, patch_sizes,
             predictions=None, glimpses=None, save_as=None,
             denormalize_imgs=False, class_to_plot=None,
             cmap='Greys'):
    """make animation of examples with fixations shown as a bounding box on images, and if provided,
    actual glimpses seen by network, as well as predictions output after last glimpse

    Parameters
    ----------
    fixations : numpy.ndarray
        containing fixations of RAM
    images : numpy.ndarray
        containing images shown to RAM
    patch_sizes : list
        of patch sizes
    predictions : numpy.ndarray
        containing predictions made by RAM
    glimpses : numpy.ndarray
        containing glimpses extracted by RAM
    save_as : str
        path and basename to use when saving animation. Default is None, in which case animation is not saved.
    denormalize_imgs : bool
        if True, convert images back to RGB from normalized (mean 0, divided by standard deviation)

    Returns
    -------
    None
    """
    if denormalize_imgs:
        images = denormalize_img(images)
        if glimpses is not None:
            glimpses = denormalize_img(glimpses)

    # grab useful params
    # one frame for each 'glimpse' / fixation
    num_frames = fixations.shape[1]
    num_cols = images.shape[0]

    if glimpses is not None:
        glimpse_rows = len(patch_sizes)
        num_rows = glimpse_rows + 1  # + 1 is top row with input images
    else:
        num_rows = 1

    figure, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    # used by ArtistAnimation, list of lists (of artists per each frame)
    artists_list = []

    for frame in range(num_frames):
        artists_this_frame = []

        for j, ax in enumerate(axes[0, :].flat):
            im = ax.imshow(images[j].squeeze(), animated=True, cmap=cmap)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            artists_this_frame.append(im)

        for j, ax in enumerate(axes[0, :].flat):
            coords = fixations[j, frame]
            for patch_size in patch_sizes:
                rect = bounding_box(
                    coords[0], coords[1],
                    patch_size[0], patch_size[1],
                    color='blue'
                )
                ax.add_patch(rect)
                artists_this_frame.append(rect)

        if frame == num_frames - 1:  # if last frame
            if predictions is not None:  # then plot them
                for j, ax in enumerate(axes[0, :].flat):
                    text = ax.set_xlabel(predictions[j])
                    artists_this_frame.append(text)

        if glimpses is not None:
            for glimpse_patch in range(len(patch_sizes)):
                row = glimpse_patch + 1
                for j, ax in enumerate(axes[row, :].flat):
                    im = ax.imshow(glimpses[j, frame, glimpse_patch].squeeze(),
                                   animated=True)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    artists_this_frame.append(im)

        artists_list.append(artists_this_frame)

    # animate
    anim = animation.ArtistAnimation(
        figure, artists_list, interval=500, blit=True,
        repeat=True, repeat_delay=1000
    )

    # save as mp4
    if save_as:
        anim.save(save_as)


def discrete_cmap(N=6, base_cmap='Greens'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N+1))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def behavior(examples_dir, suffix,
             num_samples_per_batch=500, num_mc_episodes=10,
             plot_same_sample_diff_eps=False,
             which_sample=None,
             suptitle=None, save_as=None):
    """plot 'behavior' of model"""
    style.use('dark_background')

    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.labelweight'] = 'regular'
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 48

    fixation_cmap = discrete_cmap()
    # get data
    fix_path = os.path.join(examples_dir, f'fixations{suffix}.npy')
    fix = np.load(fix_path)

    images_path = os.path.join(examples_dir, f'images{suffix}.npy')
    images = np.load(images_path)

    true_lbl_path = os.path.join(examples_dir, f'labels{suffix}.npy')
    true_lbl = np.load(true_lbl_path)

    num_examples = fix.shape[0]

    # make dictionary where key is digit
    # and value is fixations for all instances of that digit
    fixes_by_digit = {}
    digits = range(10)
    for digit in digits:
        digit_inds = np.where(true_lbl == digit)[0]
        if plot_same_sample_diff_eps:
            specific_sample_ind = digit_inds[which_sample]
            that_sample_all_episodes = [specific_sample_ind + (num_samples_per_batch * episode)
                                        for episode in range(num_mc_episodes)]
            fixes_by_digit[digit] = fix[that_sample_all_episodes]
        else:
            fixes_by_digit[digit] = fix[digit_inds]

    fig, ax = plt.subplots(2, 5)
    fig.set_size_inches(25, 15)
    ax = ax.ravel()

    for digit in digits:
        fixes_this_digit = fixes_by_digit[digit]
        xx = fixes_this_digit[:, :, 0]
        yy = fixes_this_digit[:, :, 1]
        for x, y in zip(xx, yy):
            ax[digit].plot(x, y, color='gray', linewidth=0.5)
        xmean = xx.mean(axis=0)
        ymean = yy.mean(axis=0)
        xy = np.stack([xmean, ymean], axis=1)
        for ind, (start, stop) in enumerate(zip(xy[:-1], xy[1:])):
            x, y = zip(start, stop)
            ax[digit].plot(x, y, color=fixation_cmap(ind),
                           linewidth=4, linestyle='--', marker='o')
        ax[digit].set_xlim([0, 28])
        ax[digit].set_ylim([0, 28])
        ax[digit].set_xlabel(f'digit = {digit}')

    if suptitle:
        fig.suptitle(suptitle)

    if save_as:
        plt.savefig(save_as)
