import os

from matplotlib import pyplot as plt
import matplotlib as mpl
import muspy

import constants


def plot_pianoroll(muspy_song, save_dir=None, name='pianoroll'):

    lines_linewidth = 4
    axes_linewidth = 4
    font_size = 34
    fformat = 'png'
    xticklabel = False
    label = 'y'
    figsize = (20, 10)
    dpi = 200

    with mpl.rc_context({'lines.linewidth': lines_linewidth,
                         'axes.linewidth': axes_linewidth,
                         'font.size': font_size}):

        fig, axs_ = plt.subplots(constants.N_TRACKS, sharex=True,
                                 figsize=figsize)
        fig.subplots_adjust(hspace=0)
        axs = axs_.tolist()
        muspy.show_pianoroll(music=muspy_song, yticklabel='off', xtick='off',
                             label=label, xticklabel=xticklabel,
                             grid_axis='off', axs=axs, preset='full')

        if save_dir:
            plt.savefig(os.path.join(save_dir, name + "." + fformat),
                        format=fformat, dpi=dpi)


def plot_structure(s_tensor, save_dir=None, name='structure'):

    lines_linewidth = 1
    axes_linewidth = 1
    font_size = 14
    fformat = 'svg'
    dpi = 200

    n_bars = s_tensor.shape[0]
    figsize = (3 * n_bars, 3)

    n_timesteps = s_tensor.size(2)
    resolution = n_timesteps // 4
    s_tensor = s_tensor.permute(1, 0, 2)
    s_tensor = s_tensor.reshape(s_tensor.shape[0], -1)

    with mpl.rc_context({'lines.linewidth': lines_linewidth,
                         'axes.linewidth': axes_linewidth,
                         'font.size': font_size}):

        plt.figure(figsize=figsize)
        plt.pcolormesh(s_tensor, edgecolors='k', linewidth=1)
        ax = plt.gca()

        plt.xticks(range(0, s_tensor.shape[1], resolution),
                   range(1, 4*n_bars + 1))
        plt.yticks(range(0, s_tensor.shape[0]), constants.TRACKS)

        ax.invert_yaxis()

        if save_dir:
            plt.savefig(os.path.join(save_dir, name + "." + fformat),
                        format=fformat, dpi=dpi)