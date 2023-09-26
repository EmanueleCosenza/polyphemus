import copy
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import muspy

from constants import PitchToken, DurationToken
import constants
import config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Builds multitrack pianoroll (mtp) from content tensor containing logits and
# structure binary tensor
def mtp_from_logits(c_logits, s_tensor):

    mtp = torch.zeros((s_tensor.size(0), s_tensor.size(1), s_tensor.size(2),
                       s_tensor.size(3), c_logits.size(-2), c_logits.size(-1)),
                      device=c_logits.device, dtype=c_logits.dtype)

    size = mtp.size()
    mtp = mtp.view(-1, mtp.size(-2), mtp.size(-1))
    silence = torch.zeros((mtp.size(-2), mtp.size(-1)),
                          device=c_logits.device, dtype=c_logits.dtype)

    # Create silences with pitch EOS and PAD tokens
    silence[0, PitchToken.EOS.value] = 1.
    silence[1:, PitchToken.PAD.value] = 1.

    # Fill the multitrack pianoroll
    mtp[s_tensor.bool().view(-1)] = c_logits
    mtp[torch.logical_not(s_tensor.bool().view(-1))] = silence
    mtp = mtp.view(size)

    return mtp


def muspy_from_mtp(mtp):

    n_timesteps = mtp.size(2)
    resolution = n_timesteps // 4

    # Collapse bars dimension
    mtp = mtp.permute(1, 0, 2, 3, 4)
    size = (mtp.shape[0], -1, mtp.shape[3], mtp.shape[4])
    mtp = mtp.reshape(*size)

    tracks = []

    for track_idx in range(mtp.size(0)):

        notes = []

        for t in range(mtp.size(1)):
            for note_idx in range(mtp.size(2)):

                # Compute pitch and duration values
                pitch = mtp[track_idx, t, note_idx, :constants.N_PITCH_TOKENS]
                dur = mtp[track_idx, t, note_idx, constants.N_PITCH_TOKENS:]
                pitch, dur = torch.argmax(pitch), torch.argmax(dur)

                if (pitch == PitchToken.EOS.value or
                    pitch == PitchToken.PAD.value or
                    dur == DurationToken.EOS.value or
                        dur == DurationToken.PAD.value):
                    # This chord contains no additional notes, go to next chord
                    break

                if (pitch == PitchToken.SOS.value or
                        pitch == PitchToken.SOS.value):
                    # Skip this note
                    continue

                # Remapping duration values from [0, 95] to [1, 96]
                dur = dur + 1
                # Do not sustain notes beyond sequence limit
                dur = min(dur.item(), mtp.size(1) - t)

                notes.append(muspy.Note(t, pitch.item(), dur, 64))

        track_name = constants.TRACKS[track_idx]
        midi_program = config.MIDI_PROGRAMS[track_name]
        is_drum = (track_name == 'Drums')

        track = muspy.Track(
            name=track_name,
            is_drum=is_drum,
            program=(0 if is_drum else midi_program),
            notes=copy.deepcopy(notes)
        )
        tracks.append(track)

    meta = muspy.Metadata()
    music = muspy.Music(tracks=tracks, metadata=meta, resolution=resolution)

    return music


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


def save_midi(muspy_song, save_dir, name):
    muspy.write_midi(os.path.join(save_dir, name + ".mid"), muspy_song)


def save_audio(muspy_song, save_dir, name):
    soundfont_path = (config.SOUNDFONT_PATH 
                      if os.path.exists(config.SOUNDFONT_PATH) else None)
    muspy.write_audio(os.path.join(save_dir, name + ".wav"), muspy_song,
                      soundfont_path=soundfont_path)