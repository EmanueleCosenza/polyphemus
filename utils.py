import copy
import os
import random

import numpy as np
import torch
import muspy
from prettytable import PrettyTable

from constants import PitchToken, DurationToken
import constants
import generation_config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def append_dict(dest_d, source_d):

    for k, v in source_d.items():
        dest_d[k].append(v)


def print_params(model):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Parameters: {total_params}")

    return total_params


def print_divider():
    print('—' * 40)


# Builds multitrack pianoroll (mtp) from content tensor containing logits and
# structure binary tensor
# c_logits: num_nodes x MAX_SIMU_TOKENS x d_token
# s_tensor: n_batches x n_bars x n_tracks x n_timesteps
def mtp_from_logits(c_logits, s_tensor):

    mtp = torch.zeros((s_tensor.size(0), s_tensor.size(1), s_tensor.size(2),
                       s_tensor.size(3), c_logits.size(-2), c_logits.size(-1)),
                      device=c_logits.device, dtype=c_logits.dtype)

    size = mtp.size()
    mtp = mtp.reshape(-1, mtp.size(-2), mtp.size(-1))
    silence = torch.zeros((mtp.size(-2), mtp.size(-1)),
                          device=c_logits.device, dtype=c_logits.dtype)

    # Create silences with pitch EOS and PAD tokens
    silence[0, PitchToken.EOS.value] = 1.
    silence[1:, PitchToken.PAD.value] = 1.

    # Fill the multitrack pianoroll
    mtp[s_tensor.bool().reshape(-1)] = c_logits
    mtp[torch.logical_not(s_tensor.bool().reshape(-1))] = silence
    mtp = mtp.reshape(size)

    return mtp


# mtp: n_bars x n_tracks x n_timesteps x MAX_SIMU_TOKENS x d_token
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
                    # The chord contains no additional notes, go to next chord
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
        midi_program = generation_config.MIDI_PROGRAMS[track_name]
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


def loop_muspy_music(muspy_music, n_loop, num_bars, resolution):
    
    # Get a deep copy of the original music object to avoid modifying it
    looped_music = copy.deepcopy(muspy_music)
    
    # Loop over the number of times we want to repeat the sequence
    for i in range(1, n_loop):
        # Loop over each track in the original music object
        for track_idx, track in enumerate(muspy_music.tracks):
            # Adjust the start times of the notes for each repetition and 
            # add them to the corresponding track in the looped_music object
            for note in track.notes:
                new_note = copy.deepcopy(note)
                new_note.time += i * num_bars * 4 * resolution
                looped_music.tracks[track_idx].notes.append(new_note)
                
    return looped_music



def save_midi(muspy_song, save_dir, name):
    muspy.write_midi(os.path.join(save_dir, name + ".mid"), muspy_song)


def save_audio(muspy_song, save_dir, name):
    soundfont_path = (generation_config.SOUNDFONT_PATH
                      if os.path.exists(generation_config.SOUNDFONT_PATH) 
                      else None)
    muspy.write_audio(os.path.join(save_dir, name + ".wav"), muspy_song,
                      soundfont_path=soundfont_path)
