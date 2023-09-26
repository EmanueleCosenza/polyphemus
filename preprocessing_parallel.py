import os
import time
import sys
import multiprocessing
import itertools
from itertools import product

import numpy as np
import tqdm
import pypianoroll as pproll
import muspy


# Todo: to config file (or separate files)
MAX_SIMU_NOTES = 16  # 14 + SOS and EOS

PITCH_SOS = 128
PITCH_EOS = 129
PITCH_PAD = 130
DUR_SOS = 96
DUR_EOS = 97
DUR_PAD = 98

MAX_DUR = 96

# Number of time steps per quarter note
# To get bar resolution -> RESOLUTION*4
RESOLUTION = 8
NUM_BARS = 2

num_files = 612090
dest_dir = "/data/cosenza/datasets/MMD/preprocessed_2bars_par"


def preprocess_file(filepath):

    print("Preprocessing file " + filepath)

    filename = os.path.basename(filepath)
    saved_samples = 0

    # Load the file both as a pypianoroll song and a muspy song
    # (Need to load both since muspy.to_pypianoroll() is expensive)
    try:
        pproll_song = pproll.read(filepath, resolution=RESOLUTION)
        muspy_song = muspy.read(filepath)
    except Exception as e:
        print("Song skipped (Invalid song format)")
        return 0

    # Only accept songs that have a time signature of 4/4 and no time changes
    for t in muspy_song.time_signatures:
        if t.numerator != 4 or t.denominator != 4:
            print("Song skipped ({}/{} time signature)".
                  format(t.numerator, t.denominator))
            return 0

    # Gather tracks of pypianoroll song based on MIDI program number
    drum_tracks = []
    bass_tracks = []
    guitar_tracks = []
    strings_tracks = []

    for track in pproll_song.tracks:
        if track.is_drum:
            track.name = 'Drums'
            drum_tracks.append(track)
        elif 0 <= track.program <= 31:
            track.name = 'Guitar'
            guitar_tracks.append(track)
        elif 32 <= track.program <= 39:
            track.name = 'Bass'
            bass_tracks.append(track)
        else:
            # Tracks with program > 39 are all considered as strings tracks
            # and will be merged into a single track later on
            strings_tracks.append(track)

    # Filter song if it does not contain drum, guitar, bass or strings tracks
    # if not guitar_tracks \
    if not drum_tracks or not guitar_tracks \
            or not bass_tracks or not strings_tracks:
        print("Song skipped (does not contain drum or "
              "guitar or bass or strings tracks)")
        return 0

    # Merge strings tracks into a single pypianoroll track
    strings = pproll.Multitrack(tracks=strings_tracks)
    strings_track = pproll.Track(pianoroll=strings.blend(mode='max'),
                                 program=48, name='Strings')

    combinations = list(product(drum_tracks, bass_tracks, guitar_tracks))

    # Single instruments can have multiple tracks.
    # Consider all possible combinations of drum, bass, and guitar tracks
    for i, combination in enumerate(combinations):

        print("Processing combination", i+1, "of", len(combinations))

        # Process combination (called 'subsong' from now on)
        drum_track, bass_track, guitar_track = combination
        tracks = [drum_track, bass_track, guitar_track, strings_track]

        pproll_subsong = pproll.Multitrack(
            tracks=tracks,
            tempo=pproll_song.tempo,
            resolution=RESOLUTION
        )
        muspy_subsong = muspy.from_pypianoroll(pproll_subsong)

        tracks_notes = [track.notes for track in muspy_subsong.tracks]

        # Obtain length of subsong (maximum of each track's length)
        length = 0
        for notes in tracks_notes:
            track_length = max(note.end for note in notes) if notes else 0
            length = max(length, track_length)
        length += 1

        # Add timesteps until length is a multiple of RESOLUTION
        length = length if length % (RESOLUTION*4) == 0 \
            else length + (RESOLUTION*4-(length % (RESOLUTION*4)))

        tracks_tensors = []
        tracks_activations = []

        # Todo: adapt to velocity
        for notes in tracks_notes:

            # Initialize encoder-ready track tensor
            # track_tensor: (length x max_simu_notes x 2 (or 3 if velocity))
            # The last dimension contains pitches and durations (and velocities)
            # int16 is enough for small to medium duration values
            track_tensor = np.zeros((length, MAX_SIMU_NOTES, 2), np.int16)

            track_tensor[:, :, 0] = PITCH_PAD
            track_tensor[:, 0, 0] = PITCH_SOS
            track_tensor[:, :, 1] = DUR_PAD
            track_tensor[:, 0, 1] = DUR_SOS

            # Keeps track of how many notes have been stored in each timestep
            # (int8 imposes MAX_SIMU_NOTES < 256)
            notes_counter = np.ones(length, dtype=np.int8)

            # Todo: np.put_along_axis?
            for note in notes:
                # Insert note in the lowest position available in the timestep

                t = note.time

                if notes_counter[t] >= MAX_SIMU_NOTES-1:
                    # Skip note if there is no more space
                    continue

                pitch = max(min(note.pitch, 127), 0)
                track_tensor[t, notes_counter[t], 0] = pitch
                dur = max(min(MAX_DUR, note.duration), 1)
                track_tensor[t, notes_counter[t], 1] = dur-1
                notes_counter[t] += 1

            # Add EOS token
            track_tensor[np.arange(0, length), notes_counter, 0] = PITCH_EOS
            track_tensor[np.arange(0, length), notes_counter, 1] = DUR_EOS

            # Get track activations, a boolean tensor indicating whether notes
            # are being played in a timestep (sustain does not count)
            # (needed for graph rep.)
            activations = np.array(notes_counter-1, dtype=bool)

            tracks_tensors.append(track_tensor)
            tracks_activations.append(activations)

        # (#tracks x length x max_simu_notes x 2 (or 3))
        subsong_tensor = np.stack(tracks_tensors, axis=0)

        # (#tracks x length)
        subsong_activations = np.stack(tracks_activations, axis=0)

        # Slide window over 'subsong_tensor' and 'subsong_activations' along the
        # time axis (2nd dimension) with the stride of a bar
        # Todo: np.lib.stride_tricks.as_strided(song_proll)
        for i in range(0, length-NUM_BARS*RESOLUTION*4+1, RESOLUTION*4):

            # Get the sequence and its activations
            seq_tensor = subsong_tensor[:, i:i+NUM_BARS*RESOLUTION*4, :]
            seq_acts = subsong_activations[:, i:i+NUM_BARS*RESOLUTION*4]
            seq_tensor = np.copy(seq_tensor)
            seq_acts = np.copy(seq_acts)

            if NUM_BARS > 1:
                # Skip sequence if it contains more than one bar of consecutive
                # silence in at least one track
                bars = seq_acts.reshape(seq_acts.shape[0], NUM_BARS, -1)
                bars_acts = np.any(bars, axis=2)

                if 1 in np.diff(np.where(bars_acts == 0)[1]):
                    continue

                # Skip sequence if it contains one bar of complete silence
                # (in terms of note activations)
                silences = np.logical_not(np.any(bars_acts, axis=0))
                if np.any(silences):
                    continue

            else:
                # In the case of just 1 bar, skip it if all tracks are silenced
                bar_acts = np.any(seq_acts, axis=1)
                if not np.any(bar_acts):
                    continue

            # Randomly transpose the pitches of the sequence (-5 to 6 semitones)
            # Not considering pad, sos, eos tokens,
            # Not transposing drums/percussions
            shift = np.random.choice(np.arange(-5, 7), 1)
            cond = (seq_tensor[1:, :, :, 0] != PITCH_PAD) &                    \
                   (seq_tensor[1:, :, :, 0] != PITCH_SOS) &                    \
                   (seq_tensor[1:, :, :, 0] != PITCH_EOS)
            non_perc = seq_tensor[1:, ...]
            non_perc[cond, 0] += shift
            non_perc[cond, 0] = np.clip(non_perc[cond, 0], a_min=0, a_max=127)

            # Save sample (seq_tensor and seq_acts) to file
            sample_filepath = os.path.join(
                dest_dir, filename+str(saved_samples))
            np.savez(sample_filepath, seq_tensor=seq_tensor, seq_acts=seq_acts)

            saved_samples += 1


# Total number of files (LMD): 116189
# Number of unique files (LMD): 45129
def preprocess_dataset(dataset_dir, dest_dir, workers=1,
                       num_files=612090, early_exit=None):

    print("Starting preprocessing")
    start = time.time()

    with multiprocessing.Pool(workers) as pool:

        walk = os.walk(dataset_dir)
        fn_gen = itertools.chain.from_iterable((os.path.join(dirpath, file)
                                                for file in files)
                                               for dirpath, dirs, files in walk)

        r = list(tqdm.tqdm(pool.imap(preprocess_file, fn_gen), total=num_files))

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Preprocessing completed in (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
          .format(int(hours), int(minutes), seconds))


if __name__ == "__main__":

    workers = 48

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        dest_dir = sys.argv[2]

    preprocess_dataset(dataset_dir, dest_dir, workers=workers)
