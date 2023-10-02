import os
import time
import sys
import multiprocessing
import itertools
import argparse
from itertools import product

import numpy as np
import tqdm
import pypianoroll as pproll
import muspy

import constants
from constants import PitchToken, DurationToken


def preprocess_midi_file(filepath, dest_dir, n_bars, resolution):

    print("Preprocessing file {}".format(filepath))

    filename = os.path.basename(filepath)
    saved_samples = 0

    # Load the file both as a pypianoroll song and a muspy song
    # (Need to load both since muspy.to_pypianoroll() is expensive)
    try:
        pproll_song = pproll.read(filepath, resolution=resolution)
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

        print("Processing combination {} of {}".format(i + 1, 
                                                       len(combinations)))

        # Process combination (called 'subsong' from now on)
        drum_track, bass_track, guitar_track = combination
        tracks = [drum_track, bass_track, guitar_track, strings_track]

        pproll_subsong = pproll.Multitrack(
            tracks=tracks,
            tempo=pproll_song.tempo,
            resolution=resolution
        )
        muspy_subsong = muspy.from_pypianoroll(pproll_subsong)

        tracks_notes = [track.notes for track in muspy_subsong.tracks]

        # Obtain length of subsong (maximum of each track's length)
        length = 0
        for notes in tracks_notes:
            track_length = max(note.end for note in notes) if notes else 0
            length = max(length, track_length)
        length += 1

        # Add timesteps until length is a multiple of resolution
        length = length if length % (4*resolution) == 0 \
            else length + (4*resolution-(length % (4*resolution)))

        tracks_content = []
        tracks_structure = []

        for notes in tracks_notes:

            # track_content: length x MAX_SIMU_TOKENS x 2
            # This is used as a basis to build the final content tensors for
            # each sequence.
            # The last dimension contains pitches and durations. int16 is enough
            # to encode small to medium duration values.
            track_content = np.zeros((length, constants.MAX_SIMU_TOKENS, 2), 
                                    np.int16)

            track_content[:, :, 0] = PitchToken.PAD.value
            track_content[:, 0, 0] = PitchToken.SOS.value
            track_content[:, :, 1] = DurationToken.PAD.value
            track_content[:, 0, 1] = DurationToken.SOS.value

            # Keeps track of how many notes have been stored in each timestep
            # (int8 imposes MAX_SIMU_TOKENS < 256)
            notes_counter = np.ones(length, dtype=np.int8)

            # Todo: np.put_along_axis?
            for note in notes:
                # Insert note in the lowest position available in the timestep

                t = note.time

                if notes_counter[t] >= constants.MAX_SIMU_TOKENS-1:
                    # Skip note if there is no more space
                    continue

                pitch = max(min(note.pitch, constants.MAX_PITCH_TOKEN), 0)
                track_content[t, notes_counter[t], 0] = pitch
                dur = max(min(note.duration, constants.MAX_DUR_TOKEN + 1), 1)
                track_content[t, notes_counter[t], 1] = dur-1
                notes_counter[t] += 1

            # Add EOS token
            t_range = np.arange(0, length)
            track_content[t_range, notes_counter, 0] = PitchToken.EOS.value
            track_content[t_range, notes_counter, 1] = DurationToken.EOS.value

            # Get track activations, a boolean tensor indicating whether notes
            # are being played in a timestep (sustain does not count)
            # (needed for graph rep.)
            activations = np.array(notes_counter-1, dtype=bool)

            tracks_content.append(track_content)
            tracks_structure.append(activations)

        # n_tracks x length x MAX_SIMU_TOKENS x 2
        subsong_content = np.stack(tracks_content, axis=0)

        # n_tracks x length
        subsong_structure = np.stack(tracks_structure, axis=0)

        # Slide window over 'subsong_content' and 'subsong_structure' along the
        # time axis (2nd dimension) with the stride of a bar
        # Todo: np.lib.stride_tricks.as_strided(song_proll)?
        for i in range(0, length-n_bars*4*resolution+1, 4*resolution):

            # Get the content and structure tensors of a single sequence
            c_tensor = subsong_content[:, i:i+n_bars*4*resolution, :]
            s_tensor = subsong_structure[:, i:i+n_bars*4*resolution]
            c_tensor = np.copy(c_tensor)
            s_tensor = np.copy(s_tensor)

            if n_bars > 1:
                # Skip sequence if it contains more than one bar of consecutive
                # silence in at least one track
                bars = s_tensor.reshape(s_tensor.shape[0], n_bars, -1)
                bars_acts = np.any(bars, axis=2)

                if 1 in np.diff(np.where(bars_acts == 0)[1]):
                    continue

                # Skip sequence if it contains one bar of complete silence
                silences = np.logical_not(np.any(bars_acts, axis=0))
                if np.any(silences):
                    continue

            else:
                # Skip if all tracks are silenced
                bar_acts = np.any(s_tensor, axis=1)
                if not np.any(bar_acts):
                    continue

            # Randomly transpose the pitches of the sequence (-5 to 6 semitones)
            # Not considering SOS, EOS or PAD tokens. Not transposing drums.
            shift = np.random.choice(np.arange(-5, 7), 1)
            cond = (c_tensor[1:, :, :, 0] != PitchToken.PAD.value) &           \
                   (c_tensor[1:, :, :, 0] != PitchToken.SOS.value) &           \
                   (c_tensor[1:, :, :, 0] != PitchToken.EOS.value)
            non_drums = c_tensor[1:, ...]
            non_drums[cond, 0] += shift
            non_drums[cond, 0] = np.clip(non_drums[cond, 0], a_min=0, 
                                         a_max=constants.MAX_PITCH_TOKEN)

            # Save sample (content and structure) to file
            sample_filepath = os.path.join(
                dest_dir, filename+str(saved_samples))
            np.savez(sample_filepath, c_tensor=c_tensor, s_tensor=s_tensor)

            saved_samples += 1


def preprocess_midi_dataset(midi_dataset_dir, preprocessed_dir, n_bars, 
                            resolution, n_files=None, n_workers=1):

    print("Starting preprocessing")
    start = time.time()

    # Visit recursively the directories inside the dataset directory
    with multiprocessing.Pool(n_workers) as pool:

        walk = os.walk(midi_dataset_dir)
        fn_gen = itertools.chain.from_iterable(
            ((os.path.join(dirpath, file), preprocessed_dir, n_bars, resolution)
                for file in files)
                for dirpath, dirs, files in walk
        )

        r = list(tqdm.tqdm(pool.starmap(preprocess_midi_file, fn_gen),
                           total=n_files))

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Preprocessing completed in (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
          .format(int(hours), int(minutes), seconds))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocesses a MIDI dataset. MIDI files can be arranged " 
            "hierarchically in subdirectories, similarly to the Lakh MIDI "
            "Dataset (lmd_matched) and the MetaMIDI Dataset."
    )
    parser.add_argument(
        'midi_dataset_dir',
        type=str, 
        help='Directory of the MIDI dataset.'
    )
    parser.add_argument(
        'preprocessed_dir',
        type=str,
        help='Directory to save the preprocessed dataset.'
    )
    parser.add_argument(
        '--n_bars',
        type=int,
        default=2,
        help="Number of bars for each sequence of the resulting preprocessed "
            "dataset. Defaults to 2 bars."
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=8,
        help="Number of timesteps per beat. When set to r, given that only "
            "4/4 songs are preprocessed, there will be 4*r timesteps in a bar. "
            "Defaults to 8."
    )
    parser.add_argument(
        '--n_files',
        type=int,
        help="Number of files in the MIDI dataset. If set, the script "
            "will provide statistics on the time remaining."
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help="Number of parallel workers. Defaults to 1."
    )

    args = parser.parse_args()
    
    # Create the output directory if it does not exist
    if not os.path.exists(args.preprocessed_dir):
        os.makedirs(args.preprocessed_dir)

    preprocess_midi_dataset(args.midi_dataset_dir, args.preprocessed_dir, 
                            args.n_bars, args.resolution, args.n_files,
                            n_workers=args.n_workers)
