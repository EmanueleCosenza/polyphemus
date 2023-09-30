from enum import Enum


N_TRACKS = 4
TRACKS = ['Drums', 'Bass', 'Guitar', 'Strings']

# These defaults, which should not be changed, are set when the related 
# variables in the YAML config file are not set. 
DEFAULT_MIDI_PROGRAMS = {
    'Drums': -1,
    'Bass': 34,
    'Guitar': 1,
    'Strings': 83,
}
DEFAULT_SOUNDFONT_PATH = '/usr/share/soundfonts/FluidR3_GM.sf2'


# Pitch tokens have values in the range [0, 130]. Tokens from 0 to 127 represent
# MIDI pitches. Token 60 represents middle C (C4).
# See https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
# for the complete list of MIDI pitches.
class PitchToken(Enum):
    SOS = 128
    EOS = 129
    PAD = 130


N_PITCH_TOKENS = 131
MAX_PITCH_TOKEN = 127


# Duration tokens have values in the range [0, 98]. Tokens from 0 to 95 have to
# be interpreted as durations from 1 to 96 timesteps.
class DurationToken(Enum):
    SOS = 96
    EOS = 97
    PAD = 98


N_DUR_TOKENS = 99
MAX_DUR_TOKEN = 95


# Dimension of (pitch, duration) token pair
D_TOKEN_PAIR = N_PITCH_TOKENS + N_DUR_TOKENS

# Number of maximum tokens stored in each timestep (14 + SOS and EOS)
MAX_SIMU_TOKENS = 16


# This enum contains edge type indices for each edge type
class EdgeTypes(Enum):
    TRACK = 0 # This has to be interpreted as the starting index
    ONSET = N_TRACKS
    NEXT = N_TRACKS + 1

# N_TRACKS track types + 1 onset edge type + 1 next edge type
N_EDGE_TYPES = N_TRACKS + 2


