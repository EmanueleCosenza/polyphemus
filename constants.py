from enum import Enum


class PitchToken(Enum):
    SOS = 128
    EOS = 129
    PAD = 130

N_PITCH_TOKENS = 131


class DurationToken(Enum):
    SOS = 96
    EOS = 97
    PAD = 98
    
N_DUR_TOKENS = 99


TRACKS = ['Drums', 'Bass', 'Guitar', 'Strings']
MIDI_PROGRAMS = {
    'Drums': -1,
    'Bass': 34,
    'Guitar': 1,
    'Strings': 83,
}