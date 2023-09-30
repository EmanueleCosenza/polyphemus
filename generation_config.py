import yaml
from constants import DEFAULT_MIDI_PROGRAMS, DEFAULT_SOUNDFONT_PATH


CONFIG_FILENAME = 'generation_config.yaml'


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


config = load_config(CONFIG_FILENAME)


MIDI_PROGRAMS = config.get('MIDI_PROGRAMS', DEFAULT_MIDI_PROGRAMS)
SOUNDFONT_PATH = config.get('SOUNDFONT_PATH', DEFAULT_SOUNDFONT_PATH)