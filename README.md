# polyphemus

A graph-based deep MIDI music generator. Official repository of the [paper](https://arxiv.org/abs/2307.14928) "Graph-based Polyphonic Multitrack Music Generation".

You can listen to the samples generated in the experimental phase [here](https://emanuelecosenza.github.io/polyphemus/).


## Prerequisites

To run polyphemus, follow these steps:

1. **Create an Environment:**
   Use conda to create a Python 3.7 environment:
   ```sh
   conda create -n polyphemus-env python=3.7
   ```
   An alternative is to install Python 3.7 and create a new virtual environment with `venv`.
   
2. **Install the Required Python Packages:**
   Navigate to the directory where the repository is cloned and install the required packages from `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare Your Environment for Audio Generation (Optional):**
   If you intend to generate audio from MIDI files directly with the provided scripts, you will need a sound synthesizer like [`fluidsynth`](https://github.com/FluidSynth/fluidsynth/wiki), and a [SoundFont](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont) file that tells `fluidsynth` what are the audio samples to play for each MIDI instrument. You can get both by doing the following:
   ```sh
   sudo apt-get install fluidsynth fluid-soundfont-gm
   ```
   where `fluid-soundfont-gm` is the Fluid (R3) General MIDI SoundFont, a SoundFont that is known to work well with `fluidsynth`. By default, after the installation, the SoundFont file will be found in `/usr/share/sounds/sf2`.
   
   If you don't have sudo access to your machine, `fluidsynth` can also be installed through `conda` as follows:
   ```sh
   conda install fluidsynth
   ```
   You can also get the Fluid SoundFont and a bunch of other SoundFonts by following the links provided in the [FluidSynth project Wiki](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont).

4. **Download the Trained Models:**
   Run the `download_models.py` script to download the models trained on the LMD dataset.
   ```python
   python download_models.py models/
   ```
   The script will download LMD2, a model that generates short 2-bar sequences of music, and LMD16, which generates longer 16-bar sequences. The models will be stored in the `models/` directory. The two models have been trained on 4/4 music. For them, the tracks (=instruments) in each sequence are 4 (in order, drums, bass, guitar and strings), while the number of timesteps for each bar is 32, which means that each timestep has rhythmic value 1/32.

## Usage

### Generation

If you have already installed polyphemus and downloaded the pretrained models, to generate music, you can just run the `generate.py` script as follows:
```sh
python generate.py models/LMD2/ music/ --n 10 --n_loops 4
```
Using this command, the LMD2 model located in `models/LMD2/` will generate 10 sequences and the results will be saved in `music/`. The `--n_loops` argument tells the application to output, for each sequence, an additional extended sequence obtained by looping the generated one 4 times (this is advisable for 2-bar sequences, since they are quite short).

By default, `generate.py` outputs `.wav` audio files in addition to MIDI files. You can change the way audio is generated from MIDI by editing the `generation_config.yaml` file. Here, you can specify the SoundFont file that has to be used while generating audio and the MIDI programs that have to be associated to each track. Run `generate.py` with the `--help` flag to get a complete list of all the arguments you can pass to the script.

#### Customizing Track Activations

With Polyphemus, users can specify which tracks (i.e. instruments) should be activated at specific timesteps by editing the `structure.json` file and by passing the `--s_file` argument to `generate.py` as follows:
```sh
python generate.py models/LMD2/ music/ --n 10 --s_file structure.json
```

The `structure.json` file can be interpreted as a sort of pianoroll for instrument activations. It contains a JSON representation of a `[n_bars, n_tracks, n_timesteps]` binary tensor. To give a concrete example, consider the following JSON file:
```
[
    # Bar 1
    [
        [0, 0, 0, 0,  0, 0, 0, 0], # Track 1
        [1, 0, 0, 0,  0, 0, 0, 0], # Track 2
        [0, 0, 0, 0,  0, 0, 0, 0], # Track 3
        [0, 0, 0, 0,  1, 0, 0, 0]  # Track 4
        
    ],
    # Bar 2
    [
        [0, 0, 0, 0,  0, 0, 0, 0], # Track 1
        [0, 0, 0, 0,  1, 0, 0, 0], # Track 2
        [0, 0, 0, 0,  0, 0, 0, 0], # Track 3
        [0, 0, 0, 0,  0, 0, 0, 0]  # Track 4
        
    ]
]
```
Here, for simplicity, `n_timesteps=8`. In the first bar, Track 2 and Track 4 will be activated, respectively, in the first and fifth timestep. In the second bar, just the second track will be activated in the fifth timestep.

Notice that the binary activations above refer to actual note activations and that sustain is not considered. This means that a track could end up playing a sustained note even if the corresponding value has been set to 0 in the file. In the example above, Track 2, which has been activated in the first timestep of the sequence, could actually sustain its activated notes for more than a timestep.
When editing this file for the LMD2 and LMD16 models, remember that `n_timesteps=32`, which means that each timestep has rhythmic value 1/32, and that the tracks used to train these models are, in order, drums, bass, guitar and strings.

The number of tracks for each bar in the JSON file must be strictly equal to the number of tracks used to train the model. The number of bars, instead, may be smaller than the number of bars that can be handled by the model. If this is the case, the script will just replicate the partial binary structure to reach `n_bars`. In this way, it is possible e.g. to compile the binary activations for just one bar and let the model use this same structure for each bar.

### Training

You can train a new model from scratch by using the dedicated script:
```sh
python train.py
```
Run the command with the `--help` flag to find out how you can customize the training procedure.

## License

This project is licensed under the MIT License.
