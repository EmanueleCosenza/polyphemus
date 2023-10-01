# polyphemus

A graph-based deep MIDI music generator. Official repository of the [paper](https://arxiv.org/abs/2307.14928) "Graph-based Polyphonic Multitrack Music Generation".

You can listen to the samples generated in the experimental phase [here](https://emanuelecosenza.github.io/polyphemus/).


## Prerequisites

To run polyphemus, ensure that you have Python 3.7 installed on your system and clone this repository.

1. **Create a Virtual Environment:**
   ```sh
   python -m venv polyphemus-env
   ```
   ```sh
   source polyphemus-env/bin/activate  # On Windows, use `polyphemus-env\Scripts\activate`
   ```
   
2. **Install Required Packages:**
   Navigate to the directory where the repository is cloned and install the required packages from `requirements.txt`.
   ```sh
   pip install -r requirements.txt
   ```

3. **Install Additional Packages for Audio Generation:**
   If you intend to generate audio from MIDI files, install the following additional packages.
   ```sh
   sudo apt-get install fluidsynth fluid-soundfont-gm fluid-soundfont-gs
   ```

4. **Download Trained Models:**
   Run the `download.py` script to download models trained on the LMD dataset.
   ```python
   python download.py
   ```
   The script will download LMD2, a model that generates short 2-bar sequences of music, and LMD16, which generates longer 16-bar sequences.

## Usage

### Generation

To generate music, just run the `generate.py` script as follows:
```sh
python generate.py model_dir output_dir ...
```
Using this command, the model located in `model_dir` will generate music and the outputs will be saved in `output_dir`.

#### Customizing Track Activation

With Polyphemus, users can specify which instruments should be activated at specific timesteps by editing the `structure.json` file and by passing the `--s_file` argument to `generate.py`. 

The `structure.json` file can be interpreted as a sort of pianoroll for track activations. It contains a JSON representation of a `[n_bars, n_tracks, n_timesteps]` binary tensor. To give a concrete example, consider the following JSON file:
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
When using this kind of conditioning for the LMD2 and LMD16 models, remember that `n_timesteps=32`, which means that each timestep has rhythmic value 1/32, and that the tracks used to train these models are, in order, drums, bass, guitar and strings.

The number of tracks in the JSON file must be strictly equal to the number of tracks used to train the model. The number of bars, instead, can be smaller than the number of bars that can be handled by the model. If this is the case, the script will just replicate the partial binary structure to reach `n_bars`. In this way, it is possible e.g. to compile the binary activations for just one bar and let the model use this same structure for each bar.

### Training

You can train a new model from scratch by using the dedicated script:
```sh
python train.py
```
Run the command with the `--help` flag to find out how you can customize the training procedure.

## License

This project is licensed under the MIT License.
