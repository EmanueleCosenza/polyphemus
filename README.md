# polyphemus

A graph-based deep MIDI music generator. Official repository of the [paper](https://arxiv.org/abs/2307.14928) "Graph-based Polyphonic Multitrack Music Generation".

You can interact with the model in the dedicated Hugging Face [space](https://huggingface.co/spaces/EmanueleCosenza/polyphemus) and listen to the samples from the experimental phase on the paper's [site](https://emanuelecosenza.github.io/polyphemus/).

The source code is written in Python 3.7, using PyTorch and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) to implement and train Deep Graph Networks.

## Prerequisites

To run Polyphemus, follow these steps:

1. **Create an Environment:**
   Use conda to create a Python 3.7 environment and activate it:
   ```sh
   conda create -n polyphemus-env python=3.7
   conda activate polyphemus-env
   ```
   Alternatively, you can install Python 3.7 and create a new virtual environment using `venv`.

2. **Clone the Repository:**
   ```sh
   git clone https://github.com/EmanueleCosenza/polyphemus
   ```
   
4. **Install the Required Python Packages:**
   Navigate to the directory where the repository is cloned and install the required packages from `requirements.txt`:
   ```sh
   pip3 install -r requirements.txt
   ```

5. **Prepare Your Environment for Audio Generation (Recommended):**
   If you intend to generate audio from MIDI files directly with the provided scripts, you will need a sound synthesizer like [`fluidsynth`](https://github.com/FluidSynth/fluidsynth/wiki) and a [SoundFont](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont) `.sf2` file that tells `fluidsynth` which audio samples should be played for each MIDI instrument.
   
   You can get `fluidsynth` with the following command:
   ```sh
   sudo apt-get install fluidsynth
   ```
   or, if you prefer, through `conda` as follows:
   ```sh
   conda install -c conda-forge fluidsynth
   ```

   To quickly get a working SoundFont, you can run the following:
   ```sh
   sudo apt-get install fluid-soundfont-gm
   ```
   This installs the Fluid (R3) General MIDI SoundFont, a SoundFont that is known to work well with `fluidsynth`. By default, after the installation, the SoundFont will be stored in `/usr/share/sounds/sf2/FluidR3_GM.sf2`.

   If you don't have root access to your machine, you can do the same by running the shell script provided in the repo:
   ```sh
   ./download_soundfont.sh soundfonts/
   ```
   This automatically finds the `FluidR3_GM.sf2` file in the `fluid-soundfont-gm` package and downloads it in the `soundfonts/` directory.

   You can get many other SoundFonts by following the links provided in the [FluidSynth project Wiki](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont).
   
   ⚠️ After downloading the SoundFont in one of the ways shown above, make sure to properly set the `SOUNDFONT_PATH` variable in the `generation_config.yaml` file.

6. **Download the Trained Models:**
   Run the `download_models.py` script to download the models trained on the LMD dataset from Polyphemus' Hugging Face model [repo](https://huggingface.co/EmanueleCosenza/polyphemus):
   ```sh
   python3 download_models.py models/
   ```
   The script will download LMD2, a model that generates short 2-bar sequences of music, and LMD16, which generates longer 16-bar sequences, in the `models/` directory. Both models have been trained on 4/4 music and are designed to process sequences comprising 4 tracks (=instruments). These are, in order, drums, bass, guitar and strings. The guitar track typically plays accompaniment chords and it can also be intepreted as a piano/accompaniment track. The strings track typically contains a lead melody. Finally, for both models, each bar in the sequence is composed of 32 timesteps, which means that each timestep has a rhythmic value of 1/32.


## Generation

If you have already installed Polyphemus and downloaded the pretrained models, in order to generate music, you can simply run the `generate.py` script as follows:
```sh
python3 generate.py models/LMD2/ music/ --n 10 --n_loops 4
```
By executing this command, the LMD2 model located in `models/LMD2/` will generate 10 sequences and the results will be saved in `music/`. The `--n_loops` argument tells the script to output, for each sequence, an additional extended sequence obtained by looping the generated one 4 times (this is advisable for 2-bar sequences, since they are quite short).

By default, `generate.py` outputs `.wav` audio files in addition to MIDI files. You can change the way audio is generated from MIDI data by editing the `generation_config.yaml` file. Here, you can specify the SoundFont file that has to be used while generating audio and the MIDI programs that have to be associated to each track.

Run `generate.py` with the `--help` flag to get a complete list of all the arguments you can pass to the script.

### Structure Conditioning	

With Polyphemus, you can condition generation by specifying which tracks (i.e. instruments) should be activated at specific timesteps. You can do so by editing the `structure.json` file and by passing the `--s_file` argument to `generate.py` as follows:
```sh
python3 generate.py models/LMD2/ music/ --n 10 --n_loops 4 --s_file structure.json
```

The `structure.json` file represents the rhythmic structure of the music to be generated and can be interpreted as a sort of pianoroll for instrument activations. The file contains a JSON representation of a `[n_bars, n_tracks, n_timesteps]` binary tensor. To give a concrete example, consider the following JSON data:
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
Here, for simplicity, `n_timesteps=8`. In the first bar, Track 2 and Track 4 will be activated, respectively, in the first and fifth timestep. In the second bar, just the second track will be activated in the fifth timestep. All other tracks will be silenced throughout the sequence.

Notice that the binary activations above refer to actual note activations and that sustain is not considered. This means that a track could end up playing a sustained note even if the corresponding value has been set to 0 in the file. For instance, Track 2, which has been activated in the first timestep of the sequence, could actually sustain its activated notes for more than a timestep.

The number of tracks for each bar in the JSON file must be strictly equal to the number of tracks used to train the model. The number of bars, instead, may be smaller than the number of bars that can be handled by the model. If this is the case, the script will just replicate the partial binary structure to reach `n_bars`. In this way, it is possible e.g. to compile the binary activations for just one bar and let the model use this same structure for each bar.

When editing this file for the LMD2 and LMD16 models, remember that `n_timesteps=32`, which means that each timestep has a rhythmic value of 1/32, and that the tracks used to train these models are, in order, drums, bass, guitar and strings.

## Training a New Model from Scratch

### Preprocessing

Before you can train Polyphemus from scratch, you have to preprocess a MIDI dataset. This can be done by running the `preprocess.py` script as follows:
```sh
python3 preprocess.py midi_dataset_dir preprocessed_dir
```
where `midi_dataset_dir` is the directory of the MIDI dataset and `preprocessed_dir` is the directory to save the preprocessed dataset. For the script to work, the `midi_dataset_dir` directory must only contain `.mid` files in a flat or hierarchical fashion (i.e. in a tree of subdirectories).

If you want to preprocess the Lahk MIDI Dataset (`LMD-matched`), you can first download it from [here](https://colinraffel.com/projects/lmd/), or just execute the following:
```sh
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
```
Then, you can decompress the archive by using `tar`.


### Training

 You can train a new model from scratch by using the dedicated script:
```sh
python3 train.py dataset_dir model_dir config_file --model_name test --use_gpu
```
where `dataset_dir` is the directory of the preprocessed dataset to be used for training, `model_dir` is the directory to save the trained model, and `config_file` is the path to a JSON training configuration file. An example of this file is provided in the repo as `training.json`.

Make sure to run the command with the `--help` flag to find out how you can customize the training procedure.

No integration with TensorBoard or W&B is provided, but you can still check the progress of training with the `training_stats.ipynb` Jupyter Notebook. 

## License

This project is licensed under the MIT License.
