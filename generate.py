import argparse
import os
import json
import time
import math

import torch
import os
from matplotlib import pyplot as plt

import generation_config
from model import VAE
from utils import set_seed
from utils import mtp_from_logits, muspy_from_mtp, set_seed
from utils import print_divider
from utils import save_midi, save_audio
from plot import plot_pianoroll, plot_structure


def generate_music(vae, z, s_cond=None, s_tensor_cond=None):

    # Decoder pass to get structure and content logits
    s_logits, c_logits = vae.decoder(z, s_cond)

    if s_tensor_cond != None:
        s_tensor = s_tensor_cond
    else:
        # Compute binary structure tensor from logits
        s_tensor = vae.decoder._binary_from_logits(s_logits)

    # Build (n_batches x n_bars x n_tracks x n_timesteps x Sigma x d_token)
    # multitrack pianoroll tensor containing logits for each activation and
    # hard silences elsewhere
    mtp = mtp_from_logits(c_logits, s_tensor)

    return mtp, s_tensor


def save(mtp, dir, s_tensor=None, n_loops=1,
         looped_only=False, plot_proll=False, plot_struct=False):

    # Clear matplotlib cache (this solves formatting problems with first plot)
    plt.clf()

    # Iterate over batches
    for i in range(mtp.size(0)):

        # Create the directory if it does not exist
        save_dir = os.path.join(dir, str(i))
        os.makedirs(save_dir, exist_ok=True)

        print("Saving MIDI sequence {}...".format(str(i+1)))

        if not looped_only:
            # Generate MIDI song from multitrack pianoroll and save
            muspy_song = muspy_from_mtp(mtp[i])
            save_midi(muspy_song, save_dir, name='generated')
            save_audio(muspy_song, save_dir, name='generated')

        if plot_proll:
            plot_pianoroll(muspy_song, save_dir)

        if plot_struct:
            plot_structure(s_tensor[i].cpu(), save_dir)

        if n_loops > 1:
            # Copy the generated sequence n_loops times and save the looped
            # MIDI and audio files
            print("Saving MIDI sequence "
                  "{} looped {} times...".format(str(i + 1), n_loops))
            extended = mtp[i].repeat(n_loops, 1, 1, 1, 1)
            extended = muspy_from_mtp(extended)
            save_midi(extended, save_dir, name='extended')
            save_audio(extended, save_dir, name='extended')

        print()


def generate_z(bs, d_model, device):
    shape = (bs, d_model)

    z_norm = torch.normal(
        torch.zeros(shape, device=device),
        torch.ones(shape, device=device)
    )

    return z_norm


def load_model(model_dir, device):

    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint'),
                            map_location='cpu')
    params = torch.load(os.path.join(model_dir, 'params'),
                        map_location='cpu')

    state_dict = checkpoint['model_state_dict']

    model = VAE(**params['model'], device=device).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, params


def main():

    parser = argparse.ArgumentParser(
        description='Generates MIDI music with a trained model.'
    )
    parser.add_argument(
        'model_dir',
        type=str, help='Directory of the model.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save the generated MIDI files.'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=5,
        help='Number of sequences to be generated. Default is 5.'
    )
    parser.add_argument(
        '--n_loops',
        type=int,
        default=1,
        help="If greater than 1, outputs an additional MIDI file containing "
        "the sequence looped n_loops times."
    )
    parser.add_argument(
        '--s_file',
        type=str,
        help='Path to the JSON file containing the binary structure tensor.'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        default=False,
        help='Flag to enable or disable GPU usage. Default is False.'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default='0',
        help='Index of the GPU to be used. Default is 0.'
    )
    parser.add_argument(
        '--seed',
        type=int
    )

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    if args.use_gpu:
        torch.cuda.set_device(args.gpu_id)

    print_divider()
    print("Loading the model on {} device...".format(device))

    model, params = load_model(args.model_dir, device)

    d_model = params['model']['d']
    n_bars = params['model']['n_bars']
    n_tracks = params['model']['n_tracks']
    n_timesteps = 4 * params['model']['resolution']
    output_dir = args.output_dir
    bs = args.n

    s, s_tensor = None, None

    if args.s_file is not None:

        print("Loading the structure tensor "
              "from {}...".format(args.model_dir))

        # Load structure tensor from file
        with open(args.s_file, 'r') as f:
            s_tensor = json.load(f)

        s_tensor = torch.tensor(s_tensor)

        # Check structure dimensions
        dims = list(s_tensor.size())
        expected = [n_bars, n_tracks, n_timesteps]
        if dims != expected:
            if (len(dims) != len(expected) or dims[1:] != expected[1:]
                    or dims[0] > n_bars):
                raise ValueError(f"Loaded structure tensor dimensions {dims} "
                                 f"do not match expected dimensions {expected}")
            elif dims[0] > n_bars:
                raise ValueError(f"First structure tensor dimension {dims[0]} "
                                 f"is higher than {n_bars}")
            else:
                # Repeat partial structure tensor
                r = math.ceil(n_bars / dims[0])
                s_tensor = s_tensor.repeat(r, 1, 1)
                s_tensor = s_tensor[:n_bars, ...]

        s_tensor = s_tensor.bool()
        s_tensor = s_tensor.unsqueeze(0).repeat(bs, 1, 1, 1)
        s = model.decoder._structure_from_binary(s_tensor)

    print()
    print("Generating z...")
    z = generate_z(bs, d_model, device)

    print("Generating music with the model...")
    s_t = time.time()
    mtp, s_tensor = generate_music(model, z, s, s_tensor)
    print("Inference time: {:.3f} s".format(time.time() - s_t))

    print()
    print("Saving MIDI files in {}...\n".format(output_dir))
    save(mtp, output_dir, s_tensor, args.n_loops)
    print("Finished saving MIDI files.")


if __name__ == '__main__':
    main()
