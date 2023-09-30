import argparse
import os
import json
import uuid

import torch
import os
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from data import PolyphemusDataset
import torch.optim as optim

from model import VAE
from utils import set_seed, print_params, print_divider
from training import PolyphemusTrainer, ExpDecayLRScheduler, StepBetaScheduler


def main():

    parser = argparse.ArgumentParser(
        description='Trains Polyphemus.'
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Directory of the Polyphemus dataset to be used for training.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save the output of the training.'
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON training configuration file.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model to be trained.'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help="If set to n, the script will save the model every n batches. "
        "Default is 10."
    )
    parser.add_argument(
        '--print_every',
        type=int,
        default=1,
        help="If set to n, the script will print statistics every n batches. "
        "Default is 1."
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        default=False,
        help='Flag to enable evaluation on a validation set.'
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        help="If the eval flag is set, when set to n, the script will evaluate "
        "the model on the validation set every n batches. "
        "Default is every epoch."
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
        '--num_workers',
        type=int,
        default='10',
        help="The number of processes to use for loading the data. "
        "Default is 10."
    )
    parser.add_argument(
        '--tr_split',
        type=float,
        default='0.7',
        help="Percentage of samples in the dataset used for the training split."
        " Default is 0.7."
    )
    parser.add_argument(
        '--vl_split',
        type=float,
        default='0.1',
        help="Percentage of samples in the dataset used for the validation "
        "split. Default is 0.1. This value is ignored if the --eval option is "
        "not specified."
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default='100',
    )
    parser.add_argument(
        '--seed',
        type=int
    )

    args = parser.parse_args()
    print_divider()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    if args.use_gpu:
        torch.cuda.set_device(args.gpu_id)
        
    # Load config file
    print("Loading the configuration file {}...".format(args.config_file))

    # Load structure tensor from file
    with open(args.config_file, 'r') as f:
        training_config = json.load(f)
    
    n_bars = training_config['model']['n_bars']
    batch_size = training_config['batch_size']
        
    print("Preparing datasets and dataloaders...")
    
    dataset = PolyphemusDataset(args.dataset_dir, n_bars)
    
    tr_len = int(args.tr_split * len(dataset))
    
    if args.eval:
        vl_len = int(args.vl_split * len(dataset))
        ts_len = len(dataset) - tr_len - vl_len
        lengths = (tr_len, vl_len, ts_len)
    else:
        ts_len = len(dataset) - tr_len
        lengths = (tr_len, ts_len)
        
    split = random_split(dataset, lengths)
    tr_set = split[0]
    vl_set = split[1] if args.eval else None

    trainloader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, 
                             num_workers=args.num_workers)
    if args.eval:
        validloader = DataLoader(vl_set, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        eval_every = len(trainloader)
    else:
        validloader = None
        eval_every = None

    
    model_name = (args.model_name if args.model_name is not None 
                  else str(uuid.uuid1()))
    model_dir = os.path.join(args.output_dir, model_name)
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model output directory (raise error if it already exists to avoid
    # overwriting a trained model) 
    os.makedirs(model_dir, exist_ok=False)
    
    # Create the model
    print("Creating the model and moving it on {} device...".format(device))
    vae = VAE(**training_config['model'], device=device).to(device)
    print_params(vae)
    print()
    
    # Creating optimizer and schedulers
    optimizer = optim.Adam(vae.parameters(), **training_config['optimizer'])
    lr_scheduler = ExpDecayLRScheduler(
        optimizer=optimizer,
        **training_config['lr_scheduler']
    )
    beta_scheduler = StepBetaScheduler(**training_config['beta_scheduler'])
    
    
    # Save config
    config_path = os.path.join(model_dir, 'configuration')
    torch.save(training_config, config_path) 
    
    print("Starting training...")
    print_divider()

    trainer = PolyphemusTrainer(
        model_dir,
        vae,
        optimizer,
        lr_scheduler=lr_scheduler,
        beta_scheduler=beta_scheduler,
        save_every=args.save_every,
        print_every=args.print_every,
        eval_every=eval_every,
        device=device
    )
    trainer.train(trainloader, validloader=validloader, epochs=args.max_epochs)


if __name__ == '__main__':
    main()
