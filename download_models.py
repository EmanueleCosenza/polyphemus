import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Downloads Polyphemus' pretrained models from Hugging "
        "Face."
    )
    parser.add_argument(
        'models_dir',
        type=str, help='Directory to store the pretrained models.'
    )
    
    args = parser.parse_args()
    
    repo_id = 'EmanueleCosenza/polyphemus'

    r = snapshot_download(
        repo_id=repo_id,
        local_dir=args.models_dir,
        local_dir_use_symlinks=False    
    )
    print("Models successfully downloaded in {}".format(r))


if __name__ == '__main__':
    main()