import argparse
import logging
import sys

import torch

from Diffusion.diffusion import train_diffusion
from GPT.char_gpt import train_char_gpt
from LeNet.lenet import train_lenet
from UNet.unet import train_unet

def main(args):
    parser = argparse.ArgumentParser(description="Train a model")
    models = parser.add_subparsers(dest="model")

    char_gpt_parser = models.add_parser("char-gpt", help="Train a Character-level GPT")
    char_gpt_parser.set_defaults(train=train_char_gpt)

    diff_parser = models.add_parser("diff", help="Train a ddpm diffusion network")
    diff_parser.set_defaults(train=train_diffusion)

    lenet_parser = models.add_parser("lenet", help="Train a LeNet")
    lenet_parser.add_argument("--epochs", "-e", metavar="INPUT", help="Number of epochs", type=int, default=30)
    lenet_parser.set_defaults(train=train_lenet)
    
    unet_parser = models.add_parser("unet", help="Train a UNet")
    unet_parser.set_defaults(train=train_unet)
    
    args = parser.parse_args()
    if not hasattr(args, "train"):
        parser.print_help()
        return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    args.train(device, args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
