import argparse
import logging
import sys

import torch

from LeNet.lenet import lenet_predict
from UNet.unet import unet_predict


def main(args):
    parser = argparse.ArgumentParser(description="Prdict using a model")
    models = parser.add_subparsers(dest="model_type")

    lenet_parser = models.add_parser("lenet", help="Predict with LeNet")
    lenet_parser.add_argument("--input", "-i", metavar="INPUT", help="Input directory", required=True)
    lenet_parser.set_defaults(predict=lenet_predict)

    unet_parser = models.add_parser("unet", help="Predict with UNet")
    unet_parser.add_argument("--model_path", "-m", metavar="FILE", help="Path to model", required=True)
    unet_parser.add_argument("--input", "-i", metavar="INPUT", nargs="+", help="Input files", required=True)
    unet_parser.add_argument("--output", "-o", metavar="OUTPUT", nargs='+', help="Output files")
    unet_parser.add_argument("--display", "-d", action="store_true", help="Display images")
    unet_parser.add_argument('--no-save', '-n', action='store_true', help='Do not save intermediate outputs')
    unet_parser.set_defaults(predict=unet_predict)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    if not hasattr(args, "predict"):
        parser.print_help()
        return 1
    
    args.predict(device, args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))