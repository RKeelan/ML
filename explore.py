import argparse
import sys

from D2L.d2l import explore_d2l
from Diffusion.diffusion import explore_diffusion


def main(args):
    parser = argparse.ArgumentParser(description="Explore a model's data")
    commands = parser.add_subparsers(dest="model")
    
    diffusion_cmd = commands.add_parser("diff", help="Explore diffusion data")
    diffusion_cmd.add_argument("--show_images", "-i", action="store_true", help="Show Images")
    diffusion_cmd.add_argument("--noise_image", "-n", action="store_true", help="Simulate forward diffusion")
    diffusion_cmd.set_defaults(explore=explore_diffusion)

    d2l_cmd = commands.add_parser("d2l", help="Explore d2l data")
    d2l_cmd.add_argument("--dataset", "-ds", required=False)
    d2l_cmd.set_defaults(explore=explore_d2l)
    
    args = parser.parse_args()
    if not hasattr(args, "explore"):
        parser.print_help()
        return 1

    args.explore(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
