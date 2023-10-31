import argparse
import sys

from Diffusion.diffusion import explore_diffusion


def main(args):
    parser = argparse.ArgumentParser(description="Explore a model's data")
    models = parser.add_subparsers(dest="model")
    
    diffusion_parser = models.add_parser("diff", help="Explore diffusion data")
    diffusion_parser.add_argument("--show_images", "-i", action="store_true", help="Show Images")
    diffusion_parser.add_argument("--noise_image", "-n", action="store_true", help="Simulate forward diffusion")
    diffusion_parser.set_defaults(explore=explore_diffusion)
    
    args = parser.parse_args()
    if not hasattr(args, "explore"):
        parser.print_help()
        return 1

    args.explore(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
