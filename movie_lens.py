import argparse
import os
import sys

import pandas as pd
from mxnet import gluon, np
from d2l import mxnet as d2l

#@save
d2l.DATA_HUB['ml-100k'] = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "cd4dcac4241c8a4ad7badc7ca635da8a69dddb83"
)

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, "u.data"), sep="\t",
                       names=names, engine="python")
    num_users = data.user_id.unique.shape[0]
    num_items = data.item_id.unique.shape[0]
    return data, num_users, num_items

def explore_movie_lens(args):
    data, num_users, num_items = read_data_ml100k()
    sparsity = 1 - len(data) / (num_users * num_items)
    pass

def main(args):
    parser = argparse.ArgumentParser(description="Movie Lens")
    commands = parser.add_subparsers(dest="cmd")


    explore_cmd = commands.add_parser("explore", help="Explore MovieLens data")
    explore_cmd.set_defaults(action=explore_movie_lens)
    
    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_help()
        return 1

    args.action(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))