import argparse
import sys

import torch
import matplotlib.pyplot as plt


def show_plot(N, itos):
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.show()


def explore_make_more(args):
    words = open('data/names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    N = torch.zeros((27,27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # show_plot(N, itos)
    
    g = torch.Generator().manual_seed(2147483647)
    p = torch.rand(3, generator=g)
    p = p / p.sum()
    # print(p)
    # print(torch.multinomial(p, num_samples=100, replacement=True, generator=g))

    p = N[0].float()
    p = p / p.sum()
    # print(p)

    g = torch.Generator().manual_seed(2147483647)
    P = N.float()
    P /= P.sum(1, keepdim=True) # Sum along the rows
    for _ in range(10):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

def main(args):
    parser = argparse.ArgumentParser(description="Explore data")
    commands = parser.add_subparsers(dest="cmd")

    explore_cmd = commands.add_parser("explore", help="Explore MakeMore data")
    explore_cmd.set_defaults(action=explore_make_more)
    
    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_help()
        return 1

    args.action(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
