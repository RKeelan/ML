import argparse
import sys

import torch
import torch.nn.functional as F
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

def make_model():
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
    P = (N+1).float() # Add 1 to avoid dividing by zero (model smoothing)
    P /= P.sum(1, keepdim=True) # Sum along the rows
    return P, N, stoi, itos, words


def explore_make_more(args):
    P, N, stoi, itos, words = make_model()
    g = torch.Generator().manual_seed(2147483647)
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


def evaluate_make_more(args):
    P, N, stoi, itos, words = make_model()
    g = torch.Generator().manual_seed(2147483647)
    log_likelihood = 0.0
    n = 0
    # for w in words:
    for w in ["andrejq"]:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    print(f'{log_likelihood=}')
    negative_log_likelihood = -log_likelihood
    normalized_log_likelihood = negative_log_likelihood / n
    print(f'{negative_log_likelihood=}')
    print(f'{normalized_log_likelihood=}')


def make_training_set():
    xs, ys = [], []
    words = open('data/names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    for w in words[:1]:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return itos, xs, ys


def train_make_more(args):
    itos, xs, ys = make_training_set()

    # Randomly initialize 27 neurons' weights using the Gaussian (Normal) distribution
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g)

    # Forward pass
    x_encoded = F.one_hot(xs, num_classes=27).float()
    # Hidden layer
    logits = x_encoded @ W

    # Softmax
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    nlls = torch.zeros(5) # Negative log-likelihoods
    for i in range(5):
        x = xs[i].item() # Input character index
        y = ys[i].item() # Label character index
        print("--------")
        print(f"Bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})")
        print("Input to the neural net: ", x)
        print("Output probabilities from the neural net: ", probs[i])
        print("Label (actual next character): ", y)
        p = probs[i, y]
        print("Probability assigned by the net to the the correct character: ", p.item())
        logp = torch.log(p)
        print("Log likelihood: ", logp.item())
        nll = -logp
        print("Negative log likelihood: ", nll.item())
        nlls[i] = nll

    print("========")
    print("Average negative log likelihood: ", nlls.mean().item())


def main(args):
    parser = argparse.ArgumentParser(description="Explore data")
    commands = parser.add_subparsers(dest="cmd")

    explore_cmd = commands.add_parser("explore", help="Explore MakeMore data")
    explore_cmd.set_defaults(action=explore_make_more)

    evaluate_cmd = commands.add_parser("eval", help="Evaluate MakeMore data")
    evaluate_cmd.set_defaults(action=evaluate_make_more)

    train_cmd = commands.add_parser("train", help="Train MakeMore")
    train_cmd.set_defaults(action=train_make_more)
    
    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_help()
        return 1

    args.action(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
