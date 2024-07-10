import argparse
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def main(args):
    words = open('data/names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0

    # Build the dataset
    CONTEXT_LENGTH = 3
    block_size = CONTEXT_LENGTH
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # Crop and  append
    
    # Input
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"{len(X)=} examples")

    # Parameters
    VOCABULARY_SIZE = 27
    EMBEDDING_SIZE = 2
    NUM_NEURONS = 100
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((VOCABULARY_SIZE, EMBEDDING_SIZE), generator=g)
    W1 = torch.randn((CONTEXT_LENGTH * EMBEDDING_SIZE, NUM_NEURONS), generator=g)
    b1 = torch.randn(NUM_NEURONS, generator=g)
    W2 = torch.randn((NUM_NEURONS, VOCABULARY_SIZE), generator=g)
    b2 = torch.randn(VOCABULARY_SIZE, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    print(sum(p.nelement() for p in parameters), "parameters")

    # Candidate learning rate


    NUM_EPOCHS = 1
    NUM_MINI_BATCHES = 10000
    MINI_BATCH_SIZE = 32
    LEARNING_RATE = 1e-1    
    # lre = torch.linspace(-3, 0, NUM_MINI_BATCHES) # Learning rate exponents
    # lrs = 10**lre # Learning rate steps
    # lre_i = [] # Ith learning rate
    # loss_i = [] # Ith loss
    for _ in range(NUM_EPOCHS):
        for i in range(NUM_MINI_BATCHES):
            # Make minibatch
            ix = torch.randint(0, X.shape[0], (MINI_BATCH_SIZE,), generator=g)
            
            # Forward pass
            embeddings = C[X[ix]] # (32, 3, 2)
            h = torch.tanh(embeddings.view(-1, CONTEXT_LENGTH * EMBEDDING_SIZE) @ W1 + b1) # (32, 100)
            logits = h @ W2 + b2 # (32, 27)
            loss = F.cross_entropy(logits, Y[ix])
            if i % (NUM_MINI_BATCHES/100) == 0:
                print(f"Loss: {loss.item():.4f}")
            
            # Backward pass
            for p in parameters:
                p.grad = None
            loss.backward()
            # lr = lrs[i]
            for p in parameters:
                p.data += -LEARNING_RATE * p.grad
            
            # Track stats
            # lre_i.append(lre[i])
            # loss_i.append(loss.item())

        # Model loss
        embeddings = C[X] # (32, 3, 2)
        h = torch.tanh(embeddings.view(-1, CONTEXT_LENGTH * EMBEDDING_SIZE) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Y)
        print(f"Loss: {loss.item():.4f}")

        # Plot the loss against the learning rate
        # plt.plot(lre_i, loss_i)
        # plt.show()




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))