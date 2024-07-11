import argparse
import random
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

CONTEXT_LENGTH = 3

def build_dataset(stoi: dict, words: list) -> tuple:
    block_size = CONTEXT_LENGTH
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # Crop and  append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def main(args):
    words = open('data/names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    
    Xtr, Ytr = build_dataset(stoi, words[:n1])
    Xdev, Ydev = build_dataset(stoi, words[n1:n2])
    Xte, Yte = build_dataset(stoi, words[n2:])

    print(f"{len(Xtr)=} training examples")

    # Parameters
    VOCABULARY_SIZE = 27
    EMBEDDING_SIZE = 10
    NUM_NEURONS = 200
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((VOCABULARY_SIZE, EMBEDDING_SIZE), generator=g)
    W1 = torch.randn((CONTEXT_LENGTH * EMBEDDING_SIZE, NUM_NEURONS), generator=g)
    b1 = torch.randn(NUM_NEURONS, generator=g)
    W2 = torch.randn((NUM_NEURONS, VOCABULARY_SIZE), generator=g)
    b2 = torch.randn(VOCABULARY_SIZE, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    num_params = sum(p.numel() for p in parameters)
    print(f"{num_params:,} parameters")

    NUM_EPOCHS = 200000
    MINI_BATCH_SIZE = 32
    LEARNING_RATE = 0.1
    # lre = torch.linspace(-3, 0, NUM_MINI_BATCHES) # Learning rate exponents
    # lrs = 10**lre # Learning rate steps
    step_i = [] # Ith step
    lre_i = [] # Ith learning rate
    loss_i = [] # Ith loss
    for i in range(NUM_EPOCHS):
        # Make minibatch
        ix = torch.randint(0, Xtr.shape[0], (MINI_BATCH_SIZE,), generator=g)
        
        # Forward pass
        embeddings = C[Xtr[ix]] # (32, 3, 2)
        h = torch.tanh(embeddings.view(-1, CONTEXT_LENGTH * EMBEDDING_SIZE) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Ytr[ix])
        if i % (NUM_EPOCHS/20) == 0:
            print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # lr = lrs[i]
        lr = 0.1 if i < (NUM_EPOCHS/2) else 0.01
        for p in parameters:
            p.data += -LEARNING_RATE * p.grad
        
        # Track stats
        # lre_i.append(lre[i])
        step_i.append(i)
        loss_i.append(loss.log10().item())

    # Model loss
    embeddings = C[Xtr] # (32, 3, 2)
    h = torch.tanh(embeddings.view(-1, CONTEXT_LENGTH * EMBEDDING_SIZE) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, Ytr)
    print(f"Training loss: {loss.item():.4f}")

    embeddings = C[Xdev] # (32, 3, 2)
    h = torch.tanh(embeddings.view(-1, CONTEXT_LENGTH * EMBEDDING_SIZE) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, Ydev)
    print(f"Dev loss: {loss.item():.4f}")

    # Plot the loss against the learning rate
    # plt.plot(lre_i, loss_i)
    # plt.show()
    
    # Plot the loss against the step
    # plt.plot(step_i, loss_i)
    # plt.show()

    # Visualize the trained embeddings (when they're length 2)
    # plt.figure(figsize=(8,8))
    # plt.scatter(C[:,0].data, C[:,1].data, s=200)
    # for i in range(C.shape[0]):
    #     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha='center', va='center', color='white')
    # plt.grid('minor')
    # plt.show()

    # Sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        context = [0] * CONTEXT_LENGTH
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:] + [ix]
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

            





if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))