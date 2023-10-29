import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embedding_size, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        _,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Calculate attention scores
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)

        # Performed weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attenion in parallel"""
    def __init__(self, num_heads, head_size, embedding_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(embedding_size, head_size, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatente over the channel dimension
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size), # The "projection" layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: Communication followed by computation"""
    def __init__(self, num_heads, embedding_size, block_size, dropout):
        super().__init__()
        head_size = embedding_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size, embedding_size, block_size, dropout)
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, block_size, num_heads, dropout, num_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[
            Block(num_heads, embedding_size, block_size, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.language_modelling_head = nn.Linear(embedding_size, vocabulary_size)


    def forward(self, indices, targets=None):
        B, T = indices.shape

        token_embedding = self.token_embedding_table(indices) # (B,T,C)
        # Is it really necessary to pass the device into here? It doesn't matter for me, becuase I don't have
        # a graphics card, but I want to keep this line  in case I get set up with a graphics card and this breaks
        #position_embedding = self.position_embedding_table(torch.arange(T), device=device) # (T,C)
        position_embedding = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = token_embedding + position_embedding # (B,T,C)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.language_modelling_head(x) # (B,T,vocabulary_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, indices, block_size, max_new_tokens):
        # indices is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop the indices to the last block_size tokens
            idx_cropped = indices[:, -block_size:]

            # Get the predictions
            logits, _ = self(idx_cropped)

            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # Sample from the distribution
            idxs = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append sampled index to the running sequence
            indices = torch.cat((indices, idxs), dim=1) # (B, T+1)

        return indices


def get_batch(device: torch.device, data, batch_size, block_size):
    """Generate a small batch of data of inputs x and targets y"""
    # Generate random offsets into the data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    device: torch.device,
    train_data,
    val_data,
    batch_size,
    block_size,
    evaluation_iterations
):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            data = train_data if split == 'train' else val_data
            X, Y = get_batch(device, data, batch_size, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_char_gpt(device: torch.device, args):
    logging.info("Training Character-level GPT")

    with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Build vocabulary
    chars = sorted(list(set(text)))
    vocabulary_size = len(chars)

    # Tokenizer
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    data = torch.tensor(encode(text), dtype=torch.long)
    logging.info(f"Input shape: {data.shape}")

    # Train and test splits
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Hyperparameters
    if torch.cuda.is_available():
        batch_size = 64 # Number of independent sequences to process in parallel
        block_size = 256 # Maximum context length for predictions
        max_iterations = 5000
        evaluation_interval = 500
        learning_rate = 3e-4
        evaluation_iterations = 200
        embedding_size = 384
        num_heads = 6
        dropout = 0.2
        num_layers = 6
    else:
        batch_size = 32 # Number of independent sequences to process in parallel
        block_size = 128 # Maximum context length for predictions
        max_iterations = 5000
        evaluation_interval = 500
        learning_rate = 3e-4
        evaluation_iterations = 200
        embedding_size = 192
        num_heads = 6
        dropout = 0.2
        num_layers = 6

    torch.manual_seed(1337)

    logging.info(f"""Starting training:
        Batch size: {batch_size}
        Block size: {block_size}
        Max iterations: {max_iterations}
        Evaluation interval: {evaluation_interval}
        Learning rate: {learning_rate}
        Evaluation intervals: {evaluation_iterations}
        Embedding size: {embedding_size}
        Num heads: {num_heads}
        Dropout: {dropout}
        Num layers: {num_layers}
    """)
    model = CharGPT(vocabulary_size, embedding_size, block_size, num_heads, dropout, num_layers)
    model = model.to(device)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(max_iterations):

        if i % evaluation_interval == 0:
            losses = estimate_loss(model, device, train_data, val_data, batch_size, block_size, evaluation_iterations)
            logging.info(f"Step: {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(device, train_data, batch_size, block_size)

        # Evaluate the lossii O:
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, block_size, max_new_tokens=500)[0].tolist()))