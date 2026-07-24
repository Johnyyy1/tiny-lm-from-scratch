import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


data_file = Path(
    os.environ.get("MINIBPE_DATA_FILE", "/Users/jonas/Downloads/training_data.txt")
)
if not data_file.is_file():
    raise FileNotFoundError(
        f"Training data not found at {data_file}. "
        "Set MINIBPE_DATA_FILE to a UTF-8 text file with one sample per line."
    )

with data_file.open(encoding="utf-8") as handle:
    data = [line.rstrip("\n") for line in handle]

if len(data) < 2:
    raise ValueError("Training data must contain at least two lines.")

split_index = max(1, min(len(data) - 1, int(len(data) * 0.9)))
data_trn = data[:split_index]
data_val = data[split_index:]

# Get the unique characters across the training set
unique_chars = set()

for story in data_trn:
    unique_chars.update(set(story))

unique_chars = ''.join(sorted(unique_chars))

# We use '^' as a special start character in the model
assert '^' not in unique_chars

stop_char = '^'

# Add stop character to each string
data_trn[:] = [s + "^" for s in data_trn]
data_val[:] = [s + "^" for s in data_val]

stoi = {s:i+1 for i, s in enumerate(unique_chars)}
stoi[stop_char] = 0
itos = {i:s for s, i in stoi.items()}

print(f"Initial vocabulary size (pre-BPE): {len(stoi)}")

# initial tokens are from the raw data
cur_tokens = []
nr_trn = len(data_trn)

# Tokenization loop

for s in data_trn[:nr_trn]:
    cur_tokens += list(s)

init_token_count = len(cur_tokens)
nr_actual_tokens = len(stoi)

# This is pretty arbitrary - I set another limit below to stop when the
# token length exceeds 40 characters (determined empirically)
nr_desired_tokens = int(os.environ.get("MINIBPE_VOCAB_SIZE", "5000"))
if nr_desired_tokens < len(stoi):
    raise ValueError(
        f"MINIBPE_VOCAB_SIZE must be at least the initial vocabulary size ({len(stoi)})."
    )

print(f"Initial token count: {init_token_count}")

while nr_actual_tokens < nr_desired_tokens:
    new_tokens = []
    counts = {}
    candidates = {}

    for i in range(len(cur_tokens)):
        if i == len(cur_tokens) - 1:
            break

        left = cur_tokens[i]
        right = cur_tokens[i+1]

        # Prevent merging on the stop_char so it is easier to deliminate
        # each sample from the dataset
        if left == stop_char or right == stop_char:
            continue

        tok = left + right

        if tok not in counts:
            counts[tok] = 1
            candidates[tok] = {}
            candidates[tok]["left"] = left
            candidates[tok]["right"] = right
        else:
            counts[tok] += 1

    if not counts:
        print("No mergeable token pairs remain (stopping)")
        break

    # Need the key which has max count
    new_token = max(counts, key=counts.get)
    left = candidates[new_token]['left']
    right = candidates[new_token]['right']
    cursor = 0

    for i in range(len(cur_tokens)):
        if i == 0:
            continue

        # Check for merge condition
        #
        # We merge if the left and right tokens match, and the cursor is not i.
        # If cursor is i, it means we just merged, and could merge again (two matches
        # in a row), but merges are non-overlapping, so we just skip over, keeping
        # cursor where it is.
        if cur_tokens[i-1] == left and cur_tokens[i] == right and cursor != i:
            if cursor < i - 1:
                # Cursor is behind the left token, so copy [cursor, left token)
                new_tokens += cur_tokens[cursor:i-1] + [new_token]
            else:
                new_tokens += [new_token]

            # anytime we merge, we move the cursor to the right of the right-merge token
            cursor = i + 1

    # Grab the end. this also cleanly covers the degenerate case where no merges happened
    if cursor <= len(cur_tokens) - 1:
        new_tokens += cur_tokens[cursor:]

    cur_tokens = new_tokens
    new_len = len(cur_tokens)

    # A string can be produced by more than one merge path. Reuse its existing
    # id rather than overwriting the forward map with a second id.
    if new_token not in stoi:
        stoi[new_token] = nr_actual_tokens
        itos[nr_actual_tokens] = new_token
        nr_actual_tokens += 1

    print(f"Merged {new_token}")

    if nr_actual_tokens % 1000 == 0:
        print(f"Tokens: {nr_actual_tokens} ({(init_token_count - new_len) / init_token_count:.2f}% reduction)")

    if len(new_token) > 40:
        print(f"Max token length is {len(new_token)}: {new_token}   (stopping)")
        print(f"Number of tokens: {nr_actual_tokens}")
        break

print(f"Vocab size (post-BPE): {len(stoi)}")

def tokenize_string(s, sorted_stoi):
    tokens = []
    cursor = 0

    while cursor < len(s):
        for k in sorted_stoi:
            if s[cursor:].startswith(k):
                tokens.append(k)
                cursor += len(k)
                break
        else:
            raise ValueError(
                f"No vocabulary token matches character {s[cursor]!r} "
                f"at position {cursor}."
            )

    return tokens

sorted_stoi = {k : stoi[k] for k in sorted(stoi, key=len, reverse=True)}

# We have the tokenization map stoi. Now we need to tokenize the training and validation sets
trn_stories = data_trn
val_stories = data_val

trn_tokenized = []
val_tokenized = []

for s in trn_stories:
    tokens = tokenize_string(s, sorted_stoi)
    trn_tokenized.append(tokens)

for s in val_stories:
    tokens = tokenize_string(s, sorted_stoi)
    val_tokenized.append(tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Create tokenized training and validation samples
# Note I dropped the sequence length down from 1024
# to 512 because the vast majority of tokenized stories
# are less than ~500 tokens
seq_len = int(os.environ.get("MINIBPE_SEQ_LEN", "512"))
if seq_len < 1:
    raise ValueError("MINIBPE_SEQ_LEN must be positive.")

trn_tokenized = [s for s in trn_tokenized if len(s) <= seq_len]
val_tokenized = [s for s in val_tokenized if len(s) <= seq_len]
if not trn_tokenized:
    raise ValueError("No training samples fit within MINIBPE_SEQ_LEN.")
if not val_tokenized:
    raise ValueError("No validation samples fit within MINIBPE_SEQ_LEN.")
trn_tokenized_lens = []
val_tokenized_lens = []

pads = []
for s in trn_tokenized:
    trn_tokenized_lens.append(len(s))

    if seq_len == len(s):
        pads.append([])
    else:
        pads.append([stop_char] * (seq_len - len(s)))

trn_tokenized = [s+p for s, p in zip(trn_tokenized, pads)]

pads = []
for s in val_tokenized:
    val_tokenized_lens.append(len(s))

    if seq_len == len(s):
        pads.append([])
    else:
        pads.append([stop_char] * (seq_len - len(s)))

val_tokenized = [s+p for s, p in zip(val_tokenized, pads)]

for s, l in zip(trn_tokenized, trn_tokenized_lens):
    assert len(s) == seq_len
    assert l <= len(s)

for s, l in zip(val_tokenized, val_tokenized_lens):
    assert len(s) == seq_len
    assert l <= len(s)

def get_batch(indices, tokenized_data, device):
    X, Y = [], []

    for i in indices:
        token_list = tokenized_data[i]

        x = [stoi[t] for t in token_list]
        y = [stoi[t] for t in token_list[1:] + [stop_char]]

        X.append(x)
        Y.append(y)

    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)

    return X, Y

trn_valid_lens = torch.tensor(trn_tokenized_lens, device=device)
val_valid_lens = torch.tensor(val_tokenized_lens, device=device)

trn_valid_lens.shape, val_valid_lens.shape

def check_tensors(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

    import torch.nn.functional as F

class Embedding():
    def __init__(self, vocab_size, embed_dim, g, device):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.emb = torch.randn(self.vocab_size, self.embed_dim, generator=g, device=device)

    # X is (batch, seq_len)
    def __call__(self, X):

        # returns (batch, seq_len, embed_dim)
        out = F.embedding(X, self.emb)
        check_tensors(out, "embedding")

        return out

    def params(self):
        return [self.emb]

class MultiHeadSelfAttn():
    def __init__(self, d_model, nr_heads, g, device):
        if nr_heads < 1 or d_model % nr_heads != 0:
            raise ValueError("nr_heads must be positive and divide d_model evenly.")
        self.device = device
        self.d_model = d_model
        self.nr_heads = nr_heads
        self.hidden_dim = d_model // nr_heads
        self.mh_proj = torch.nn.init.kaiming_normal_(torch.randn(nr_heads, 3, d_model, self.hidden_dim, generator=g, device=device), nonlinearity='relu')
        self.Wo = torch.nn.init.kaiming_normal_(torch.randn(d_model, d_model, generator=g, device=device), nonlinearity='relu')
        self.attn = None

    # X is (batch, seq_len, d_model)
    # valid_lens is (batch)
    def __call__(self, X, valid_lens):
        assert X.dim() == 3
        seq_len = X.shape[1]

        # (batch, 1, 1, seq_len, d_model)
        X = X.unsqueeze(dim=1).unsqueeze(dim=1)

        # (batch, nr_heads, 3, seq_len, d_model)
        X = X.expand(-1, self.nr_heads, 3, -1, -1)

        # (batch, 1, 1)
        valid_lens = valid_lens.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # (batch, nr_heads, 1)
        valid_lens = valid_lens.expand(-1, self.nr_heads, -1)
        valid_lens = valid_lens.reshape(-1, 1)

        # (batch, nr_heads, 3, d_model, hidden_dim)
        mh_proj = self.mh_proj.unsqueeze(dim=0).expand(X.shape[0], -1, -1, -1, -1)

        # flatten outer dims for batched multiply: (batch*nr_heads*3, _, _)
        # Since this is self attention, the input is reused across each
        # head's linear projection matrix (each head attends to the same input).
        X = X.reshape(-1, seq_len, self.d_model)
        mh_proj = mh_proj.reshape(-1, self.d_model, self.hidden_dim)

        # (batch*nr_heads*3, seq_len, hidden_dim)
        X = torch.bmm(X, mh_proj)
        check_tensors(X, "mhattn_first_bmm")

        X = X.reshape(-1, self.nr_heads, 3, seq_len, self.hidden_dim)

        # Pluck out Q, K, V
        Q = X[:, :, 0, :, :]
        K = X[:, :, 1, :, :]
        V = X[:, :, 2, :, :]

        Q = Q.reshape(-1, seq_len, self.hidden_dim)

        KT = K.transpose(2, 3)
        KT = KT.reshape(-1, self.hidden_dim, seq_len)

        # (batch*nr_heads, seq_len, seq_len)
        check_tensors(Q, "mhattn_Q")
        check_tensors(KT, "mhattn_KT")

        scaled_dp = torch.bmm(Q, KT) / math.sqrt(self.hidden_dim)
        check_tensors(scaled_dp, "mhattn_scaled_dp")

        # (batch*nr_heads, seq_len, seq_len)
        attn = self.masked_softmax(scaled_dp, valid_lens, self.device)
        self.attn = attn
        check_tensors(attn, "mhattn_masked_attn")
        V = V.reshape(-1, seq_len, self.hidden_dim)

        out = torch.bmm(attn, V)
        check_tensors(out, "mhattn_third_bmm")
        out = out.reshape(-1, self.nr_heads, seq_len, self.hidden_dim)

        # gives us tuple of nr_heads tensors, each with shape (batch, seq_len, hidden_dim)
        out = out.unbind(dim=1)

        # (batch, seq_len, hidden_dim * nr_heads) = (batch, seq_len, d_model)
        out = torch.cat(out, dim=-1)

        # Final linear projection (batch, seq_len, d_model)
        return out @ self.Wo

    def params(self):
        return [self.mh_proj, self.Wo]

    # X is (batch*nr_heads, seq_len, seq_len)
    # valid_lens is (batch*nr_heads, 1). Each entry should be <= seq_len
    def masked_softmax(self, X, valid_lens, device):
        assert X.dim() == 3
        assert valid_lens.dim() == 2

        seq_len = X.shape[1]
        assert torch.all(valid_lens <= seq_len)

        # (1, 1, seq_len) -> ranging from 0 to seq_len - 1
        idxs = torch.arange(0, seq_len, device=device).unsqueeze(dim=0).unsqueeze(dim=0)

        # (batch*nr_heads, seq_len, seq_len)
        idxs = idxs.expand(X.shape[0], seq_len, -1)

        # (batch*nr_heads, 1, 1)
        valid_lens = valid_lens.unsqueeze(dim=-1)

        # (batch*nr_heads, seq_len, seq_len)
        valid_lens = valid_lens.expand(X.shape[0], seq_len, seq_len)

        mask = (idxs < valid_lens).float()
        idxs = idxs.transpose(1, 2)
        mask = mask * (idxs < valid_lens).float()

        # Now we create per-token mask. Since we are doing autoregression,
        # we have to ensure that token at position i can not look ahead
        # at any tokens after i. The first token only has itself. The second
        # token has itself and the first token, and so on.
        valid_lens = idxs + 1
        idxs = idxs.transpose(1, 2)

        # take intersection of padding and token masks
        tok_mask = (idxs < valid_lens).float()
        mask = tok_mask * mask

        # invert mask so we can add large negative value to cleared indices
        X = X.masked_fill(~(mask.bool()), float('-inf'))

        # Get attn/probs over the last dimension
        result = F.softmax(X, dim=-1)
        return torch.nan_to_num(result, nan=0.0)

class LayerNorm():
    def __init__(self, d_model, device):
        self.gain = torch.ones(d_model, device=device)
        self.bias = torch.zeros(d_model, device=device)

    # X is (batch, seq_len, d_model)
    def __call__(self, X):
        assert X.dim() == 3

        # (batch, seq_len, 1)
        mean = torch.mean(X, dim=2, keepdim=True)

        # (batch, seq_len, 1)
        std = torch.sqrt(torch.mean((X - mean)**2, dim=2, keepdim=True) + 1e-5)
        check_tensors(std, "layer_norm_std")

        # (batch, seq_len, d_model)
        return self.gain / std * (X - mean) + self.bias

    def params(self):
        return [self.gain, self.bias]

class PositionalEncoding():
    def __init__(self, max_seq_len, d_model, g, device):
        self.device = device
        self.emb = torch.randn(max_seq_len, d_model, generator=g, device=device)

    # X is (batch, seq_len, d_model)
    def __call__(self, x):
        seq_len = x.shape[1]
        assert seq_len <= self.emb.shape[0]
        indices = torch.arange(0, seq_len, device=self.device)
        pe = F.embedding(indices, self.emb)
        check_tensors(pe, "pe")

        return x + pe

    def params(self):
        return [self.emb]

class FFN():
    def __init__(self, d_model, d_ff, g, device):
        self.W_i = torch.nn.init.kaiming_normal_(torch.randn(d_model, d_ff, generator=g, device=device), nonlinearity='relu')
        self.W_o = torch.randn(d_ff, d_model, generator=g, device=device)
        self.b_i = torch.zeros(d_ff, device=device)
        self.b_o = torch.zeros(d_model, device=device)

    # X is (batch, seq_len, d_model)
    def __call__(self, X):
        # (batch, seq_len, d_ff)
        X = torch.relu(X @ self.W_i + self.b_i)

        # (batch, seq_len, d_model)
        return X @ self.W_o + self.b_o

    def params(self):
        return [self.W_i, self.W_o, self.b_i, self.b_o]

class Linear():
    def __init__(self, d_model, vocab_size, g, device):
        self.W = torch.randn(d_model, vocab_size, generator=g, device=device) * 0.1 # make initial logits less confident
        self.b = torch.zeros(vocab_size, device=device)

    # X is (batch, seq_len, d_model)
    def __call__(self, X):
        # (batch, seq_len, vocab_size)
        return X @ self.W + self.b

    def params(self):
        return [self.W, self.b]

class TransformerBlock():
    def __init__(self, d_model, d_ff, nr_heads, g, device):
        self.mh_attn = MultiHeadSelfAttn(d_model, nr_heads, g, device)
        self.layer_norm1 = LayerNorm(d_model, device)
        self.ffn = FFN(d_model, d_ff, g, device)
        self.layer_norm2 = LayerNorm(d_model, device)
        self.device = device
        self.g = g
        self.attn = None

    # X is (batch, seq_len, d_model)
    # X is the output of the sum between the embedding and positional encoding
    #
    # valid_lens is (batch)
    def __call__(self, X, valid_lens, dropout_prob):
        dropout = Dropout(self.g, self.device)

        res = X
        X = dropout(X, dropout_prob)
        X = self.mh_attn(X, valid_lens)
        X = dropout(X, dropout_prob)
        X = X + res
        X = dropout(X, dropout_prob)
        X = self.layer_norm1(X)

        res = X
        X = self.ffn(X)
        X = X + res
        X = dropout(X, dropout_prob)
        X = self.layer_norm2(X)
        self.attn = self.mh_attn.attn

        return X

    def params(self):
        return self.mh_attn.params() + self.layer_norm1.params() + self.ffn.params() + self.layer_norm2.params()

class Dropout():
    def __init__(self, g, device):
        self.generator = g
        self.device = device

    def __call__(self, x, dropout_prob):
        if not 0 <= dropout_prob < 1:
            raise ValueError("dropout_prob must be in the range [0, 1).")
        if dropout_prob == 0:
            return x

        mask = (torch.rand(x.shape, generator=self.generator, device=self.device) > dropout_prob).float()

        return x * mask / (1 - dropout_prob)

    def params(self):
        return []

vocab_size = len(stoi)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = torch.Generator(device=device).manual_seed(42)
d_model = int(os.environ.get("MINIBPE_D_MODEL", "1024"))
nr_heads = int(os.environ.get("MINIBPE_NUM_HEADS", "8"))
d_ff = int(os.environ.get("MINIBPE_D_FF", "2048"))
device

import torch.optim as optim
import torch.optim.lr_scheduler as lrs

model = [
    Embedding(vocab_size, d_model, g, device),
    PositionalEncoding(seq_len, d_model, g, device),
    TransformerBlock(d_model, d_ff, nr_heads, g, device), # call requires valid_lens
    TransformerBlock(d_model, d_ff, nr_heads, g, device),
    Linear(d_model, vocab_size, g, device)
]


def forward_model(model, token_ids, valid_lens, dropout_prob=0.0):
    x = token_ids
    for layer in model:
        if isinstance(layer, TransformerBlock):
            x = layer(x, valid_lens, dropout_prob)
        else:
            x = layer(x)
    return x


params = [p for layer in model for p in layer.params()]

for p in params:
    p.requires_grad = True

print(f"Transformer has {sum([p.nelement() for p in params])} params")

batch_size = int(os.environ.get("MINIBPE_BATCH_SIZE", "64"))
max_step = int(os.environ.get("MINIBPE_MAX_STEPS", "200000"))
if batch_size < 1:
    raise ValueError("MINIBPE_BATCH_SIZE must be positive.")
if max_step < 0:
    raise ValueError("MINIBPE_MAX_STEPS cannot be negative.")
cosine_start = max_step * 2 // 100
lr = 3e-4
dropout_prob = 0.1

optimizer = optim.Adam(params, lr)
if max_step > 0:
    warmup_steps = max(1, cosine_start)

    def lr_factor(step):
        if step < warmup_steps:
            progress = step / warmup_steps
            return (1 / 1000) + (1 - 1 / 1000) * progress

        progress = (step - warmup_steps) / max(1, max_step - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    lr_sched = lrs.LambdaLR(optimizer, lr_factor)
else:
    lr_sched = None

print_step_mod = max(1, max_step // 10)

trn_loss = []
val_loss = []
stop_id = stoi[stop_char]

for i in range(max_step):
    ix = torch.randint(0, len(trn_tokenized), (batch_size,), generator=g, device=device)
    X, Y = get_batch(ix.tolist(), trn_tokenized, device)
    X_valid_lens = trn_valid_lens[ix]

    X = forward_model(model, X, X_valid_lens, dropout_prob)

    # X is (batch, seq_len, vocab_size)
    # Y is (batch, seq_len)
    # mask=T iff the corresponding token is not padding. We don't do a manual reduction (mean) after
    # we've masked the padding tokens.
    has_stop = (Y == stop_id).any(dim=1)
    first_stop = (Y == stop_id).int().argmax(dim=1)
    first_stop[~has_stop] = seq_len - 1
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (positions <= first_stop.unsqueeze(1)).view(-1)
    loss = (F.cross_entropy(X.reshape(-1, vocab_size), Y.view(-1), reduction='none') * mask).sum() / mask.sum()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trn_loss.append(loss.item())
    optimizer.step()

    with torch.no_grad():
        ix = torch.randint(0, len(val_tokenized), (batch_size,), generator=g, device=device)
        X, Y = get_batch(ix.tolist(), val_tokenized, device)
        X_valid_lens = val_valid_lens[ix]

        X = forward_model(model, X, X_valid_lens)

        has_stop = (Y == stop_id).any(dim=1)
        first_stop = (Y == stop_id).int().argmax(dim=1)
        first_stop[~has_stop] = seq_len - 1
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = (positions <= first_stop.unsqueeze(1)).view(-1)
        loss = (F.cross_entropy(X.reshape(-1, vocab_size), Y.view(-1), reduction='none') * mask).sum() / mask.sum()
        val_loss.append(loss.item())

    lr_sched.step()

    if i % print_step_mod == 0:
        print(f"{i:7d} / {max_step}: trn_loss={trn_loss[-1]:4f} val_loss={val_loss[-1]:4f}")

def sequence_nll(device, model, tokens, stoi):
    if len(tokens) < 2:
        return 0.0

    inputs = torch.tensor(
        [[stoi[token] for token in tokens[:-1]]], device=device
    )
    targets = torch.tensor(
        [stoi[token] for token in tokens[1:]], device=device
    )
    valid_lens = torch.tensor([inputs.shape[1]], device=device)
    logits = forward_model(model, inputs, valid_lens)
    return F.cross_entropy(logits[0], targets, reduction="sum").item()


def perplexity(device, model, test_strs, stoi):
    with torch.no_grad():
        total_nll = 0.0
        total_tokens = 0

        for s in test_strs:
            tokens = tokenize_string(s, sorted_stoi)
            if not 2 <= len(tokens) <= seq_len:
                continue

            total_nll += sequence_nll(device, model, tokens, stoi)
            total_tokens += len(tokens) - 1

        if total_tokens == 0:
            raise ValueError("No test sequences contain at least two usable tokens.")

        result = math.exp(total_nll / total_tokens)
        print(f"Perplexity: {result}")
        return result


def sample_story(device, model, temp, top_k, prompt="Once upon a time "):
    if temp <= 0:
        raise ValueError("temp must be positive.")
    if not 1 <= top_k <= len(stoi):
        raise ValueError(f"top_k must be between 1 and {len(stoi)}.")

    tokens = tokenize_string(prompt, sorted_stoi)
    if not tokens:
        raise ValueError("prompt must contain at least one token.")

    token_ids = [stoi[token] for token in tokens]
    with torch.no_grad():
        while len(token_ids) < seq_len:
            x = torch.tensor([token_ids], device=device)
            valid_lens = torch.tensor([len(token_ids)], device=device)
            logits = forward_model(model, x, valid_lens)[0, -1, :]
            top_logits, indices = torch.topk(logits, top_k)
            probs = F.softmax(top_logits / temp, dim=-1)
            sampled_index = torch.multinomial(probs, num_samples=1).item()
            token_id = indices[sampled_index].item()

            if token_id == stop_id:
                break
            token_ids.append(token_id)

    story = "".join(itos[token_id] for token_id in token_ids)
    print(f"Story time: {story}")
    return story


def bpc(device, model, test_strs, stoi, tokenized=True):
    with torch.no_grad():
        total_nll = 0.0
        total_chars = 0

        for s in test_strs:
            tokens = tokenize_string(s, sorted_stoi) if tokenized else list(s)
            if not 2 <= len(tokens) <= seq_len:
                continue

            total_nll += sequence_nll(device, model, tokens, stoi)
            total_chars += sum(len(token) for token in tokens[1:])

        if total_chars == 0:
            raise ValueError("No test sequences contain usable characters.")

        result = total_nll / (total_chars * math.log(2))
        print(f"BPC: {result:.2f}")
        return result
