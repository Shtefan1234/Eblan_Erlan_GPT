"""Запуск обученной модели без переобучения.
Использует gpt_checkpoint.pth, где уже лежат веса + токенизатор.
input.txt для генерации НЕ нужен (оставлен только для дообучения).
"""
import sys
import regex as re
import torch
import torch.nn as nn
from torch.nn import functional as F

from minbpe.regex import RegexTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'gpt_checkpoint.pth'

# 1. Загружаем чекпоинт, достаём конфиг и токенизатор
ckpt = torch.load(ckpt_path, map_location=device)
config = ckpt['config']
block_size = config['block_size']
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
dropout = config['dropout']
vocab_size = config['vocab_size']

print(f"config: {config}")

# 2. Восстанавливаем BPE-токенизатор БЕЗ tokenizer.train() и БЕЗ input.txt
ts = ckpt['tokenizer_state']
tokenizer = RegexTokenizer.__new__(RegexTokenizer)  # пустой объект без __init__
tokenizer.pattern = ts['pattern']
tokenizer.compiled_pattern = re.compile(tokenizer.pattern)
tokenizer.merges = ts['merges']
tokenizer.vocab = ts['vocab']
tokenizer.special_tokens = ts.get('special_tokens', {})
tokenizer.inverse_special_tokens = {v: k for k, v in tokenizer.special_tokens.items()}
print(f"Токенизатор восстановлен из чекпоинта, vocab={len(tokenizer.vocab)}")


# 3. Архитектура — та же, что в gpt_train.py
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 4. Загружаем веса
model = GPTLanguageModel()
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# 5. Генерация
prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 500
prompt_ids = tokenizer.encode_ordinary(prompt)
if not prompt_ids:
    raise ValueError("Промпт превратился в 0 токенов")
context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

with torch.no_grad():
    out_ids = model.generate(context, max_new_tokens=max_new)[0].tolist()

print("----")
print(tokenizer.decode(out_ids))
