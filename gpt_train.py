
import torch
import torch.nn as nn
from torch.nn import functional as F
from minbpe.regex import RegexTokenizer  # ИЗМЕНЕНО: импортируем BPE-токенизатор

# hyperparameters
batch_size = 16
block_size = 128
max_iters = 2000
eval_interval = 250
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# ИЗМЕНЕНО: читаем текст один раз
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ИЗМЕНЕНО: сначала делим исходный текст на train и validation
n = int(0.9 * len(text))
train_text = text[:n]
val_text = text[n:]

# ИЗМЕНЕНО: обучаем BPE-токенизатор только на тренировочном тексте
tokenizer = RegexTokenizer()
tokenizer.train(train_text, vocab_size=512)

# ИЗМЕНЕНО: размер словаря теперь берём из BPE-токенизатора
vocab_size = len(tokenizer.vocab)
print("BPE vocab size:", vocab_size)

# ИЗМЕНЕНО: кодируем текст в последовательности BPE token IDs
train_ids = tokenizer.encode_ordinary(train_text)
val_ids = tokenizer.encode_ordinary(val_text)

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)

print("Train tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Пример:", tokenizer.encode_ordinary("Hello world!"))

# ИЗМЕНЕНО: теперь x и y содержат BPE-токены, а не отдельные символы.
# x — входные данные, y — тот же фрагмент, сдвинутый на один токен вперёд.
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # ИЗМЕНЕНО: проверяем, достаточно ли данных для формирования батча
    if len(data) <= block_size:
        raise ValueError(
            f"Недостаточно токенов для {split}: {len(data)}. "
            f"Нужно больше {block_size}."
        )

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


# Это одна голова Self-Attention.
# Она позволяет каждому токену учитывать предыдущие токены контекста
# при формировании собственного представления.
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        # nn.Linear — обучаемое линейное преобразование.
        # У него есть матрица весов, которая изменяется во время обучения.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )  # создали треугольную матрицу из единиц

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        k = self.key(x)       # (B, T, hs)
        q = self.query(x)     # (B, T, hs)

        # compute attention scores ("affinities")
        # ИЗМЕНЕНО: при наших настройках q и k имеют форму (16, 128, 32).
        # k.transpose(-2, -1) превращает (B, T, hs) в (B, hs, T).
        # q @ k.transpose(-2, -1):
        # (B, T, hs) @ (B, hs, T) -> (B, T, T).
        # В нашем случае получается (16, 128, 128).
        # Элемент матрицы показывает оценку внимания одной позиции к другой.

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # k.shape[-1]**-0.5 — масштабирование оценок внимания.
        # Оно помогает стабилизировать значения перед softmax.

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )  # (B, T, T): будущие позиции закрываем

        wei = F.softmax(wei, dim=-1)
        # (B, T, T): позиции будущего получают нулевую вероятность внимания

        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # СПРОСИТЬ ПРО РАЗМЕРНОСТИ:
        v = self.value(x)  # (B, T, hs)

        # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = wei @ v

        return out  # (B, T, hs), при наших настройках (16, 128, 32)


# В нём создаются несколько независимых голов внимания.
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])  # запускаем 4 головы внимания

        # ИЗМЕНЕНО: 4 головы по 32 канала дают 128 каналов.
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ИЗМЕНЕНО: каждая голова выдаёт (B, T, 32).
        # После объединения четырёх голов получаем (B, T, 128).
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Смешиваем информацию от всех голов.
        # Размерность остаётся (B, T, 128).
        out = self.dropout(self.proj(out))

        return out


# FeedForward преобразует представление каждой позиции
# с помощью обычной нейросети.
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            # ИЗМЕНЕНО: увеличиваем размерность с 128 до 512
            nn.Linear(n_embd, 4 * n_embd),

            nn.ReLU(),  # применяем нелинейность

            # ИЗМЕНЕНО: сжимаем обратно с 512 до 128
            nn.Linear(4 * n_embd, n_embd),

            nn.Dropout(dropout),  # рандомно выключаем часть нейронов
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: размерность представления,
        # n_head: количество голов внимания
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    # residual connections: часть данных минует обработчик
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ИЗМЕНЕНО: embedding теперь соответствует BPE-словарю.
        # Каждый token ID преобразуется в обучаемый вектор размерности 128.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position_embedding_table даёт каждой позиции свой обучаемый вектор.
        # Это позволяет модели учитывать положение токена в контексте.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # blocks — Transformer-блоки, через которые проходят представления токенов.
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        # ИЗМЕНЕНО: предсказываем оценки для всех токенов BPE-словаря,
        # а не для отдельных символов.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

    # ИЗМЕНЕНО: исправлен расчёт loss.
    # logits возвращаются в исходной форме (B, T, vocab_size),
    # чтобы estimate_loss() и generate() работали правильно.
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx и targets имеют форму (B, T) и содержат целочисленные token IDs.
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)

        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)     # (B, T, n_embd)
        x = self.ln_f(x)       # (B, T, n_embd)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # ИЗМЕНЕНО: используем отдельные переменные для расчёта loss.
            # Это не меняет форму logits, возвращаемых из метода.
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx имеет форму (B, T) и содержит token IDs.
        for _ in range(max_new_tokens):

            # Оставляем только последние block_size токенов контекста.
            idx_cond = idx[:, -block_size:]

            # Получаем предсказания.
            logits, loss = self(idx_cond)

            # Берём оценки последнего временного шага: (B, vocab_size).
            logits = logits[:, -1, :]

            # Преобразуем оценки в вероятности.
            probs = F.softmax(logits, dim=-1)

            # Выбираем следующий токен случайно согласно вероятностям.
            idx_next = torch.multinomial(probs, num_samples=1)

            # Добавляем новый token ID к последовательности.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Цикл обучения
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)  # вычисляются предсказания и ошибки

    optimizer.zero_grad(set_to_none=True)  # зануляем градиенты

    loss.backward()
    # вычисляет градиенты функции потерь по обучаемым параметрам.

    optimizer.step()


# ИЗМЕНЕНО: сохраняем модель вместе с BPE-токенизатором.
# Старые stoi и itos больше не используются.
checkpoint = {
    'model_state_dict': m.state_dict(),

    'tokenizer_state': {
        'pattern': tokenizer.pattern,
        'merges': tokenizer.merges,
        'vocab': tokenizer.vocab,
        'special_tokens': tokenizer.special_tokens,
    },

    'config': {
        'batch_size': batch_size,
        'block_size': block_size,
        'max_iters': max_iters,
        'eval_interval': eval_interval,
        'learning_rate': learning_rate,
        'eval_iters': eval_iters,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'dropout': dropout,
        'vocab_size': vocab_size,
    },
}

torch.save(checkpoint, 'gpt_checkpoint.pth')
print('Модель сохранена в файл gpt_checkpoint.pth')


# ИЗМЕНЕНО: генерируем текст через BPE-токенизатор.
# Вместо старого decode() используем tokenizer.decode().
m.eval()

# Начинаем генерацию с текстового контекста.
prompt_ids = tokenizer.encode_ordinary("ROMEO:")

# Проверяем, что начальный контекст не пуст.
if not prompt_ids:
    raise ValueError("Начальный текст не был преобразован в токены.")

context = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    generated_ids = m.generate(
        context,
        max_new_tokens=500
    )[0].tolist()

print(tokenizer.decode(generated_ids))
