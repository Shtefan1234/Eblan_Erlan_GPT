# eblan_gpt — запуск на другом компе

## Что внутри (минимум для инференса)
- `generate.py` — запуск без обучения, грузит веса + токенизатор из чекпоинта
- `gpt_checkpoint.pth` — веса (4.6М) + `tokenizer_state` (pattern, merges 256, vocab 512) + `config`
- `minbpe/` — `base.py`, `regex.py` (+ `basic.py`, `gpt4.py`, `__init__.py`) — нужен `RegexTokenizer`
- `gpt_train.py` — исходник обучения (копия `gpt.py`, для справки/дообучения)
- `input.txt` — для дообучения, для генерации НЕ нужен

## Из токенизации что добавлено
В чекпоинт уже вшит `tokenizer_state`, поэтому `generate.py` токенизатор НЕ обучает:
pattern + merges + vocab + special_tokens восстанавливаются из `gpt_checkpoint.pth`,
`input.txt` читать не надо. Отдельные `.model/.vocab` файлы не нужны.

## Запуск
pip install -r requirements.txt
python generate.py "ROMEO:"
python generate.py "Привет" 300
