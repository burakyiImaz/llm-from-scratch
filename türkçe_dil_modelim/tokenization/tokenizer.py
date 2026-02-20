import json
import torch
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab_file, encoding="utf-8", auto_learn=True):
        self.vocab_file = Path(vocab_file)
        self.auto_learn = auto_learn

        # JSON'dan vocab yükle
        if self.vocab_file.exists():
            with open(vocab_file, "r", encoding=encoding) as f:
                vocab_data = json.load(f)
        else:
            vocab_data = {}

        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_categories = {}

        current_id = 0

        # JSON kategorilerini sırayla yükle
        if "model" in vocab_data and "vocab" in vocab_data["model"]:
            vocab_dict = vocab_data["model"]["vocab"]
            self.vocab_categories["main"] = {}
            for token, token_id in vocab_dict.items():
                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token
                self.vocab_categories["main"][token] = token_id
            current_id = max(vocab_dict.values()) + 1
        else:
            current_id = 0


        self.next_token_id = current_id

        special = vocab_data.get("special_tokens", {})
        self.pad_id = self.vocab.get(special.get("pad_token", "<pad>"), self.next_token_id)
        self.unk_id = self.vocab.get(special.get("unk_token", "<unk>"), self.next_token_id)
        self.start_id = self.vocab.get(special.get("bos_token", "<bos>"), self.next_token_id)
        self.end_id = self.vocab.get(special.get("eos_token", "<eos>"), self.next_token_id)
        self.uppercase_id = self.vocab.get("<büyük_harf>", self.unk_id)
        self.space_id = self.vocab.get(" ", self.unk_id)

    def _add_special_token(self, token):
        """Yeni özel token ekler"""
        token_id = self.next_token_id
        self.vocab[token] = token_id
        self.reverse_vocab[token_id] = token
        self.next_token_id += 1
        return token_id

    def save_vocab(self):
        """Vocab’ı JSON’a kaydet"""
        categories = self.vocab_categories.copy()
        # Eklenmiş tokenlar için 'ekstra' kategorisi
        ekstra_tokens = {t: {"id": i} for t, i in self.vocab.items() if all(t not in cat for cat in self.vocab_categories.values())}
        categories["ekstra"] = ekstra_tokens
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)

    def get_vocab_size(self):
        return self.next_token_id

    def encode_batch(self, texts, context_length):
        sentence_tokens = []
        for text in texts:
            tokens = self.encode(text).tolist()
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
            else:
                tokens = tokens + [self.pad_id] * (context_length - len(tokens))
            sentence_tokens.append(tokens)
        return torch.tensor(sentence_tokens, dtype=torch.long)

    def encode(self, text):
        tokens = []

        for word in text.strip().split():
            i = 0
            while i < len(word):
                found = False
                for j in range(len(word), i, -1):
                    sub = word[i:j]
                    if sub in self.vocab:
                        tokens.append(self.vocab[sub])
                        i = j
                        found = True
                        break
                if not found:
                    char = word[i]
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    elif self.auto_learn:
                        self.vocab[char] = self.next_token_id
                        self.reverse_vocab[self.next_token_id] = char
                        tokens.append(self.next_token_id)
                        self.next_token_id += 1
                    else:
                        tokens.append(self.unk_id)
                    i += 1
            tokens.append(self.space_id)

        if tokens and tokens[-1] == self.space_id:
            tokens.pop()

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids, remove_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_ids = {self.pad_id, self.start_id, self.end_id, self.unk_id, self.uppercase_id}
        text = ""
        for token_id in token_ids:
            if remove_special_tokens and token_id in special_ids:
                continue
            token_str = self.reverse_vocab.get(token_id, "")
            text += token_str
        return text

