import json
import torch


class Tokenizer:
    def __init__(self, vocab_file, encoding="utf-8", auto_learn=True):
        self.auto_learn = auto_learn

        with open(vocab_file, "r", encoding=encoding) as f:
            vocab_data = json.load(f)

        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_categories = {}

        current_id = 0

        # JSON kategorilerini sırayla ve ÇAKIŞMASIZ yükle
        for category, tokens in vocab_data.items():
            self.vocab_categories[category] = {}

            for token in tokens.keys():
                if token not in self.vocab:
                    self.vocab[token] = current_id
                    self.reverse_vocab[current_id] = token
                    self.vocab_categories[category][token] = current_id
                    current_id += 1

        self.next_token_id = current_id

        # Özel token ID'leri
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        self.start_id = self.vocab["<başla>"]
        self.end_id = self.vocab["<bitiş>"]
        self.uppercase_id = self.vocab["<büyük_harf>"]
        self.space_id = self.vocab[" "]

    def get_vocab_size(self):
        return self.next_token_id

    
    def encode_batch(self, texts,context_length):
        sentence_tokens= []
        
        for text in texts:
            tokens= self.encode(text).tolist()
            if len(tokens)>context_length:
                tokens= tokens[:context_length]
            else:
                tokens= tokens + [self.pad_id]*(context_length-len(tokens))
            sentence_tokens.append(tokens)
        return torch.tensor(sentence_tokens)
    
    
    
    
    
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
