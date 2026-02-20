import json
import re
import unicodedata
from collections import Counter, defaultdict


class TurkishBPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]



    def normalize(self, text):
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        return text



    def train(self, file_path):
        print("Metin okunuyor...")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.normalize(text)
        words = re.findall(r"\S+", text)

        corpus = [list(word) + ["</w>"] for word in words]
        vocab = Counter(tuple(word) for word in corpus)

        print("BPE eğitimi başlıyor...")

        # başlangıç token seti (karakterler)
        tokens = set()
        for word in vocab:
            for t in word:
                tokens.add(t)

        base_vocab_size = len(tokens) + len(self.special_tokens)
        merges_needed = self.vocab_size - base_vocab_size

        if merges_needed <= 0:
            raise ValueError("Vocab size çok küçük!")

        for i in range(merges_needed):

            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)

            vocab = self.merge_vocab(best, vocab)

            if (i + 1) % 500 == 0:
                print(f"Merge: {i+1}/{merges_needed}")

        self.build_vocab(vocab)

        print("Eğitim tamamlandı.")
        print("Toplam vocab:", len(self.vocab))


    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs



    
    def merge_vocab(self, pair, vocab):
        new_vocab = {}

        for word, freq in vocab.items():
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_vocab[tuple(new_word)] = freq

        return new_vocab



    def build_vocab(self, vocab):
        tokens = set()
        for word in vocab:
            for token in word:
                tokens.add(token)

        all_tokens = self.special_tokens + sorted(tokens)

        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()}



    def encode(self, text, add_special_tokens=False):
        text = self.normalize(text)
        words = re.findall(r"\S+", text)

        output_tokens = []

        for word in words:
            word_tokens = list(word) + ["</w>"]

            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i+1]) == pair:
                        word_tokens[i:i+2] = ["".join(pair)]
                    else:
                        i += 1

            output_tokens.extend(word_tokens)

        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in output_tokens]

        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]

        return ids



    def decode(self, ids, skip_special_tokens=True):

        tokens = []
        for i in ids:
            tok = self.reverse_vocab.get(i, "<unk>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        text = "".join(tokens)
        text = text.replace("</w>", " ")

        return text.strip()



    def save_tokenizer_json(self, path):

        tokenizer_json = {
            "version": "1.0",
            "model": {
                "type": "custom_bpe",
                "vocab_size": len(self.vocab),
                "vocab": self.vocab,
                "merges": [" ".join(pair) for pair in self.merges]
            },
            "normalization": {
                "type": "NFKC+lowercase"
            },
            "special_tokens": {
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<bos>",
                "eos_token": "<eos>"
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

        print("tokenizer.json oluşturuldu.")

    def save_hf_compatible(self, path):
        hf_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": self.vocab[token],
                    "content": token,
                    "special": True
                }
                for token in self.special_tokens
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFKC"},
                    {"type": "Lowercase"}
                ]
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<unk>",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": self.vocab,
                "merges": [" ".join(pair) for pair in self.merges]
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(hf_json, f, ensure_ascii=False, indent=2)

        print("HF uyumlu tokenizer.json oluşturuldu.")
