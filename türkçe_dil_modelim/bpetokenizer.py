import json
import re
import unicodedata
from collections import Counter, defaultdict


class TurkishBPETokenizer:
    def __init__(self, vocab_size=24000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []
        self.merge_ranks = {}   # encode hızlandırma için
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)




    def normalize(self, text):
        text = unicodedata.normalize("NFKC", text)
        return text.lower()



    def train(self, file_path):
        print("Metin okunuyor...")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.normalize(text)
        words = self.pattern.findall(text)

        corpus = [tuple(list(word) + ["</w>"]) for word in words]
        vocab = Counter(corpus)

        print("BPE eğitimi başlıyor...")

        tokens = set()
        for word in vocab:
            tokens.update(word)

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
        self.build_merge_ranks()

        print("Eğitim tamamlandı.")
        print("Toplam vocab:", len(self.vocab))


    def get_stats(self, vocab):
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            prev_char = word[0]
            for char in word[1:]:
                pairs[(prev_char, char)] += freq
                prev_char = char

        return pairs



    def merge_vocab(self, pair, vocab):
        new_vocab = {}

        bigram = pair
        replacement = "".join(pair)

        for word, freq in vocab.items():
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)

            # BUG FIX: freq overwrite engellendi
            new_vocab[new_word] = new_vocab.get(new_word, 0) + freq

        return new_vocab


    def build_vocab(self, vocab):
        tokens = set()
        for word in vocab:
            tokens.update(word)

        all_tokens = self.special_tokens + sorted(tokens)

        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()}

        self.unk_id = self.vocab["<unk>"]
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]


    def build_merge_ranks(self):
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}


    def encode(self, text, add_special_tokens=False):
        text = self.normalize(text)
        words = self.pattern.findall(text)

        output_tokens = []

        for word in words:
            word_tokens = list(word) + ["</w>"]

            while True:
                pairs = [(word_tokens[i], word_tokens[i+1])
                         for i in range(len(word_tokens)-1)]

                ranked = [
                    (self.merge_ranks.get(pair, float("inf")), pair, i)
                    for i, pair in enumerate(pairs)
                ]

                if not ranked:
                    break

                best_rank, best_pair, best_index = min(ranked)

                if best_rank == float("inf"):
                    break

                word_tokens[best_index:best_index+2] = [
                    "".join(best_pair)
                ]

            output_tokens.extend(word_tokens)

        ids = [self.vocab.get(t, self.unk_id) for t in output_tokens]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

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

    def load_tokenizer_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["model"]["vocab"]
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(m.split()) for m in data["model"]["merges"]]

        self.build_merge_ranks()

        self.unk_id = self.vocab["<unk>"]
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]

        print("Tokenizer başarıyla yüklendi.")