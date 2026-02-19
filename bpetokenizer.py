import json
import re
import unicodedata
from collections import Counter, defaultdict
from tqdm import tqdm


class TurkishBPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.vocab = {}
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

        # karakter seviyesinde başlat
        corpus = [list(word) + ["</w>"] for word in words]

        vocab = Counter(tuple(word) for word in corpus)

        print("BPE eğitimi başlıyor...")

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab = self.merge_vocab(best, vocab)

            if len(self.merges) % 100 == 0:
                print(f"Merge sayısı: {len(self.merges)}")

        self.build_vocab(vocab)
        print("Eğitim tamamlandı.")

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word, freq in vocab.items():
            word_str = " ".join(word)
            new_word = pattern.sub("".join(pair), word_str)
            new_vocab[tuple(new_word.split(" "))] = freq

        return new_vocab

    def build_vocab(self, vocab):
        tokens = set()
        for word in vocab:
            for token in word:
                tokens.add(token)

        all_tokens = self.special_tokens + sorted(tokens)

        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()}


    def encode(self, text):
        text = self.normalize(text)
        words = re.findall(r"\S+", text)

        tokens = []
        for word in words:
            word_tokens = list(word) + ["</w>"]
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i+1]) == pair:
                        word_tokens[i:i+2] = ["".join(pair)]
                    else:
                        i += 1
            tokens.extend(word_tokens)

        return [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]



    def decode(self, ids):
        tokens = [self.reverse_vocab[i] for i in ids]
        text = "".join(tokens)
        return text.replace("</w>", " ")



    def save_tokenizer_json(self, path):
        tokenizer_json = {
            "model": {
                "type": "BPE",
                "vocab": self.vocab,
                "merges": [" ".join(pair) for pair in self.merges]
            },
            "special_tokens": self.special_tokens
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

        print("tokenizer.json oluşturuldu.")
