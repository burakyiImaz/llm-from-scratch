#!/usr/bin/env python3
"""
Büyük harfle başlayan cümleler için tokenizer testi
"""
import json

# Tokenizer dosyasını yükle
with open("tokenizer_morfeme.json", "r", encoding='utf-8') as f:
    tokenizer_data = json.load(f)

# Tüm kategorileri birleştir
vocab = {}
for category, tokens in tokenizer_data.items():
    if isinstance(tokens, dict):
        vocab.update(tokens)

reverse_vocab = {v: k for k, v in vocab.items()}

def simple_encode(text):
    """Basit enkodlama fonksiyonu - PyTorch olmadan test etmek için"""
    tokens = []
    
    # Cümle büyük harfle başlıyorsa, <büyük_harf> tokeni ekle
    if text and text[0].isupper() and "<büyük_harf>" in vocab:
        tokens.append(vocab["<büyük_harf>"])
    
    for word in text.split():
        i = 0
        while i < len(word):
            found_match = False
            for j in range(len(word), i, -1):
                sub_word = word[i:j]
                if sub_word in vocab:
                    tokens.append(vocab[sub_word])
                    i = j
                    found_match = True
                    break
            if not found_match:
                tokens.append(vocab["<unk>"])
                i += 1
        tokens.append(vocab[" "])
    
    if not text.endswith(" "):
        tokens.pop()
    
    return tokens

def decode(token_ids):
    """Token ID'lerini metne çevir"""
    text = ""
    for id in token_ids:
        text += reverse_vocab.get(id, "?")
    return text

# Test
print("=" * 60)
print("TOKENIZER TESTI - BÜYÜK HARFLE BAŞLAYAN CÜMLELER")
print("=" * 60)

test_cases = [
    ("Merhaba dünya", True),        # Büyük harfle başlar
    ("merhaba dünya", False),       # Küçük harfle başlar
    ("Python programlama", True),   # Büyük harfle başlar
    ("kod yazıyorum", False),       # Küçük harfle başlar
]

for sentence, should_have_uppercase in test_cases:
    print(f"\nCümle: '{sentence}'")
    print(f"Büyük harfle başlar: {sentence[0].isupper()}")
    
    tokens = simple_encode(sentence)
    decoded = decode(tokens)
    
    print(f"Token sayısı: {len(tokens)}")
    
    has_uppercase_token = vocab["<büyük_harf>"] in tokens
    print(f"<büyük_harf> tokeni var: {has_uppercase_token}")
    
    if has_uppercase_token == should_have_uppercase:
        print("✓ TEST GEÇTI")
    else:
        print("✗ TEST BAŞARISIZ")
    
    print(f"İlk 5 token ID: {tokens[:5]}")
    token_names = [reverse_vocab.get(t, "?") for t in tokens[:5]]
    print(f"Token adları: {token_names}")

print("\n" + "=" * 60)
print("✓ TÜM TESTLER TAMAMLANDI")
print("=" * 60)
