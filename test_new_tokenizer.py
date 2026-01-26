#!/usr/bin/env python3
"""
Yeni MasterTokenizer sınıfı için kapsamlı test
"""
from master_tokenizer import MasterTokenizer

# Tokenizer yükle
print("=" * 70)
print("MASTERTOKENIZER - KAPSAMLI TEST")
print("=" * 70)

tokenizer = MasterTokenizer("tokenizer_morfeme.json")

print(f"\n📊 Tokenizer Bilgileri:")
print(f"   - Toplam token sayısı: {tokenizer.get_vocab_size()}")
print(f"   - Pad ID: {tokenizer.pad_id}")
print(f"   - Unk ID: {tokenizer.unk_id}")
print(f"   - Uppercase ID: {tokenizer.uppercase_id}")

# Test 1: Temel encode
print("\n" + "=" * 70)
print("TEST 1: Temel Encoding")
print("=" * 70)

test_sentences = [
    "Merhaba dünya",
    "merhaba dünya",
    "Python programlama",
    "kod yazıyorum"
]

for sentence in test_sentences:
    tokens = tokenizer.encode(sentence)
    token_names = tokenizer.tokenize(sentence)
    
    print(f"\n📝 Cümle: '{sentence}'")
    print(f"   Token sayısı: {len(tokens)}")
    print(f"   Token adları: {token_names[:5]}...")
    if tokenizer.uppercase_id in tokens:
        print(f"   ✓ Büyük harf tokeni EKLENDİ")
    else:
        print(f"   ✗ Büyük harf tokeni yok")

# Test 2: Encode batch
print("\n" + "=" * 70)
print("TEST 2: Batch Encoding (Context Length: 20)")
print("=" * 70)

batch_texts = ["Merhaba", "Python programlama yapıyorum"]
batch_tokens = tokenizer.encode_batch(batch_texts, context_length=20)

print(f"\n📦 Batch shape: {batch_tokens.shape}")
for i, text in enumerate(batch_texts):
    print(f"   {i+1}. '{text}' -> {batch_tokens[i].tolist()}")

# Test 3: Special tokens ile encoding
print("\n" + "=" * 70)
print("TEST 3: Special Tokens İle Encoding")
print("=" * 70)

sentence = "Merhaba dünya"
tokens_normal = tokenizer.encode(sentence, add_special_tokens=False)
tokens_special = tokenizer.encode(sentence, add_special_tokens=True)

print(f"\n📝 Cümle: '{sentence}'")
print(f"   Normal encoding: {tokens_normal.tolist()}")
print(f"   Special tokens ile: {tokens_special.tolist()}")
print(f"   Fark: {len(tokens_special) - len(tokens_normal)} token")

# Test 4: Decoding
print("\n" + "=" * 70)
print("TEST 4: Decoding (Token ID'lerini Metne Çevir)")
print("=" * 70)

original = "Merhaba dünya"
encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded)

print(f"\n📝 Orijinal: '{original}'")
print(f"   Encoded: {encoded.tolist()}")
print(f"   Decoded: '{decoded}'")

# Test 5: Token bilgisi
print("\n" + "=" * 70)
print("TEST 5: Token Bilgisi Sorgusu")
print("=" * 70)

test_tokens = ["<büyük_harf>", "Merhaba", "<pad>", "<unk>", "python"]
print()
for token in test_tokens:
    token_id = tokenizer.get_token_id(token)
    token_name = tokenizer.get_token_name(token_id)
    print(f"   '{token}' -> ID: {token_id}, Geri: '{token_name}'")

# Test 6: Uppercase token control
print("\n" + "=" * 70)
print("TEST 6: Büyük Harf Token Kontrolü")
print("=" * 70)

uppercase_test = [
    ("Türkiye", True),
    ("türkiye", False),
    ("İstanbul", True),
    ("istanbul", False),
    ("Python", True),
    ("python", False),
]

print()
for text, should_have_uppercase in uppercase_test:
    tokens = tokenizer.encode(text)
    has_uppercase = tokenizer.uppercase_id in tokens
    status = "✓" if has_uppercase == should_have_uppercase else "✗"
    print(f"   {status} '{text}' -> Büyük harf: {has_uppercase} (beklenen: {should_have_uppercase})")

# Test 7: Padding test
print("\n" + "=" * 70)
print("TEST 7: Padding Test")
print("=" * 70)

context_len = 15
texts = ["Selam", "Merhaba dünya nasılsın"]
padded = tokenizer.encode_batch(texts, context_length=context_len)

print(f"\n📦 Context Length: {context_len}")
for i, text in enumerate(texts):
    tokens = padded[i].tolist()
    pad_count = sum(1 for t in tokens if t == tokenizer.pad_id)
    print(f"   '{text}'")
    print(f"      -> Token listesi: {tokens}")
    print(f"      -> Padding sayısı: {pad_count}")

print("\n" + "=" * 70)
print("✓ TÜM TESTLER TAMAMLANDI")
print("=" * 70)
