---

# Türkçe Morfem-Tabanlı Tokenizer

Bu repository, **Türkçe için özel olarak tasarlanmış**,
**JSON tabanlı**, **subword (alt-birim) odaklı** ve **dinamik öğrenme destekli** bir tokenizer implementasyonu içerir.

Tokenizer, özellikle **Türkçenin eklemeli (agglutinative) yapısını** dikkate alarak geliştirilmiştir ve bir Transformer / LLM pipeline’ında doğrudan kullanılabilir.

---

## 📌 Temel Özellikler

* 📁 **JSON tabanlı vocab**
* 🧩 **Subword (alt-parça) tokenization**
* 🔁 **Greedy (en uzun eşleşme) algoritması**
* 🧠 **Dinamik vocab genişletme (auto-learn)**
* 🏷️ **Kategorilere ayrılmış token yapısı**
* 🔤 **Büyük harf bilgisi için özel token**
* 🧪 **Batch encoding + padding desteği**
* 🔄 **Encode / Decode fonksiyonları**

---

## 📂 Vocab Yapısı

Tokenizer, vocab’ı **kategorilere ayrılmış bir JSON dosyasından** yükler.

Örnek yapı:

```json
{
  "özel_tokenler": {
    "<pad>": 0,
    "<unk>": 1,
    "<başla>": 2,
    "<bitiş>": 3,
    "<büyük_harf>": 4
  },
  "kelimeler": {
    "ankara": 10,
    "türkiye": 11
  },
  "ekler": {
    "ler": 100,
    "lar": 101,
    "de": 102,
    "da": 103
  },
  "karakterler": {
    "a": 300,
    "b": 301
  }
}
```

Tokenizer bu yapıyı:

* Tek bir `vocab` sözlüğünde birleştirir
* Aynı zamanda **kategori bilgisini** korur

---

## ⚙️ Sınıf: `Tokenizer`

### Başlatma

```python
tokenizer = Tokenizer(
    vocab_file="vocab.json",
    auto_learn=True
)
```

**Parametreler:**

* `vocab_file`: Tokenlerin bulunduğu JSON dosyası
* `auto_learn`: Vocab’da olmayan karakterleri otomatik ekler

---

## 🔢 Encode (Metin → Token ID)

```python
ids = tokenizer.encode(
    "Ankara Türkiye'dedir",
    add_uppercase_token=True,
    add_special_tokens=True
)
```

### Encode sırasında yapılan işlemler:

1. Metin temizlenir (`strip`)
2. İsteğe bağlı olarak:

   * `<başla>` tokeni eklenir
   * Büyük harfle başlıyorsa `<büyük_harf>` eklenir
3. Metin kelimelere ayrılır
4. Her kelime **greedy subword tokenization** ile parçalanır
5. Kelimeler arasına **space token** eklenir
6. En sona isteğe bağlı `<bitiş>` tokeni eklenir

---

## 🧩 Subword Tokenization (Greedy)

Tokenizer, her kelimeyi **en uzun parçadan başlayarak** vocab’da arar:

Örnek:

```
"kitaplardan"
→ kitap + lar + dan
```

Algoritma:

* En uzun eşleşme aranır
* Bulunamazsa karakter seviyesine düşülür
* Karakter de yoksa:

  * `auto_learn=True` ise vocab’a eklenir
  * Aksi halde `<unk>` kullanılır

---

## 📦 Batch Encode + Padding

```python
batch = tokenizer.encode_batch(
    texts=["Merhaba dünya", "Ankara"],
    context_length=16
)
```

* Her cümle encode edilir
* Uzunsa **truncate**
* Kısaysa `<pad>` ile doldurulur
* Çıktı: `(batch_size, context_length)` tensor

---

## 🔁 Decode (Token ID → Metin)

```python
text = tokenizer.decode(ids)
```

* Token ID’ler tekrar string’e çevrilir
* Varsayılan olarak:

  * `<pad>`, `<başla>`, `<bitiş>`, `<unk>`, `<büyük_harf>` **çıkarılır**

---

## 🧪 Debug Amaçlı Token Gösterimi

```python
tokenizer.tokenize("Ankara")
```

Çıktı:

```python
["<büyük_harf>", "ankara"]
```

---

## ➕ Dinamik Token Ekleme

### Tek token ekleme

```python
tokenizer.add_token("den", category="ekler")
```

### Çoklu token ekleme

```python
tokenizer.add_tokens(["miş", "mış"], category="ekler")
```

* Token ID’ler otomatik atanır
* Hem vocab’a hem kategoriye eklenir

---

## 💾 Vocab Kaydetme

```python
tokenizer.save_vocab("yeni_vocab.json")
```

* Güncel vocab
* Kategoriler korunarak JSON’a yazılır

---

## 📊 Öğrenme İstatistikleri

```python
stats = tokenizer.get_learning_stats()
```

Örnek çıktı:

```json
{
  "toplam_token_sayısı": 742,
  "kategori_sayısı": 6,
  "kategoriler": ["kelimeler", "ekler", "karakterler"],
  "sonraki_token_id": 743,
  "auto_learn_aktif": true
}
```

---

## 🎯 Tasarım Amacı

Bu tokenizer:

* Türkçenin **eklemeli yapısına uygun**
* Küçük vocab ile **yüksek kapsama**
* Eğitim sırasında **kendini genişletebilen**
* Transformer tabanlı modellerle **doğrudan uyumlu**

bir yapı sunmayı hedefler.

---

## 📌 Not

Bu implementasyon:

* Greedy subword yaklaşımı kullanır
* Space token açıkça temsil edilir
* Morfolojik analiz **harici** değil, vocab üzerinden yapılır

---
