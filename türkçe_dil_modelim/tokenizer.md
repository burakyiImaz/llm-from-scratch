# Türkçe Morfem Tabanlı Tokenizer

Bu repository, **Türkçe için özel olarak tasarlanmış, morfoloji farkındalığı olan bir tokenizer** içerir. Proje, eklemeli (agglutinative) bir dil olan Türkçe’nin yapısal özelliklerini doğrudan tokenizasyon aşamasına entegre etmeyi amaçlar.

Tokenizer, özel olarak geliştirilmiş bir **Transformer mimarisi** ile birlikte çalışacak şekilde tasarlanmıştır.

---

## 📌 Temel Motivasyon

Klasik tokenizer yaklaşımları (BPE, WordPiece):

* Türkçe ekleri rastgele böler
* Çok büyük vocabulary üretir
* Dilbilgisel bilgiyi modele bırakır

Örnek:

```
kitaplarımdan
→ kitaplar ##ım ##dan
```

Bu proje ile hedeflenen:

```
kitap + lar + ım + dan
```

şeklinde **dilbilgisel olarak anlamlı** bir tokenizasyon elde etmektir.

---

## 🧠 Tokenizer Tasarım Prensipleri

* Morfem temelli parçalama
* Unigram Language Model ile istatistiksel seçim
* Eklerin fonksiyonel bilgisini modele aktarma
* Eğitim ve inference aşamalarının ayrılması

---

## 🧩 Tokenizer Mimarisi

### Genel Akış

```
Raw Text
   ↓
Normalization
   ↓
Word Segmentation (space token yok)
   ↓
Zemberek Morfem Analizi
   ↓
Unigram LM Token Seçimi
   ↓
Token ID + Token Type ID
```

---

## 1️⃣ Unigram Language Model (Greedy Yerine)

### Amaç

Greedy algoritmalar yerine, **en olası morfem dizisini** seçmek.

### Neden?

Greedy yöntemler yalnızca lokal en uzun eşleşmeye bakar. Unigram LM ise tüm olası tokenizasyonları değerlendirir.

### Örnek Kod

```python
# Unigram LM skor hesaplama (basitleştirilmiş)
def score(tokens, token_probs):
    return sum(token_probs.get(t, -1e9) for t in tokens)
```

---

## 2️⃣ Zemberek ile Otomatik Morfem Bölme

### Amaç

Tokenizer’ın ekleri tahmin etmesi yerine **bilerek ayırması**.

### Kullanım

```python
from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()
analysis = morphology.analyze("kitaplarımdan")

for result in analysis:
    print(result.get_stem(), result.get_suffixes())
```

### Kazanım

* Dilbilgisel doğruluk
* Daha az öğrenme yükü

---

## 3️⃣ Space Token Yerine Pozisyonel Ayrım

### Amaç

Boşluğu vocabulary’den çıkarmak.

### Yaklaşım

* Space ayrı token değildir
* Kelime sınırları pozisyonel embedding ile modellenir

```python
# space token eklenmez
vocab = {"[PAD]": 0, "[UNK]": 1}
```

---

## 4️⃣ Eklerin Token Type ID ile Encode Edilmesi

### Amaç

Token’ın **ne olduğu** bilgisini modele ayrı bir kanal olarak vermek.

### Token Type Şeması

```python
TOKEN_TYPES = {
    "ROOT": 0,
    "PLURAL_SUFFIX": 1,
    "CASE_SUFFIX": 2,
    "POSSESSIVE_SUFFIX": 3,
    "VERB_TENSE": 4
}
```

### Örnek Encoding

```python
tokens = ["kitap", "lar", "ım", "dan"]
token_ids = [1021, 204, 317, 411]
token_type_ids = [0, 1, 3, 2]
```

---

## 5️⃣ Auto-Learn Mekanizması (Sadece Train-Time)

### Amaç

Inference sırasında tokenizer davranışının değişmesini önlemek.

### Mantık

```python
class Tokenizer:
    def __init__(self, train_mode=False):
        self.train_mode = train_mode

    def add_token(self, token):
        if self.train_mode:
            self.vocab[token] = len(self.vocab)
```

---

## 🔒 Deterministik Inference

* Vocabulary inference sırasında sabittir
* Embedding uyumsuzluğu oluşmaz
* Sonuçlar reproducible’dır

---

## 📊 Klasik Tokenizer Karşılaştırması

| Özellik             | BPE / WordPiece | Bu Tokenizer |
| ------------------- | --------------- | ------------ |
| Türkçe uyumu        | Düşük           | Yüksek       |
| Morfem farkındalığı | Yok             | Var          |
| Vocabulary boyutu   | Büyük           | Daha küçük   |
| Dilbilgisel bilgi   | Öğrenilmeli     | Entegre      |

---

## 🚀 Hedeflenen Kullanım Alanları

* Türkçe LLM’ler
* Akademik NLP araştırmaları
* Dilbilgisel farkındalık gerektiren görevler

---

## 📌 Not

Bu tokenizer, Türkçe için **inductive bias** eklemeyi amaçlayan deneysel bir çalışmadır ve klasik tokenizer’ların birebir alternatifi değil, **dil-özel bir çözüm** olarak tasarlanmıştır.
