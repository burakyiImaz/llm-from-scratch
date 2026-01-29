
# 📄 TextDataset – Dil Modeli için Eğitim Verisi Hazırlama

Bu proje, **tokenize edilmiş bir metni**, bir dil modelini (GPT tarzı *next-token prediction*) eğitmek için uygun hale getiren özel bir `PyTorch Dataset` sınıfı içerir.

Amaç:

> Uzun bir token dizisini, **sabit uzunlukta bağlam pencerelerine (context window)** bölmek ve her giriş için **bir sonraki token’ları hedef (target)** olarak üretmek.

---

## 🎯 Genel Mantık

Dil modelleri şu problemi öğrenir:

> **“Bu token dizisini gördüğümde bir sonraki token ne olmalı?”**

Bunu yapmak için:

* Girdi (`input`):

  ```
  [t0, t1, t2, ..., t(n-1)]
  ```
* Hedef (`target`):

  ```
  [t1, t2, t3, ..., t(n)]
  ```

Yani hedef dizisi, girdinin **1 token kaydırılmış hali**dir.

---

## 🧱 Kodun Yapısı

### 1️⃣ `TextDataset` Sınıfı

```python
class TextDataset(Dataset):
```

Bu sınıf:

* `torch.utils.data.Dataset`’ten türetilmiştir
* PyTorch `DataLoader` ile uyumlu çalışır
* Token dizisini parçalara ayırır ve modelin anlayacağı tensörlere çevirir

---

### 2️⃣ Girdi Parametreleri

```python
def __init__(self, token_ids, context_length, stride):
```

| Parametre        | Açıklama                                         |
| ---------------- | ------------------------------------------------ |
| `token_ids`      | Tokenizer’dan çıkmış **uzun bir token listesi**  |
| `context_length` | Modelin aynı anda göreceği maksimum token sayısı |
| `stride`         | Sliding window’un ne kadar kayarak ilerleyeceği  |

📌 **stride küçükse:** daha fazla örnek, daha çok veri
📌 **stride büyükse:** daha az örnek, daha hızlı eğitim

---

### 3️⃣ Sliding Window (Kaydırmalı Pencere) Mantığı

```python
for i in range(0, len(token_ids)-context_length, stride):
```

Bu döngü:

* Token dizisi üzerinde ilerler
* Her adımda `context_length` uzunluğunda bir pencere alır
* Pencereyi `stride` kadar kaydırır

Örnek:

```text
Token dizisi: [1,2,3,4,5,6,7,8]
context_length = 4
stride = 2
```

Oluşan girişler:

```
[1,2,3,4]
[3,4,5,6]
[5,6,7,8]
```

---

### 4️⃣ Input ve Target Oluşturma

```python
input_chunk  = token_ids[i : i + context_length]
target_chunk = token_ids[i+1 : i + context_length + 1]
```

Burada:

* `input_chunk` → modelin gördüğü veri
* `target_chunk` → modelin tahmin etmeye çalıştığı veri

📌 **1 token kaydırma**, dil modelinin temel öğrenme prensibidir.

---

### 5️⃣ Padding (Eksik Token Tamamlama)

```python
input_chunk  = input_chunk  + [pad_id] * (context_length - len(input_chunk))
target_chunk = target_chunk + [pad_id] * (context_length - len(target_chunk))
```

Amaç:

* Tüm örneklerin **aynı uzunlukta** olması
* Batch işlemlerinin sorunsuz çalışması

📌 `pad_id`:

* Tokenizer’da özel olarak tanımlanmış **PAD token ID’si**
* Genellikle loss hesaplanırken maskelenir

---

### 6️⃣ Tensor’a Dönüştürme

```python
torch.tensor(input_chunk, dtype=torch.long)
```

Neden?

* PyTorch modelleri **tensor** ile çalışır
* `long` → embedding katmanları için zorunludur

---

### 7️⃣ Dataset Fonksiyonları

```python
def __len__(self):
```

* Dataset’te kaç örnek olduğunu döndürür

```python
def __getitem__(self, idx):
```

* Belirli bir index için `(input, target)` döndürür

Bu iki fonksiyon sayesinde `DataLoader` çalışabilir.

---

### 8️⃣ DataLoader Oluşturma

```python
def create_data_loader(token_ids, context_length, stride, batch_size, shuffle=True):
```

Bu fonksiyon:

* `TextDataset` oluşturur
* PyTorch `DataLoader` döndürür

Avantajları:

* Mini-batch eğitim
* Shuffle desteği
* GPU uyumluluğu

---

## 🔁 Eğitimde Nasıl Kullanılır?

```python
loader = TextDataset.create_data_loader(
    token_ids=token_ids,
    context_length=128,
    stride=64,
    batch_size=32
)

for inputs, targets in loader:
    logits = model(inputs)
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
```

