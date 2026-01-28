# Rotary Positional Embedding (RoPE) Tabanlı Embedding Modülü

Bu proje, **PyTorch** kullanılarak sıfırdan yazılmış, **Rotary Positional Encoding (RoPE)** mantığını temel alan bir embedding katmanını içerir. Amaç; klasik pozisyon embedding’lerine (absolute positional embedding) alternatif olarak, **token vektörlerinin kendi içinde döndürülmesi (rotation)** yoluyla pozisyon bilgisini modele kazandırmaktır.

Bu yapı özellikle **Transformer tabanlı dil modellerinde** (LLM) tercih edilen, modern ve matematiksel olarak güçlü bir yaklaşımdır.

---

## 📌 Genel Bakış

Kod iki ana parçadan oluşur:

1. **`get_rotary_position_encoding` fonksiyonu**
   Gömülü (embedded) token vektörlerine RoPE uygular.

2. **`Embedding` sınıfı (`nn.Module`)**
   Token embedding + rotary positional encoding işlemini tek bir modülde birleştirir.

---

## 🔢 Rotary Positional Encoding (RoPE) Nedir?

RoPE, pozisyon bilgisini vektöre **eklemek yerine**, vektörü pozisyona bağlı olarak **sinüs–kosinüs dönüşümüyle döndürür**.

Bu yaklaşımın avantajları:

* Relative (göreli) pozisyon bilgisini doğal olarak kodlar
* Uzun bağlamlarda (long context) daha kararlıdır
* Attention mekanizmasıyla matematiksel olarak uyumludur

Formül olarak her boyut çifti için:

* `x_even' = x_even · cos(θ) − x_odd · sin(θ)`
* `x_odd'  = x_even · sin(θ) + x_odd · cos(θ)`

Buradaki `θ`, token pozisyonuna ve boyuta bağlıdır.

---

## ⚙️ Fonksiyon: `get_rotary_position_encoding`

### Girdi

* **`input`** → `[batch_size, context_length, embedding_dim]`
* **`base`** → Frekans ölçeği (varsayılan: `10000`)
* **`device`** → CPU / GPU seçimi

### Adım Adım Ne Yapılıyor?

1. **Boyut kontrolü**
   Embedding boyutunun çift olması zorunludur.

2. **Boyutu ikiye bölme**
   Embedding vektörü `even` ve `odd` parçalarına ayrılır.

3. **Frekansların hesaplanması**
   Her boyut için farklı açısal frekans üretilir.

4. **Pozisyon açılarının oluşturulması**
   Her token pozisyonu için sinüs ve kosinüs değerleri hesaplanır.

5. **Rotasyon işlemi**
   Vektör çiftleri sin–cos dönüşümüyle döndürülür.

6. **Birleştirme**
   Döndürülmüş parçalar tekrar tek embedding haline getirilir.

### Çıktı

* Pozisyon bilgisi **gömülü**, aynı boyutta tensor

---

## 🧠 Sınıf: `Embedding`

Bu sınıf, PyTorch’un `nn.Module` yapısını kullanarak **öğrenilebilir token embedding** ile **RoPE**’u birleştirir.

### `__init__`

* `vocab_size` → Kelime dağarcığı boyutu
* `embedding_dim` → Embedding vektör boyutu
* `context_length` → Maksimum bağlam uzunluğu
* `device` → Çalışma cihazı

İçeride:

* `nn.Embedding` tanımlanır
* Rotary positional encoding fonksiyonu bağlanır

### `forward(x)`

1. Token ID’leri embedding vektörlerine çevrilir
2. Rotary positional encoding uygulanır
3. Son embedding döndürülür

---

## 📐 Tensor Akışı

```
Input Token IDs
      ↓
nn.Embedding
      ↓
[token embedding]
      ↓
Rotary Positional Encoding
      ↓
[final embedding with position info]
```

---

## ✅ Neden Bu Yaklaşım?

* Absolute positional embedding yok → daha esnek
* Relative pozisyon ilişkileri korunur
* GPT‑NeoX, LLaMA gibi modern modellerle uyumlu
* Kendi LLM’ini yazmak isteyenler için temiz ve öğretici bir örnek

---

## 🚀 Kullanım Senaryoları

* Kendi Transformer / LLM modelini yazanlar
* Positional encoding mantığını derinlemesine öğrenmek isteyenler
* GPT‑2 tabanlı ama daha modern bir positional yapı denemek isteyenler

---

## 📌 Notlar

* Embedding boyutu **mutlaka çift** olmalıdır
* RoPE genellikle **Q ve K matrislerine** uygulanır; burada öğretici olması için embedding seviyesinde uygulanmıştır

---

## ✨ Sonuç

Bu modül, sıfırdan bir dil modeli geliştirme sürecinde **modern positional encoding yaklaşımlarını** anlamak ve uygulamak için sağlam bir temel sunar.

