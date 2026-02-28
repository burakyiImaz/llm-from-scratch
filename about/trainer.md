
---

# LLM Trainer – Matematiksel Açıklamalı Eğitim Altyapısı

Bu proje, PyTorch tabanlı bir **Large Language Model (LLM)** eğitim altyapısıdır.
Trainer sınıfı aşağıdaki gelişmiş özellikleri içerir:

* AdamW optimizer
* Warmup + Cosine Annealing learning rate schedule
* Gradient clipping
* Gradient accumulation
* Automatic Mixed Precision (AMP)
* Early stopping
* Perplexity hesaplama
* Checkpoint kaydetme / yükleme

Bu dokümanda sistemin **matematiksel temelleri detaylı şekilde açıklanmaktadır.**

---

# 1️ Modelin Optimize Ettiği Amaç Fonksiyonu

LLM'ler temel olarak **next-token prediction** problemi çözer.

Verilen bir token dizisi:

[
x = (x_1, x_2, ..., x_T)
]

Modelin amacı:

[
P(x_t \mid x_{<t})
]

olasılığını modellemektir.

---

## Cross Entropy Loss

Trainer'da kullanılan loss:

```python
self.loss_fn = nn.CrossEntropyLoss()
```

Matematiksel olarak:

[
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
]

Batch boyutunu da dahil edersek:

[
\mathcal{L} = - \frac{1}{B T}
\sum_{b=1}^{B} \sum_{t=1}^{T}
\log P_\theta(x_{b,t})
]

Bu loss:

* Log-likelihood maximization
* Maximum likelihood estimation (MLE)

yapmaktadır.

---

# 2️ Perplexity

Evaluation sırasında hesaplanan metrik:

```python
perplexity = math.exp(avg_loss)
```

Matematiksel olarak:

[
\text{Perplexity} = e^{\mathcal{L}}
]

Perplexity şunu ifade eder:

> Model ortalama olarak bir token için kaç olası seçenek arasında kararsız kalıyor?

Örneğin:

* PPL = 10 → model ortalama 10 seçenek arasında kararsız
* PPL = 2 → çok iyi model

---

# 3️ Learning Rate Schedule

Trainer'da iki aşamalı schedule vardır:

---

##  Warmup Phase

İlk ( W ) adımda learning rate lineer artar:

[
\text{lr}_t =
\text{base_lr} \cdot
\frac{t}{W}
]

Amaç:

* Başlangıçta büyük gradient patlamasını önlemek
* Stabil başlangıç sağlamak

---

## 🌊 Cosine Annealing Phase

Warmup sonrası:

[
\text{progress} =
\frac{t - W}{T - W}
]

[
\text{lr}_t =
\frac{1}{2}
\text{base_lr}
\left(
1 + \cos(\pi \cdot \text{progress})
\right)
]

Bu:

* Learning rate'i yavaşça azaltır
* Eğitimin sonuna doğru ince ayar yapar

---

# 4️ AdamW Optimizasyonu

Kullanılan optimizer:

```python
optim.AdamW(...)
```

Adam güncellemesi:

[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
]

[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
]

Bias correction:

[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
]

[
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
]

Parametre güncellemesi:

[
\theta_{t+1} =
\theta_t -
\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
]

AdamW'de weight decay ayrı uygulanır:

[
\theta_{t+1} =
\theta_{t+1} -
\eta \lambda \theta_t
]

Bu klasik L2 regularization’dan farklıdır.

---

# 5️ Gradient Clipping

Kod:

```python
torch.nn.utils.clip_grad_norm_(...)
```

Eğer:

[
|g|_2 > c
]

ise:

[
g \leftarrow
g \cdot \frac{c}{|g|_2}
]

Amaç:

* Gradient explosion önlemek
* Stabil training

---

# 6️ Gradient Accumulation

Batch çok büyük olduğunda memory yetmeyebilir.

Trainer:

[
\text{effective batch size} =
\text{batch size} \times
\text{gradient accumulation steps}
]

Loss şu şekilde ölçeklenir:

[
\mathcal{L}_{scaled} =
\frac{\mathcal{L}}{k}
]

k = accumulation steps

Bu sayede:

* Daha büyük batch etkisi
* Daha stabil gradient

---

# 7️ Automatic Mixed Precision (AMP)

FP32 yerine:

* FP16 + FP32 karışık kullanılır

Bu:

* Bellek tüketimini azaltır
* Eğitimi hızlandırır

Gradient scaler:

[
\mathcal{L}_{scaled} =
\mathcal{L} \cdot s
]

Backprop sonrası:

[
g \leftarrow \frac{g}{s}
]

Overflow varsa scaler otomatik düşürülür.

---

# 8️ Early Stopping

Eğer:

[
\text{val_loss}_{t}
\ge
\text{best_val_loss}
]

durumu ( p ) epoch boyunca devam ederse:

[
\text{training stop}
]

Amaç:

* Overfitting önlemek
* Gereksiz hesaplama engellemek

---

# 9️ Scaling Law (LLM'lerde Temel İlişki)

LLM literatüründe yaklaşık şu ilişki gözlenir:

[
\mathcal{L}(N) =
a N^{-\alpha} + b
]

Burada:

* ( N ) = parametre sayısı
* ( \alpha \approx 0.05 - 0.1 )

Bu şunu gösterir:

* Parametre artışı → loss düşer
* Ama **diminishing returns** vardır

Daha önemli olan:

> Model büyüdükçe veri de büyümelidir.

Genelde önerilen oran:

[
\text{token sayısı} \approx 10 - 20 \times \text{parametre sayısı}
]

---

# 10 Genel Eğitim Akışı

1. Forward pass
2. Cross entropy hesapla
3. Gradient scaling
4. Backpropagation
5. Gradient clipping
6. Optimizer step
7. Learning rate update
8. Validation
9. Early stopping kontrolü
10. Checkpoint kaydet

---

#  Sonuç

Bu Trainer:

* Küçük modellerden (5M–20M parametre)
* Orta ölçekli modellere (100M+)

kadar ölçeklenebilir.

Matematiksel olarak:

* MLE optimizasyonu
* Cosine annealing schedule
* AdamW adaptif optimizasyon
* Perplexity temelli değerlendirme
* Scaling law bilinci

içermektedir.

---

