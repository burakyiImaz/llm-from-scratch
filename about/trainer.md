
---

# LLM Trainer – Matematiksel ve Teknik Açıklama

Bu proje PyTorch tabanlı bir Large Language Model (LLM) eğitim altyapısıdır. Trainer sınıfı aşağıdaki mekanizmaları içerir:

* AdamW optimizer
* Warmup + Cosine Annealing learning rate schedule
* Gradient clipping
* Gradient accumulation
* Automatic Mixed Precision (AMP)
* Early stopping
* Perplexity hesaplama
* Checkpoint kaydetme ve yükleme

Bu doküman, sistemin matematiksel temelini detaylı olarak açıklar.

---

# 1. Amaç Fonksiyonu: Next Token Prediction

Bir dil modeli verilen bir token dizisi için koşullu olasılığı öğrenir.

Verilen dizi:

$$
x = (x_1, x_2, ..., x_T)
$$

Modelin amacı:

$$
P(x_t \mid x_{<t})
$$

olasılığını modellemektir.

Toplam log-likelihood:

$$
\log P(x) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

Model negatif log-likelihood’i minimize eder.

---

## 1.1 Cross Entropy Loss

Trainer’da kullanılan kayıp fonksiyonu:

```
nn.CrossEntropyLoss()
```

Matematiksel olarak:

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

Batch boyutu dahil edildiğinde:

$$
\mathcal{L} =

* \frac{1}{B T}
  \sum_{b=1}^{B}
  \sum_{t=1}^{T}
  \log P_\theta(x_{b,t})
  $$

Bu Maximum Likelihood Estimation (MLE) optimizasyonudur.

---

# 2. Perplexity

Perplexity şu şekilde hesaplanır:

$$
\text{Perplexity} = e^{\mathcal{L}}
$$

Yorum:

* Perplexity = 1 → model mükemmel
* Düşük perplexity → daha iyi model
* Yüksek perplexity → daha fazla belirsizlik

---

# 3. Learning Rate Schedule

## 3.1 Warmup Fazı

İlk $W$ adımda learning rate lineer artar:

$$
\text{lr}_t =
\text{base_lr} \cdot \frac{t}{W}
$$

Amaç:

* Başlangıç instabilitesini azaltmak
* Büyük gradient patlamasını önlemek

---

## 3.2 Cosine Annealing

Warmup sonrası:

$$
\text{progress} =
\frac{t - W}{T - W}
$$

Learning rate:

$$
\text{lr}_t =
\frac{1}{2}
\text{base_lr}
\left(
1 + \cos(\pi \cdot \text{progress})
\right)
$$

Bu schedule:

* Eğitimin başında büyük adımlar
* Sonda küçük ve hassas güncellemeler

sağlar.

---

# 4. AdamW Optimizasyonu

Adam algoritması moment tahmini yapar.

Gradient:

$$
g_t = \nabla_\theta \mathcal{L}
$$

Birinci moment:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

İkinci moment:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parametre güncellemesi:

$$
\theta_{t+1} =
\theta_t -
\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

AdamW’de weight decay ayrı uygulanır:

$$
\theta_{t+1} =
\theta_{t+1} -
\eta \lambda \theta_t
$$

Bu klasik L2 regularization’dan farklıdır.

---

# 5. Gradient Clipping

Eğer gradient normu eşikten büyükse:

$$
|g|_2 > c
$$

yeniden ölçeklenir:

$$
g \leftarrow
g \cdot \frac{c}{|g|_2}
$$

Amaç:

* Gradient explosion önlemek
* Eğitim stabilitesini artırmak

---

# 6. Gradient Accumulation

Gerçek batch size:

$$
\text{effective batch size} =
\text{batch size} \times
k
$$

Burada $k$ accumulation step sayısıdır.

Loss şu şekilde ölçeklenir:

$$
\mathcal{L}_{scaled} =
\frac{\mathcal{L}}{k}
$$

Bu, büyük batch etkisini bellek sınırı olmadan sağlar.

---

# 7. Automatic Mixed Precision (AMP)

FP16 ve FP32 birlikte kullanılır.

Loss ölçeklenir:

$$
\mathcal{L}_{scaled} =
\mathcal{L} \cdot s
$$

Backprop sonrası:

$$
g \leftarrow \frac{g}{s}
$$

Overflow varsa ölçek otomatik düşürülür.

---

# 8. Early Stopping

Eğer validation loss $p$ epoch boyunca iyileşmezse:

$$
\text{val_loss}_t \ge \text{best_val_loss}
$$

eğitim durdurulur.

Bu overfitting’i önler.

---

# 9. LLM Scaling Law

Literatürde gözlenen yaklaşık ilişki:

$$
\mathcal{L}(N) =
a N^{-\alpha} + b
$$

Burada:

* $N$ parametre sayısı
* $\alpha \approx 0.05 - 0.1$

Bu şunu gösterir:

* Parametre artışı → loss azalır
* Ancak diminishing returns vardır

Ayrıca model büyüdükçe veri de büyümelidir.

Yaklaşık optimal oran:

$$
\text{token sayısı} \approx 10 - 20 \times \text{parametre sayısı}
$$

---

# 10. Eğitim Akışı

1. Forward pass
2. Cross entropy hesaplama
3. Gradient scaling
4. Backpropagation
5. Gradient clipping
6. Optimizer step
7. Learning rate update
8. Validation
9. Early stopping kontrolü
10. Checkpoint kaydetme

---

