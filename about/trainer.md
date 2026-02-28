
---

# LLM Trainer – Matematiksel ve Algoritmik Açıklama

Bu doküman, PyTorch tabanlı LLM Trainer sınıfının matematiksel temellerini açıklar.
Bu README yalnızca eğitim mekanizmasına (Trainer) odaklanmaktadır.

Model mimarisi, kernel optimizasyonu ve ileri seviye hesaplama iyileştirmeleri ilerleyen aşamalarda ayrı dokümante edilecektir.

---

# 1. Problem Tanımı: Otoregresif Dil Modelleme

Bir dil modeli aşağıdaki diziyi gözlemler:

$$
x = (x_1, x_2, \ldots, x_T)
$$

Model zincir kuralını kullanarak tüm dizinin olasılığını şöyle ayrıştırır:

$$
P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

Burada:

$$
x_{<t} = (x_1, \ldots, x_{t-1})
$$

Trainer’ın optimize ettiği amaç, bu olasılığı maksimize etmektir.

---

# 2. Maksimum Olabilirlik (MLE)

Log alırsak:

$$
\log P(x) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

Negatif log-likelihood minimize edilir:

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

Batch boyutu dahil edilirse:

$$
\mathcal{L} =

* \frac{1}{B T}
  \sum_{b=1}^{B}
  \sum_{t=1}^{T}
  \log P_\theta(x_{b,t})
  $$

Bu doğrudan Cross Entropy loss’tur.

---

# 3. Cross Entropy’nin Softmax ile İlişkisi

Model logits üretir:

$$
z_{t,i}
$$

Softmax:

$$
P_\theta(x_t = i) =
\frac{e^{z_{t,i}}}
{\sum_{j=1}^{V} e^{z_{t,j}}}
$$

Burada:

* $V$ = vocabulary size

Cross entropy tek token için:

$$
\ell_t = - \log P_\theta(x_t)
$$

Batch ve zaman boyunca ortalama:

$$
\mathcal{L} =

* \frac{1}{B T}
  \sum_{b,t}
  \log
  \frac{
  e^{z_{b,t,y_{b,t}}}
  }{
  \sum_{j=1}^{V} e^{z_{b,t,j}}
  }
  $$

---

# 4. Perplexity

Perplexity şu şekilde tanımlanır:

$$
\text{PPL} = \exp(\mathcal{L})
$$

Bu şu anlama gelir:

Eğer model her token için eşit olasılık dağıtsa:

$$
P = \frac{1}{K}
$$

o zaman:

$$
\text{PPL} = K
$$

Yani perplexity modelin ortalama belirsizlik derecesidir.

---

# 5. Learning Rate Schedule

Trainer iki aşamalı schedule kullanır.

## 5.1 Warmup

İlk $W$ adımda learning rate lineer artar:

$$
\text{lr}(t) =
\text{lr}_{base}
\cdot
\frac{t}{W}
$$

Bu sayede başlangıçta ani büyük parametre sıçramaları engellenir.

---

## 5.2 Cosine Annealing

Warmup sonrası:

$$
p =
\frac{t - W}
{T - W}
$$

Learning rate:

$$
\text{lr}(t) =
\frac{1}{2}
\text{lr}_{base}
\left(
1 + \cos(\pi p)
\right)
$$

Bu fonksiyon:

* Başta yüksek
* Ortada yumuşak düşüş
* Sonda küçük adımlar

üretir.

---

# 6. AdamW Optimizasyonu

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

Bias düzeltmesi:

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
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t} + \epsilon}
$$

AdamW’de weight decay ayrı uygulanır:

$$
\theta_{t+1}
============

## \theta_{t+1}

\eta
\lambda
\theta_t
$$

Bu klasik L2 regularization’dan matematiksel olarak farklıdır çünkü decay gradient’e değil doğrudan parametreye uygulanır.

---

# 7. Gradient Clipping

Gradient normu:

$$
| g |_2 =
\sqrt{
\sum_i g_i^2
}
$$

Eğer:

$$
| g |_2 > c
$$

ise yeniden ölçeklenir:

$$
g \leftarrow
g
\cdot
\frac{c}{| g |_2}
$$

Bu, özellikle Transformer mimarilerinde gradient explosion’ı engeller.

---

# 8. Gradient Accumulation

Memory kısıtında büyük batch simülasyonu yapılır.

Toplam effective batch:

$$
B_{eff} = B \times k
$$

Loss şu şekilde ölçeklenir:

$$
\mathcal{L}_{scaled} =
\frac{\mathcal{L}}{k}
$$

Backprop sonrası k adımda bir optimizer step yapılır.

---

# 9. Automatic Mixed Precision (AMP)

FP16 kullanımı için loss ölçeklenir:

$$
\mathcal{L}_{scaled} =
\mathcal{L}
\cdot
s
$$

Backward sonrası:

$$
g \leftarrow
\frac{g}{s}
$$

Eğer overflow oluşursa ölçek faktörü otomatik düşürülür.

Bu:

* Daha az bellek
* Daha hızlı eğitim

sağlar.

---

# 10. Early Stopping

Eğer $p$ epoch boyunca:

$$
\text{val_loss}_{t}
\ge
\text{best_val_loss}
$$

durumu devam ederse eğitim sonlandırılır.

Amaç:

* Overfitting önlemek
* Hesaplama maliyetini azaltmak

---

# 11. Scaling Law

Literatürde yaklaşık ilişki:

$$
\mathcal{L}(N)
==============

a N^{-\alpha}
+
b
$$

Burada:

* $N$ parametre sayısı
* $\alpha$ yaklaşık 0.05 ile 0.1 arası

Bu ilişki şunu gösterir:

Parametre artışı loss’u düşürür fakat getirisi giderek azalır.

Veri miktarı da kritik öneme sahiptir.

Yaklaşık optimal oran:

$$
\text{token sayısı}
\approx
10
\text{ ile }
20
\times
\text{parametre sayısı}
$$

---

# 12. Gelecek Çalışmalar

Bu README yalnızca Trainer mekanizmasını kapsamaktadır.

İlerleyen aşamalarda:

* Custom CUDA kernel optimizasyonu
* Memory efficient attention
* Fused optimizer step
* Kernel-level hız iyileştirmeleri
* Compute-optimal scaling stratejileri

ayrı ve detaylı şekilde dokümante edilecektir.

---
