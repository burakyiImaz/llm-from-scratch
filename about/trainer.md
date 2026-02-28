
---

# LLM Trainer – Matematiksel ve Algoritmik Açıklama

Bu doküman PyTorch tabanlı LLM Trainer sınıfının matematiksel temelini açıklar.

Bu README yalnızca eğitim mekanizmasına (Trainer) odaklanmaktadır.

Model mimarisi, kernel optimizasyonu ve düşük seviye performans iyileştirmeleri ilerleyen aşamalarda ayrı olarak dokümante edilecektir.

---

# 1. Problem Tanımı: Otoregresif Dil Modelleme

Bir dil modeli uzunluğu ( T ) olan bir token dizisini gözlemler:

$$
x = (x_{1}, x_{2}, \dots, x_{T})
$$

Zincir kuralına göre tüm dizinin olasılığı şu şekilde ayrıştırılır:

$$
P(x) =
\prod_{t=1}^{T}
P(x_{t} \mid x_{1}, \dots, x_{t-1})
$$

Burada:

* ( x_{t} ) → t anındaki token
* ( x_{1}, \dots, x_{t-1} ) → önceki tüm tokenlar

Trainer’ın amacı bu olasılığı maksimize etmektir.

---

# 2. Maksimum Olabilirlik (Maximum Likelihood Estimation)

Logaritma alırsak:

$$
\log P(x)
=========

\sum_{t=1}^{T}
\log
P(x_{t} \mid x_{1}, \dots, x_{t-1})
$$

Negatif log-likelihood minimize edilir:

$$
\mathcal{L}
===========

*

\sum_{t=1}^{T}
\log
P_{\theta}(x_{t} \mid x_{1}, \dots, x_{t-1})
$$

Batch boyutu ( B ) dahil edildiğinde:

$$
\mathcal{L}
===========

*

\frac{1}{B T}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log
P_{\theta}(x_{b,t})
$$

Bu doğrudan Cross Entropy loss’tur.

---

# 3. Cross Entropy ve Softmax İlişkisi

Model her zaman adımında logits üretir:

$$
z_{t,i}
$$

Softmax dönüşümü:

$$
P_{\theta}(x_{t} = i)
=====================

\frac{
\exp(z_{t,i})
}{
\sum_{j=1}^{V}
\exp(z_{t,j})
}
$$

Burada:

* ( V ) vocabulary size

Tek token için kayıp:

$$
\ell_{t}
========

*

\log
P_{\theta}(x_{t})
$$

Batch ve zaman boyunca ortalama:

$$
\mathcal{L}
===========

*

\frac{1}{B T}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log
\frac{
\exp(z_{b,t,y_{b,t}})
}{
\sum_{j=1}^{V}
\exp(z_{b,t,j})
}
$$

Bu formül doğrudan `nn.CrossEntropyLoss()` ile hesaplanır.

---

# 4. Perplexity

Perplexity:

$$
\text{Perplexity}
=================

\exp(\mathcal{L})
$$

Eğer model her token için eşit olasılık dağıtsa:

$$
P = \frac{1}{K}
$$

Bu durumda:

$$
\mathcal{L} = \log K
$$

Dolayısıyla:

$$
\text{Perplexity} = K
$$

Yani perplexity modelin ortalama belirsizlik derecesidir.

---

# 5. Learning Rate Schedule

Trainer iki aşamalı schedule kullanır.

## 5.1 Warmup

İlk ( W ) adımda learning rate lineer artar:

$$
\text{lr}(t)
============

\text{lr}_{base}
\cdot
\frac{t}{W}
$$

Amaç:

* Başlangıç instabilitesini azaltmak
* Büyük gradient sıçramalarını önlemek

---

## 5.2 Cosine Annealing

Warmup sonrası:

$$
p
=

\frac{t - W}{T - W}
$$

Learning rate:

$$
\text{lr}(t)
============

\frac{1}{2}
\text{lr}_{base}
\left(
1 + \cos(\pi p)
\right)
$$

Bu yöntem:

* Başta büyük adımlar
* Ortada yumuşak azalma
* Sonda ince ayar

sağlar.

---

# 6. AdamW Optimizasyonu

Gradient:

$$
g_{t}
=====

\nabla_{\theta}
\mathcal{L}
$$

Birinci moment:

$$
m_{t}
=====

\beta_{1} m_{t-1}
+
(1 - \beta_{1}) g_{t}
$$

İkinci moment:

$$
v_{t}
=====

\beta_{2} v_{t-1}
+
(1 - \beta_{2}) g_{t}^{2}
$$

Bias düzeltmeleri:

$$
\hat{m}_{t}
===========

\frac{m_{t}}{1 - \beta_{1}^{t}}
$$

$$
\hat{v}_{t}
===========

\frac{v_{t}}{1 - \beta_{2}^{t}}
$$

Parametre güncellemesi:

$$
\theta_{t+1}
============

## \theta_{t}

\eta
\frac{
\hat{m}*{t}
}{
\sqrt{\hat{v}*{t}} + \epsilon
}
$$

AdamW’de weight decay ayrı uygulanır:

$$
\theta_{t+1}
============

## \theta_{t+1}

\eta
\lambda
\theta_{t}
$$

Bu klasik L2 regularization’dan farklıdır çünkü decay gradient’e değil doğrudan parametreye uygulanır.

---

# 7. Gradient Clipping

Gradient normu:

$$
| g |_{2}
=========

\sqrt{
\sum_{i}
g_{i}^{2}
}
$$

Eğer:

$$
| g |_{2} > c
$$

ise yeniden ölçeklenir:

$$
g
\leftarrow
g
\cdot
\frac{c}{| g |_{2}}
$$

Bu yöntem gradient explosion riskini azaltır.

---

# 8. Gradient Accumulation

Bellek kısıtı varsa büyük batch simülasyonu yapılır.

Effective batch:

$$
B_{eff}
=======

B
\times
k
$$

Loss ölçekleme:

$$
\mathcal{L}_{scaled}
====================

\frac{\mathcal{L}}{k}
$$

Her ( k ) adımda bir optimizer step uygulanır.

---

# 9. Automatic Mixed Precision (AMP)

Loss ölçeklenir:

$$
\mathcal{L}_{scaled}
====================

\mathcal{L}
\cdot
s
$$

Backward sonrası:

$$
g
\leftarrow
\frac{g}{s}
$$

Overflow oluşursa ölçek faktörü otomatik düşürülür.

Bu yöntem:

* Bellek kullanımını azaltır
* Eğitimi hızlandırır

---

# 10. Early Stopping

Eğer ( p ) epoch boyunca validation loss iyileşmezse:

$$
\text{val_loss}_{t}
\ge
\text{best_val_loss}
$$

eğitim durdurulur.

Amaç:

* Overfitting önlemek
* Gereksiz hesaplamayı azaltmak

---

# 11. Scaling Law

LLM literatüründe yaklaşık ilişki:

$$
\mathcal{L}(N)
==============

a
N^{-\alpha}
+
b
$$

Burada:

* ( N ) parametre sayısı
* ( \alpha \approx 0.05 - 0.1 )

Bu ilişki şunu gösterir:

Parametre sayısı arttıkça loss azalır fakat getirisi giderek düşer.

Yaklaşık veri oranı:

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
* Memory-efficient attention
* Fused optimizer adımları
* Kernel seviyesinde hız iyileştirmeleri
* Compute-optimal scaling stratejileri

ayrı ve detaylı şekilde ele alınacaktır.

---

