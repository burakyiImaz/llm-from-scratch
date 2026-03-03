
---

# LLM Trainer – Matematiksel ve Algoritmik Açıklama

Bu doküman PyTorch tabanlı **LLM Trainer** sınıfının matematiksel temelini açıklar.
Odak noktası yalnızca eğitim mekanizmasıdır.

---

# 1. Problem Tanımı: Otoregresif Dil Modelleme

Uzunluğu (T) olan bir token dizisi:

$$
x = (x_1, x_2, \dots, x_T)
$$

Zincir kuralına göre ortak olasılık:

$$
P(x)
====

\prod_{t=1}^{T}
P(x_t \mid x_1, \dots, x_{t-1})
$$

Model her adımda bir sonraki token'ın koşullu olasılığını tahmin eder.

Amaç:

$$
\max_\theta P_\theta(x)
$$

---

# 2. Maksimum Olabilirlik (MLE)

Logaritma alırsak:

$$
\log P_\theta(x)
================

\sum_{t=1}^{T}
\log P_\theta(x_t \mid x_{<t})
$$

Negatif log-likelihood minimize edilir:

$$
\mathcal{L}
===========

*

\sum_{t=1}^{T}
\log P_\theta(x_t \mid x_{<t})
$$

Batch boyutu (B) dahil edilirse:

$$
\mathcal{L}
===========

*

\frac{1}{B T}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P_\theta(x_{b,t})
$$

Bu ifade doğrudan **cross entropy loss**'tur.

---

# 3. Cross Entropy ve Softmax

Model her adımda logits üretir:

$$
z_{t,i}
$$

Softmax dönüşümü:

$$
P_\theta(x_t = i)
=================

\frac{\exp(z_{t,i})}
{\sum_{j=1}^{V} \exp(z_{t,j})}
$$

Tek token kaybı:

$$
\ell_t
======

*

\log P_\theta(x_t = y_t)
$$

Toplam kayıp:

$$
\mathcal{L}
===========

*

\frac{1}{B T}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log
\frac{\exp(z_{b,t,y_{b,t}})}
{\sum_{j=1}^{V} \exp(z_{b,t,j})}
$$

Bu, logits ile hedef indeks üzerinden hesaplanır.

---

# 4. Perplexity

Tanım:

$$
\text{Perplexity}
=================

\exp(\mathcal{L})
$$

Eğer model her token'a eşit olasılık verse:

$$
P = \frac{1}{K}
$$

O zaman:

$$
\mathcal{L} = \log K
\quad \Rightarrow \quad
\text{Perplexity} = K
$$

Yani perplexity efektif seçim sayısını temsil eder.

---

# 5. Learning Rate Schedule

## Warmup

$$
\text{lr}(t)
============

\text{lr}_{\text{base}}
\cdot
\frac{t}{W}
$$

Amaç:

* Erken aşamada stabilite
* Büyük gradient patlamasını önleme

---

## Cosine Annealing

$$
p
=

\frac{t-W}{T-W}
$$

$$
\text{lr}(t)
============

\frac{1}{2}
\text{lr}_{\text{base}}
\left(
1 + \cos(\pi p)
\right)
$$

Bu schedule pürüzsüz azalma sağlar.

---

# 6. AdamW

Gradient:

$$
g_t
===

\nabla_\theta \mathcal{L}
$$

Birinci moment:

$$
m_t
===

\beta_1 m_{t-1}
+
(1-\beta_1) g_t
$$

İkinci moment:

$$
v_t
===

\beta_2 v_{t-1}
+
(1-\beta_2) g_t^2
$$

Bias düzeltme:

$$
\hat{m}_t
=========

\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t
=========

\frac{v_t}{1-\beta_2^t}
$$

Güncelleme:

$$
\theta_{t+1}
============

## \theta_t

\eta
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
---------------------------------------------

\eta \lambda \theta_t
$$

Son terim weight decay'dir.

---

# 7. Gradient Clipping

Norm:

$$
|g|_2
=====

\sqrt{\sum_i g_i^2}
$$

Eğer:

$$
|g|_2 > c
$$

ise:

$$
g
\leftarrow
g
\cdot
\frac{c}{|g|_2}
$$

Amaç: exploding gradient önleme.

---

# 8. Gradient Accumulation

Efektif batch:

$$
B_{\text{eff}}
==============

B \cdot k
$$

Loss ölçekleme:

$$
\mathcal{L}_{\text{scaled}}
===========================

\frac{\mathcal{L}}{k}
$$

Her (k) adımda bir optimizer step yapılır.

---

# 9. Automatic Mixed Precision

Loss scaling:

$$
\mathcal{L}_{\text{scaled}}
===========================

s \mathcal{L}
$$

Backward sonrası:

$$
g
\leftarrow
\frac{g}{s}
$$

Amaç: underflow önlemek.

---

# 10. Early Stopping

Eğer:

$$
\mathcal{L}*{\text{val},t}
\ge
\mathcal{L}*{\text{best}}
$$

durumu (p) epoch sürerse eğitim durdurulur.

---

# 13. Label Smoothing

One-hot hedef:

$$
y_i =
\begin{cases}
1 & i = y_t \
0 & \text{otherwise}
\end{cases}
$$

Label smoothing:

$$
y_i =
\begin{cases}
1-\epsilon & i = y_t \
\frac{\epsilon}{V-1} & \text{otherwise}
\end{cases}
$$

Yeni loss:

$$
\mathcal{L}
===========

*

\sum_{i=1}^{V}
y_i
\log P_\theta(i)
$$

Etkileri:

* Logit büyümesi sınırlanır
* Entropy artar
* Calibration iyileşir

---

# 15. Sharpness-Aware Minimization (SAM)

Tanım:

$$
\mathcal{L}_{\text{SAM}}(\theta)
================================

\max_{|\epsilon|_2 \le \rho}
\mathcal{L}(\theta + \epsilon)
$$

Yaklaşık çözüm:

$$
\epsilon
========

\rho
\frac{g}{|g|_2}
$$

Sonra:

$$
g_{\text{SAM}}
==============

\nabla_\theta
\mathcal{L}(\theta+\epsilon)
$$

Güncelleme:

$$
\theta
\leftarrow
\theta
------

\eta g_{\text{SAM}}
$$

Amaç: flat minimum bulmak.

---

# 16. Second-Order Yaklaşımlar

Hessian:

$$
H
=

\nabla_\theta^2 \mathcal{L}
$$

Newton adımı:

$$
\theta_{t+1}
============

## \theta_t

H^{-1} g_t
$$

Condition number:

$$
\kappa(H)
=========

\frac{\lambda_{\max}(H)}
{\lambda_{\min}(H)}
$$

Küçük (\kappa) → hızlı yakınsama.

---

# 17. Adaptive Gradient Clipping

Standart:

$$
|g|_2 > c
$$

AGC:

$$
|g|_2

>

\lambda
|\theta|_2
$$

Parametre ölçeğine duyarlıdır.

---

# 21. EMA

Güncelleme:

$$
\theta_t^{\text{EMA}}
=====================

\alpha \theta_{t-1}^{\text{EMA}}
+
(1-\alpha)\theta_t
$$

Açılım:

$$
\theta_t^{\text{EMA}}
=====================

\sum_{k=0}^{t}
(1-\alpha)
\alpha^k
\theta_{t-k}
$$

Etkisi:

$$
\mathrm{Var}(\theta^{\text{EMA}})
<
\mathrm{Var}(\theta)
$$

---

# 23. Compute-Optimal Training

Toplam compute yaklaşık:

$$
C
\approx
6 N D
$$

Scaling law:

$$
\mathcal{L}(N)
==============

a N^{-\alpha}
+
b
$$

Optimal veri oranı:

$$
D \propto N
$$

---

# 24. Multi-Objective Training

Toplam loss:

$$
\mathcal{L}_{\text{total}}
==========================

\mathcal{L}*{\text{LM}}
+
\lambda_1 \mathcal{L}*{\text{reg}}
+
\lambda_2 \mathcal{L}_{\text{aux}}
$$

---

# 25. Knowledge Distillation

Softmax with temperature:

$$
P(i)
====

\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
$$

Distillation loss:

$$
\mathcal{L}_{\text{KD}}
=======================

T^2
,
\mathrm{KL}
\left(
P_T^{(T)}
;|;
P_S^{(T)}
\right)
$$

Toplam:

$$
\mathcal{L}
===========

\alpha \mathcal{L}*{\text{CE}}
+
(1-\alpha)\mathcal{L}*{\text{KD}}
$$

---

# 26. Checkpoint Averaging

$$
\theta_{\text{avg}}
===================

\frac{1}{k}
\sum_{i=1}^{k}
\theta_i
$$

EMA’ye benzer şekilde varyansı azaltır.

---

# 27. Monitoring

Gradient norm:

$$
|g|_2
$$

Update norm:

$$
|\Delta \theta|_2
$$

Attention entropy:

$$
H_{\text{attn}}
===============

*

\sum
p
\log p
$$

---



İstersen bir sonraki adımda bunu tamamen **akademik makale formatına (ICLR style)** dönüştürebiliriz.
