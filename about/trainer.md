# LLM Trainer – Matematiksel ve Algoritmik Açıklama

Bu doküman PyTorch tabanlı **LLM Trainer** sınıfının matematiksel temelini açıklar.  

README yalnızca eğitim mekanizmasına (Trainer) odaklanmaktadır.

---

# 1. Problem Tanımı: Otoregresif Dil Modelleme

Bir dil modeli uzunluğu T olan bir token dizisini gözlemler:

$$
x = (x_1, x_2, \dots, x_T)
$$

Zincir kuralına göre dizinin olasılığı:

$$
P(x) = \prod_{t=1}^{T} P(x_t \, | \, x_1, x_2, \dots, x_{t-1})
$$

Trainer’ın amacı bu olasılığı maksimize etmektir.

---

# 2. Maksimum Olabilirlik (MLE)

Logaritma alırsak:

$$
\log P(x) = \sum_{t=1}^{T} \log P(x_t \, | \, x_1, x_2, \dots, x_{t-1})
$$

Negatif log-likelihood minimize edilir:

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t \, | \, x_1, x_2, \dots, x_{t-1})
$$

Batch boyutu B dahil edildiğinde:

$$
\mathcal{L} = - \frac{1}{B T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log P_\theta(x_{b,t})
$$

Bu doğrudan **Cross Entropy Loss** ile hesaplanır.

---

# 3. Cross Entropy ve Softmax İlişkisi

Model her adımda logits üretir: $z_{t,i}$  

Softmax dönüşümü:

$$
P_\theta(x_t = i) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}
$$

Tek token için kayıp:

$$
\ell_t = - \log P_\theta(x_t = y_t)
$$

Batch ve zaman boyunca ortalama:

$$
\mathcal{L} = - \frac{1}{B T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \frac{\exp(z_{b,t,y_{b,t}})}{\sum_{j=1}^{V} \exp(z_{b,t,j})}
$$

---

# 4. Perplexity

$$
\text{Perplexity} = \exp(\mathcal{L})
$$

Eğer model her token için eşit olasılık dağıtsa $P = 1/K$:

$$
\mathcal{L} = \log K \quad \Rightarrow \quad \text{Perplexity} = K
$$

---

# 5. Learning Rate Schedule

## 5.1 Warmup

İlk W adımda learning rate lineer artar:

$$
\text{lr}(t) = \text{lr}_\text{base} \cdot \frac{t}{W}
$$

## 5.2 Cosine Annealing

$$
p = \frac{t - W}{T - W}, \quad
\text{lr}(t) = \frac{1}{2} \text{lr}_\text{base} \left( 1 + \cos(\pi p) \right)
$$

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

Bias düzeltmeleri:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parametre güncellemesi:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Weight decay:

$$
\theta_{t+1} = \theta_{t+1} - \eta \lambda \theta_t
$$

---

# 7. Gradient Clipping

$$
\lVert g \rVert_2 = \sqrt{\sum_i g_i^2}, \quad
\text{if } \lVert g \rVert_2 > c, \quad
g \leftarrow g \cdot \frac{c}{\lVert g \rVert_2}
$$

---

# 8. Gradient Accumulation

$$
B_\text{eff} = B \times k, \quad
\mathcal{L}_\text{scaled} = \frac{\mathcal{L}}{k}
$$

Her k adımda bir optimizer step yapılır.

---

# 9. Automatic Mixed Precision (AMP)

$$
\mathcal{L}_\text{scaled} = \mathcal{L} \cdot s, \quad
g \leftarrow \frac{g}{s}
$$

---
# 10. Early Stopping

Eğer $p$ epoch boyunca validation loss iyileşmezse:

$$
\mathcal{L}_{\mathrm{val}, t} \ge \mathcal{L}_{\mathrm{best\_val}} \;\;\Rightarrow\;\; \text{stop training}
$$

Açıklama:

- $\mathcal{L}_{\mathrm{val}, t}$ → t. epoch’daki validation loss  
- $\mathcal{L}_{\mathrm{best\_val}}$ → şimdiye kadar görülen en iyi validation loss  
---

# 11. Scaling Law

$$
\mathcal{L}(N) = a N^{-\alpha} + b, \quad \alpha \approx 0.05-0.1
$$

Yaklaşık veri oranı:

$$
\text{token sayısı} \approx 10 \text{ ile } 20 \times \text{parametre sayısı}
$$

---

# 12. Gelecek Çalışmalar

- Custom CUDA kernel optimizasyonu  
- Memory-efficient attention  
- Fused optimizer adımları  
- Kernel seviyesinde hız iyileştirmeleri  
- Compute-optimal scaling stratejileri  

---

# 13. Advanced Regularization Teknikleri

---

## 13.1 Label Smoothing

Standart cross-entropy one-hot hedef kullanır:

$$
y_i =
\begin{cases}
1 & \text{if } i = y_t \
0 & \text{otherwise}
\end{cases}
$$

Bu durumda modelin optimum çözümü:

$$
P_\theta(x_t = y_t) \to 1
$$

Bu da **aşırı güven (overconfidence)** üretir.

Label smoothing ile hedef dağılım yumuşatılır:

$$
y_i =
\begin{cases}
1 - \epsilon & \text{if } i = y_t \
\frac{\epsilon}{V - 1} & \text{otherwise}
\end{cases}
$$

Yeni loss:

$$
\mathcal{L}
===========

* \sum_{i=1}^{V} y_i \log P_\theta(i)
  $$

### Neden işe yarar?

* Logit büyümesini sınırlar
* Gradient magnitüdünü stabilize eder
* Modelin entropy’sini artırır
* Calibration iyileştirir

Pratikte:

* $\epsilon \in [0.05, 0.2]$

---

## 13.2 Dropout Scheduling

Sabit dropout yerine zamanla azalan dropout:

$$
p(t) = p_{max} \left(1 - \frac{t}{T}\right)
$$

Erken aşama:

* Model feature keşfeder
* Daha güçlü regularization gerekir

Geç aşama:

* İnce ayar yapılır
* Deterministik davranış avantajlıdır

Bu yaklaşım özellikle büyük Transformer’larda stabilite sağlar.

---

## 13.3 Stochastic Depth

Transformer katmanları olasılıksal olarak atlanır:

$$
h_{l+1} =
\begin{cases}
\text{Layer}(h_l) & \text{with prob } 1-p_l \
h_l & \text{with prob } p_l
\end{cases}
$$

Genelde:

$$
p_l = \frac{l}{L} p_{max}
$$

Avantaj:

* Derin ağların eğitilebilirliği artar
* Implicit ensemble etkisi oluşur
* Gradient flow kolaylaşır

---

# 14. Gradient Noise Injection

Gradient’e Gaussian gürültü eklenir:

$$
g_t \leftarrow g_t + \mathcal{N}(0, \sigma_t^2)
$$

Zamana bağlı varyans:

$$
\sigma_t^2 = \frac{\eta}{(1+t)^\gamma}
$$

Burada:

* $\eta$ → learning rate ölçeği
* $\gamma \approx 0.5$

### Teorik Etki

Bu yöntem SGD'nin:

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

güncellemesini stokastik diferansiyel denkleme yaklaştırır.

Sonuç:

* Sharp minimum yerine flat minimum
* Daha iyi genelleme

---

# 15. Sharpness-Aware Minimization (SAM)

Amaç:

Sadece düşük loss değil, aynı zamanda **düz (flat) minimum** bulmak.

Tanım:

$$
\mathcal{L}_{SAM}(\theta)
=========================

\max_{|\epsilon| \le \rho}
\mathcal{L}(\theta + \epsilon)
$$

Yaklaşık çözüm:

1. İlk gradient hesapla:

$$
g = \nabla_\theta \mathcal{L}
$$

2. Pertürbasyon oluştur:

$$
\epsilon = \rho \frac{g}{|g|}
$$

3. Yeni noktada gradient hesapla:

$$
g_{SAM} = \nabla_\theta \mathcal{L}(\theta + \epsilon)
$$

4. Parametre güncelle:

$$
\theta \leftarrow \theta - \eta g_{SAM}
$$

### Etki

* Loss yüzeyinin curvature’ı azalır
* Validation perplexity düşer
* Overfitting azalır

---

# 16. Second-Order Yaklaşımlar

Hessian:

$$
H = \nabla^2_\theta \mathcal{L}
$$

Newton yöntemi:

$$
\theta_{t+1}
============

## \theta_t

H^{-1} g_t
$$

Tam Hessian pahalıdır:

* Hesaplama: $O(N^2)$
* Bellek: $O(N^2)$

Yaklaşık yöntemler:

* K-FAC → Kronecker factorization
* Shampoo → Matrix preconditioning
* Diagonal Hessian approx

Amaç:

Condition number düşürmek:

$$
\kappa(H) = \frac{\lambda_{max}}{\lambda_{min}}
$$

Daha küçük $\kappa$ → daha hızlı convergence.

---

# 17. Adaptive Gradient Clipping (AGC)

Standart clipping:

$$
|g|_2 > c
$$

AGC:

$$
|g|_2 > \lambda |\theta|_2
$$

Bu ölçek bağımsızdır.

Büyük modellerde:

* Layer bazlı clipping yapılır
* Daha stabil Transformer eğitimi sağlar

---

# 18. Curriculum Learning

Zorluk metriği:

$$
\text{difficulty}(x)
====================

* \log P_\theta(x)
  $$

Veri sırası:

$$
D_1 \rightarrow D_2 \rightarrow D_3
$$

Avantaj:

* Optimization landscape daha pürüzsüz olur
* Erken aşamada hızlı düşüş

---

# 19. Dynamic Sequence Length Scheduling

Başlangıçta kısa context:

$$
L(t) =
L_{min}
+
\left(\frac{t}{T}\right)
(L_{max} - L_{min})
$$

Neden?

Self-attention karmaşıklığı:

$$
O(L^2)
$$

Başlangıçta kısa L → daha hızlı iteration.

---

# 20. Token-Level Adaptive Sampling

Uniform sampling yerine:

$$
P(x) \propto \text{loss}(x)^\alpha
$$

$\alpha > 0$ ise:

* Zor örneklere ağırlık verilir

Hard example mining:

* Rare token öğrenimi hızlanır
* Vocabulary coverage artar

---

# 21. EMA (Exponential Moving Average)

$$
\theta^{EMA}_t
==============

\alpha \theta^{EMA}_{t-1}
+
(1-\alpha)\theta_t
$$

Genelde:

$$
\alpha \in [0.999, 0.9999]
$$

Validation ve inference EMA ile yapılır.

### Etki

* Parametre varyansı azalır
* Daha düşük perplexity

---

# 22. Activation Checkpointing

Bellek:

$$
O(L \cdot d \cdot N)
$$

Checkpointing ile:

* Ara aktivasyonlar saklanmaz
* Backward sırasında yeniden hesaplanır

Trade-off:

$$
\text{Memory} \downarrow
\quad
\text{Compute} \uparrow
$$

---

# 23. Compute-Optimal Training

Toplam compute:

$$
C \approx 6 N D
$$

Burada:

* $N$ → parametre
* $D$ → token

Scaling law:

$$
\mathcal{L}(N)
==============

a N^{-\alpha} + b
$$

Optimal oran:

$$
D \propto N
$$

Trainer bunu takip edip:

* Token/parametre oranı
* Compute budget

üzerinden otomatik durdurma yapabilir.

---

# 24. Multi-Objective Training

Toplam loss:

$$
\mathcal{L}_{total}
===================

\mathcal{L}*{LM}
+
\lambda_1 \mathcal{L}*{reg}
+
\lambda_2 \mathcal{L}_{aux}
$$

Örnek auxiliary loss:

* Contrastive
* Sentence embedding
* Distillation

Bu, temsil kalitesini artırabilir.

---

# 25. Knowledge Distillation

Teacher dağılımı:

$$
P_T
$$

Student:

$$
P_S
$$

Temperature scaling:

$$
P(i)
====

\frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Distillation loss:

$$
\mathcal{L}_{KD}
================

T^2
\cdot
\text{KL}(P_T^T | P_S^T)
$$

Toplam:

$$
\mathcal{L}
===========

\alpha \mathcal{L}*{CE}
+
(1-\alpha) \mathcal{L}*{KD}
$$

Amaç:

* Küçük modelde büyük model davranışı
* Daha düşük perplexity / parametre

---

# 26. Checkpoint Averaging

Son k checkpoint:

$$
\theta_{avg}
============

\frac{1}{k}
\sum_{i=1}^{k}
\theta_i
$$

EMA’ye alternatif.

Genelde:

* Son epoch’larda uygulanır
* Validation stabilizasyonu sağlar

---

# 27. Research-Level Monitoring

Ek ölçümler:

Gradient norm:

$$
|g|_2
$$

Update norm:

$$
|\Delta \theta|_2
$$

Curvature approx:

$$
\text{trace}(H)
$$

Attention entropy:

$$
H_{attn}
========

*

\sum p \log p
$$

Bu metrikler scaling ve optimization analizi için kritik.

---

