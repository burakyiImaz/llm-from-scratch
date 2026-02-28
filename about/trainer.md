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

Eğer p epoch boyunca validation loss iyileşmezse:

$$
\mathrm{val\_loss}_t \ge \mathrm{best\_val\_loss} \;\Rightarrow\; \text{training stop}
$$
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

