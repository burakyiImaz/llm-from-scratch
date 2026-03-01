

---

# 1️ Problem Tanımı: Otoregresif Dil Modelleme

Verilen:

[
x = (x_1, x_2, \dots, x_T)
]

Bu bir token dizisi.

Buradaki temel varsayım şudur:

> Dil bir **olasılık süreci** olarak modellenebilir.

Bir cümleyi üretmek demek, her adımda bir sonraki token’ı seçmek demektir.

Zincir kuralı:

[
P(x) = \prod_{t=1}^{T} P(x_t , | , x_1, x_2, \dots, x_{t-1})
]

Bu ifade şunu söyler:

* Tüm dizinin olasılığı
* Her adımda “geçmiş verildiğinde şimdiki token’ın olasılığı”
* Çarpımıdır.

Bu tam olarak **faktörizasyon** problemidir.

### Neden bu form?

Çünkü doğrudan:

[
P(x_1, x_2, ..., x_T)
]

gibi devasa bir joint dağılımı modellemek imkansızdır.

Ama zincir kuralı sayesinde:

* Yüksek boyutlu ortak dağılım
* Koşullu dağılımların çarpımına ayrılır.

Bu da transformer gibi modellerin çalışmasını mümkün kılar.

---

# 2️⃣ Maksimum Olabilirlik (MLE)

Amaç:

[
\max_\theta P_\theta(x)
]

Yani parametreleri, gerçek veri en olası olacak şekilde ayarlamak.

Log alıyoruz:

[
\log P(x) = \sum_{t=1}^{T} \log P(x_t , | , x_1, x_2, \dots, x_{t-1})
]

### Neden log alıyoruz?

1. Çarpım → toplama dönüşür (numerical stability)
2. Gradient hesaplamak kolaylaşır
3. Underflow problemi azalır

Negatif log-likelihood:

[
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(x_t , | , x_1, x_2, \dots, x_{t-1})
]

Bu aslında şudur:

> Model yanlış tahmin yaptığında büyük ceza alır.

Çünkü:

* Eğer doğru token’a düşük olasılık verirsen
* log değeri büyük negatif olur
* negatifini alınca büyük pozitif kayıp oluşur

Batch versiyonu:

[
\mathcal{L} = - \frac{1}{B T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log P_\theta(x_{b,t})
]

Bu artık:

* Zaman boyunca
* Batch boyunca
* Ortalama kayıptır

Yani stochastic gradient descent için optimize edilebilir hale gelir.

---

# 3️⃣ Cross Entropy ve Softmax

Model logits üretir:

[
z_{t,i}
]

Bu değerler:

* Normalize edilmemiş skorlar
* Olasılık değildir
* Negatif veya pozitif olabilir

Softmax:

[
P_\theta(x_t = i) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}
]

### Softmax neden gerekli?

Çünkü:

* Olasılıklar ≥ 0 olmalı
* Toplamları 1 olmalı

exp fonksiyonu:

* Negatif değerleri pozitif yapar
* Büyük logit’i üstel büyütür

Bu nedenle softmax:

> Rekabetçi normalizasyon

gibi davranır.

Tek token kaybı:

[
\ell_t = - \log P_\theta(x_t = y_t)
]

Bu ifade aslında:

> Gerçek dağılım ile model dağılımı arasındaki çapraz entropidir.

Cross entropy:

[
H(p, q) = - \sum p(x) \log q(x)
]

Dil modelinde:

* p(x) → one-hot dağılım
* q(x) → model tahmini

Bu nedenle loss sadeleşir.

---

# 4️⃣ Perplexity

[
\text{Perplexity} = \exp(\mathcal{L})
]

Bu ölçü:

> Model ortalama kaç seçenek arasında kararsız kalıyor?

Eğer model tamamen uniform tahmin yaparsa:

[
\mathcal{L} = \log K
]

[
\text{Perplexity} = K
]

Yani:

* 50k vocab varsa
* model rastgele tahmin yapıyorsa
* perplexity ≈ 50k

İyi modelde perplexity düşer çünkü:

* Model belirsizliği azaltır.

Perplexity aslında:

[
2^{H}
]

gibi entropinin üstel versiyonudur.

---

# 5️⃣ Learning Rate Schedule

## Warmup

[
\text{lr}(t) = \text{lr}_\text{base} \cdot \frac{t}{W}
]

Neden?

Transformer başlangıçta:

* Rastgele init
* Büyük gradient
* LayerNorm hassasiyeti

Direkt büyük LR → divergence.

Warmup:

> Sistemi stabil hale getirir.

---

## Cosine Annealing

[
\text{lr}(t) = \frac{1}{2} \text{lr}_\text{base} \left( 1 + \cos(\pi p) \right)
]

Bu şunu yapar:

* Başta yüksek LR
* Sonda yumuşak düşüş

Cosine seçilmesinin nedeni:

* Smooth derivative
* Sert düşüş yok
* Ani optimizasyon sıçraması olmaz

Bu global minimuma yakın bölgede:

> Küçük adımlar atılmasını sağlar.

---

# 6️⃣ AdamW Optimizasyonu

Gradient:

[
g_t = \nabla_\theta \mathcal{L}
]

Adam’ın mantığı:

SGD:
[
\theta_{t+1} = \theta_t - \eta g_t
]

Adam:

* Momentum (m_t)
* RMS scaling (v_t)

Birinci moment:

[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
]

Bu:

> Exponential moving average of gradients

İkinci moment:

[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
]

Bu:

> Gradient variance tahmini

Bias correction:

[
\hat{m}_t, \hat{v}_t
]

Başlangıçta momentlar sıfır olduğu için correction gerekir.

Güncelleme:

[
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
]

Bu:

* Büyük variance → küçük adım
* Küçük variance → büyük adım

Yani her parametre için adaptif learning rate.

---

## AdamW’de Weight Decay

Normal L2 regularization gradient’e eklenir.

AdamW’de:

[
\theta_{t+1} = \theta_{t+1} - \eta \lambda \theta_t
]

Bu decoupled decay’dir.

Neden önemli?

Adam’da L2 regularization:

* Moment hesaplarını bozar

AdamW:

> Regularization’ı gradient’den ayırır.

Bu daha doğru bir weight shrinkage sağlar.

---

# 7️⃣ Gradient Clipping

[
\lVert g \rVert_2
]

Transformer’larda özellikle:

* Uzun sequence
* Büyük model

Gradient explosion olabilir.

Eğer norm > c ise:

[
g \leftarrow g \cdot \frac{c}{\lVert g \rVert_2}
]

Bu:

* Yönü değiştirmez
* Sadece büyüklüğü sınırlar

Yani:

> Stabilizasyon mekanizmasıdır.

---

# 8️⃣ Gradient Accumulation

[
B_\text{eff} = B \times k
]

GPU memory yetmezse:

* Küçük batch ile k adım gradient biriktirilir.

Loss scaling:

[
\mathcal{L}_\text{scaled} = \frac{\mathcal{L}}{k}
]

Bu yapılmazsa:

* Gradient k kat büyür.

Bu yöntem:

> Büyük batch simülasyonu yapar.

Büyük batch:

* Gradient variance azaltır
* Daha stabil optimizasyon sağlar

---

# 9️⃣ Automatic Mixed Precision (AMP)

Float16 kullanılır.

Problem:

* Küçük gradientler underflow olur.

Çözüm:

[
\mathcal{L}_\text{scaled} = \mathcal{L} \cdot s
]

Backward:

* Gradient s ile büyütülür
* Step öncesi tekrar bölünür

Bu:

> Sayısal hassasiyet kaybını önler.

Aynı zamanda:

* Bellek %50 azalır
* TensorCore hızlanır

---

# 🔟 Early Stopping

[
\mathcal{L}*{\mathrm{val}, t} \ge \mathcal{L}*{\mathrm{best_val}}
]

Validation loss iyileşmezse:

> Model overfit etmeye başlamıştır.

Bu mekanizma:

* Genelleme performansını korur
* Gereksiz compute harcamaz

---

# 11️⃣ Scaling Law

[
\mathcal{L}(N) = a N^{-\alpha} + b
]

Bu ifade:

* Model büyüdükçe loss azalır
* Ama azalma hızı düşer

(\alpha) küçük:

→ Logaritmik benzeri yavaş iyileşme

Token sayısı:

[
\text{token sayısı} \approx 10-20 \times \text{parametre sayısı}
]

Bu compute-optimal regime’den gelir.

Çok az veri:

* Overfit

Çok fazla veri ama küçük model:

* Under-parameterized

Optimal nokta:

> Parametre ve veri dengesi

---

# 12️⃣ Gelecek Çalışmaların Matematiksel Sebebi

* Memory efficient attention → O(n²) yerine daha düşük complexity
* Fused kernels → memory bandwidth bottleneck azaltma
* Compute-optimal scaling → FLOPs minimizasyonu
* Custom CUDA → kernel launch overhead azaltma

Transformer training’in çoğu:

> Memory bandwidth bound problemidir.

---
