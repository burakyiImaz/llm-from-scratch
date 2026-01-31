import torch
import torch.nn as nn

# kedi köpeği kovaladı , köpek kediyi kovaladı
#yukarıda her ne kadar kelimeler aynı olsa da anlamsal bir farklılık vardır. Bu farklılığı sağlamak için pozisyonel kodlama kullanılır. Deepseek in kullandığı RoPE yaklaşımı
def get_rotary_position_encoding(input: torch.Tensor, base=10000, device="cpu"):
    """
    Rotary Positional Encoding (RoPE) uygular.

    Bu fonksiyon, embedding vektörlerine pozisyon bilgisini
    sinüs–kosinüs tabanlı bir rotasyon ile ekler.
    Böylece attention mekanizması göreli konumları öğrenebilir.

    Girdi:
        input: [batch_size, context_length, dimension]
        dimension mutlaka çift olmalıdır.

    Çıkış:
        Aynı shape'te, pozisyon bilgisi eklenmiş embedding tensorü
    """

    context_length, dimension = input.shape

    """
    RoPE çalışabilmesi için embedding boyutunun çift olması gerekir.
    Çünkü her iki boyut bir 2D vektör gibi ele alınır.
    """
    assert dimension % 2 == 0, "boyut çift olmalı"

    half_dimension = dimension // 2

    """
    Her embedding boyutu için farklı frekanslar üretilir.
    Düşük boyutlar → yavaş değişen sinüsler
    Yüksek boyutlar → hızlı değişen sinüsler
    """
    freqs_indices = torch.arange(
        0, half_dimension,
        device=device,
        dtype=torch.float32
    )

    freqs = 1.0 / (base ** (freqs_indices / dimension))

    """
    Token pozisyonlarını temsil eden indeksler.
    Her token kendi pozisyonuna göre döndürülecek.
    """
    positions = torch.arange(
        0, context_length,
        device=device,
        dtype=torch.float32
    )

    """
    Açılar, pozisyon ile frekansın çarpımıdır.
    Bu açılar sin ve cos fonksiyonlarına girecek.
    """
    angles = positions[:, None] * freqs[None, :]

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    """
    Embedding vektörü iki parçaya ayrılır.
    İlk yarı x bileşeni,
    ikinci yarı y bileşeni gibi düşünülür.
    """
    input_even = input[:, :, :half_dimension]
    input_odd  = input[:, :, half_dimension:]

    """
    Klasik 2 boyutlu rotasyon uygulanır.
    Bu işlem embedding'i pozisyona bağlı olarak döndürür.
    """
    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated  = input_even * sin_angles + input_odd * cos_angles

    """
    Döndürülmüş parçalar tekrar birleştirilir
    ve orijinal embedding boyutuna dönülür.
    """
    input_rotated = torch.cat(
        [input_even_rotated, input_odd_rotated],
        dim=-1
    )

    return input_rotated


#günümüzde aktif olarak kullanılan bir yöntemdir.
def get_position_encoding(context_length, embedding_dim,base=10000 ,device= "cpu"):
    pos_embedding= torch.zeros(context_length, embedding_dim,device=device)
    for pos in range(context_length):
        for i in range(0, embedding_dim//2):
            pos_embedding[pos, 2*i]= torch.sin(pos / (base ** (2*i/ embedding_dim)))
            if i+1 < embedding_dim//2:
                pos_embedding[pos, 2*i +1]= torch.cos(pos / (base ** (2*i/ embedding_dim)))
    return pos_embedding.unsqueeze(0)  # [1, context_length, embedding_dim] unsqueeze ile batch dimension eklenir ve tensore dönüştürülür.

class Model(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_length,device="cpu"):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size, embedding_dim, device=device) # şu an için torch kullanılıyor ilerleyen süreçte bireysel olarak modelin özellikle büyük matris işlemlerinde performansı da göz önünde bulundurularak daha sağlam bir optimizasyon geliştiricem.
        self.pos_embedding= nn.Embedding(context_length, embedding_dim, device=device)
        self.get_pos= get_rotary_position_encoding
        self.device= device

    def forward(self,x):
        x= self.embedding(x) #tokenlerin sözlükteki anlamlarını vektörlere dönüştürür.
        x= self.get_pos(x,device=self.device) #tokenlerin pozisyonel bilgilerini ekler.
        x= self.pos_embedding(x)
        return x

