import torch
import torch.nn as nn


# kedi köpeği kovaladı , köpek kediyi kovaladı
#yukarıda her ne kadar kelimeler aynı olsa da anlamsal bir farklılık vardır. Bu farklılığı sağlamak için pozisyonel kodlama kullanılır. Deepseek in kullandığı RoPE yaklaşımı

def get_rotary_position_encoding(input, base=10000, device="cpu"):
    batch_size, context_length, dimension = input.shape
    assert dimension % 2 == 0

    half_dim = dimension // 2

    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device) / dimension))
    positions = torch.arange(0, context_length, device=device)

    angles = positions[:, None] * freqs[None, :]
    sin = torch.sin(angles).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)

    x_even = input[:, :, :half_dim]
    x_odd  = input[:, :, half_dim:]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    return torch.cat([x_rot_even, x_rot_odd], dim=-1)

#günümüzde aktif olarak kullanılan bir yöntemdir.
def get_position_encoding(context_length, embedding_dim,base=10000 ,device= "cpu"):
    pos_embedding= torch.zeros(context_length, embedding_dim,device=device)
    for pos in range(context_length):
        for i in range(0, embedding_dim//2):
            pos_embedding[pos, 2*i]= torch.sin(pos / (base ** (2*i/ embedding_dim)))
            if i+1 < embedding_dim//2:
                pos_embedding[pos, 2*i +1]= torch.cos(pos / (base ** (2*i/ embedding_dim)))
    return pos_embedding.unsqueeze(0)  # [1, context_length, embedding_dim] unsqueeze ile batch dimension eklenir ve tensore dönüştürülür.
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, device="cpu"):
        super().__init__()

        #  vocab_size'ı dışarıdan erişilebilir yaptım
        # Böylece model.forward içinde
        # self.embedding.num_embeddings HATASI almayacağız
        self.num_embeddings = vocab_size  

        #  token embedding katmanı
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        ).to(device)

        #  context length saklanıyor (RoPE için mantıklı)
        self.context_length = context_length

        #  rotary position encoding fonksiyonunu referans olarak tutuyoruz
        # (her forward'da yeniden import vs olmasın diye)
        self.get_pos = get_rotary_position_encoding

        #  device bilgisini sınıf içinde saklıyoruz
        self.device = device

    def forward(self, x):
        """
        x: (batch, seq_len) veya (seq_len,)
        """

        #  Güvenlik: token id vocab dışına çıkmış mı?
        if torch.any(x >= self.num_embeddings):
            raise ValueError("Token id vocab_size dışına çıktı")

        #  Token embedding
        x = self.embedding(x)

        #  Rotary Position Encoding
        # RoPE embedding boyutunu bozmaz, sadece rotate eder
        x = self.get_pos(x, device=self.device)

        return x
