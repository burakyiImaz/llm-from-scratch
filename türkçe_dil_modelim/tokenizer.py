import json
import torch
import numpy as np


class Tokenizer:
    """
    Türkçe LLM için morfem-tabanlı tokenizer
    JSON yapısı: kategorilere bölünmüş token sözlüğü
    Dinamik öğrenme: Yeni kelimeleri otomatik olarak ekleyebilir
    """
    
    def __init__(self, vocab_file, encoding='utf-8', auto_learn=True):
        """
        Tokenizer'ı başlat
        
        Args:
            vocab_file (str): Tokenizer JSON dosyasının yolu
            encoding (str): Dosya encoding tipi (varsayılan: utf-8)
            auto_learn (bool): Yeni kelimeleri otomatik olarak öğren (varsayılan: True)
        """
        self.vocab_file = vocab_file
        self.encoding = encoding
        self.auto_learn = auto_learn
        
        with open(vocab_file, 'r', encoding=encoding) as f:
            vocab_data = json.load(f)
        
        # Tüm kategorileri birleştirerek tek sözlük oluştur
        self.vocab = {}
        self.vocab_categories = {}  # Kategorileri de sakla
        
        for category, tokens in vocab_data.items():
            if isinstance(tokens, dict):
                self.vocab_categories[category] = tokens
                self.vocab.update(tokens)
        
        # Ters sözlük (ID -> Token) - özel tokenleri önceliklendir
        self.reverse_vocab = {}
        
        # Önce özel tokenları ekle
        special_tokens = ["<pad>", "<unk>", "<başla>", "<bitiş>", "<büyük_harf>"]
        for token, token_id in self.vocab.items():
            if token in special_tokens:
                self.reverse_vocab[token_id] = token
        
        # Sonra diğerlerini ekle (çift ID durumunda, açıkça tanımlanmış olanı al)
        for token, token_id in self.vocab.items():
            if token not in special_tokens:
                self.reverse_vocab[token_id] = token
        
        # Sonraki token ID'si (dinamik ekleme için)
        self.next_token_id = max(self.vocab.values()) + 1 if self.vocab else 0
        
        # Özel tokenler
        self.pad_id = self.vocab.get("<pad>", 722)
        self.unk_id = self.vocab.get("<unk>", 721)
        self.start_id = self.vocab.get("<başla>", 723)
        self.end_id = self.vocab.get("<bitiş>", 724)
        self.uppercase_id = self.vocab.get("<büyük_harf>", 727)
        self.space_id = self.vocab.get(" ", 717)
    
    def encode(self, text, add_uppercase_token=True, add_special_tokens=False):
        """
        Metni token ID'lerine çevir (Subword tokenization)
        
        Args:
            text (str): Kodlanacak metin
            add_uppercase_token (bool): Büyük harfle başlayan cümlelere token ekle
            add_special_tokens (bool): <başla> ve <bitiş> tokenlerini ekle
            
        Returns:
            torch.Tensor: Token ID'leri
        """
        tokens = []
        
        if not text:
            return torch.tensor(tokens, dtype=torch.long)
        
        text = text.strip()
        
        # <başla> tokeni ekle
        if add_special_tokens:
            tokens.append(self.start_id)
        
        # Büyük harfle başlayan cümleler için token ekle
        if add_uppercase_token and text and text[0].isupper():
            tokens.append(self.uppercase_id)
        
        # Kelime kelime işle
        for word in text.split():
            if not word:
                continue
            
            # Greedy subword tokenization
            self._tokenize_word(word, tokens)
            
            # Kelimeler arası boşluk
            tokens.append(self.space_id)
        
        # Son boşluğu sil (metin boşlukla bitmediyse)
        if tokens and tokens[-1] == self.space_id:
            tokens.pop()
        
        # <bitiş> tokeni ekle
        if add_special_tokens:
            tokens.append(self.end_id)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _tokenize_word(self, word, tokens):
        """
        Kelimeyi subword token'lere ayır (Greedy matching)
        Eşleşme bulunamayan karakterleri dinamik olarak ekler
        
        Args:
            word (str): Tokenize edilecek kelime
            tokens (list): Token ID'lerinin ekleneceği liste
        """
        i = 0
        while i < len(word):
            found_match = False
            
            # En uzundan en kısaya doğru kontrol et
            for j in range(len(word), i, -1):
                sub_word = word[i:j]
                
                if sub_word in self.vocab:
                    tokens.append(self.vocab[sub_word])
                    i = j
                    found_match = True
                    break
            
            # Eşleşme bulunamadıysa
            if not found_match:
                # Tek karakteri al
                char = word[i]
                
                # Eğer karakter zaten vocab'da varsa, kullan
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    # Auto-learn açıksa, yeni karakteri ekle
                    if self.auto_learn:
                        self.add_token(char, "karakterler")
                        tokens.append(self.vocab[char])
                    else:
                        # Aksi halde unknown token ekle
                        tokens.append(self.unk_id)
                
                i += 1
    
    def encode_batch(self, texts, context_length, 
                    add_uppercase_token=True, add_special_tokens=False):
        """
        Bir grup metni kodla ve padding uygula
        
        Args:
            texts (list): Kodlanacak metinlerin listesi
            context_length (int): İstenilen token uzunluğu
            add_uppercase_token (bool): Büyük harf tokeni ekle
            add_special_tokens (bool): Başla/bitiş tokenlerini ekle
            
        Returns:
            torch.Tensor: Shape (batch_size, context_length)
        """
        sentence_tokens = []
        
        for text in texts:
            tokens = self.encode(text, add_uppercase_token, add_special_tokens).tolist()
            
            # Padding veya truncation uygula
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
            else:
                # Boşluk padding yap
                pad_length = context_length - len(tokens)
                tokens = tokens + [self.pad_id] * pad_length
            
            sentence_tokens.append(tokens)
        
        return torch.tensor(sentence_tokens, dtype=torch.long)
    
    def tokenize(self, text):
        """
        Metni token adlarına çevir (debug için)
        
        Args:
            text (str): Metni
            
        Returns:
            list: Token adlarının listesi
        """
        token_ids = self.encode(text).tolist()
        return [self.reverse_vocab.get(id, f"<id:{id}>") for id in token_ids]
    
    def decode(self, token_ids, remove_special_tokens=True):
        """
        Token ID'lerini metne çevir
        
        Args:
            token_ids (list or torch.Tensor): Token ID'leri
            remove_special_tokens (bool): Özel tokenları kaldır
            
        Returns:
            str: Çevrilmiş metin
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Özel token ID'leri
        special_ids = {self.pad_id, self.start_id, self.end_id, 
                      self.unk_id, self.uppercase_id}
        
        text = ""
        for token_id in token_ids:
            # Özel tokenları atla
            if remove_special_tokens and token_id in special_ids:
                continue
            
            token_str = self.reverse_vocab.get(token_id, "")
            text += token_str
        
        return text
    
    def get_vocab_size(self):
        """Toplam token sayısını döndür"""
        return len(self.vocab)
    
    def get_token_id(self, token):
        """Token adından ID'sini al"""
        return self.vocab.get(token, self.unk_id)
    
    def get_token_name(self, token_id):
        """Token ID'sinden adını al"""
        return self.reverse_vocab.get(token_id, f"<id:{token_id}>")
    
    def add_token(self, token, category="diğer"):
        """
        Yeni bir token ekle
        
        Args:
            token (str): Eklenecek token
            category (str): Tokenin kategorisi
            
        Returns:
            int: Token'in ID'si
        """
        # Token zaten varsa, mevcut ID'sini döndür
        if token in self.vocab:
            return self.vocab[token]
        
        # Yeni ID ata
        token_id = self.next_token_id
        self.next_token_id += 1
        
        # Vocab'a ekle
        self.vocab[token] = token_id
        self.reverse_vocab[token_id] = token
        
        # Kategoriye ekle
        if category not in self.vocab_categories:
            self.vocab_categories[category] = {}
        self.vocab_categories[category][token] = token_id
        
        return token_id
    
    def add_tokens(self, tokens_list, category="diğer"):
        """
        Birden fazla token ekle
        
        Args:
            tokens_list (list): Eklenecek tokenler listesi
            category (str): Tokenların kategorisi
            
        Returns:
            list: Eklenen tokenların ID'leri
        """
        token_ids = []
        for token in tokens_list:
            token_ids.append(self.add_token(token, category))
        return token_ids
    
    def save_vocab(self, output_file=None):
        """
        Güncellenmiş vocab'ı JSON dosyasına kaydet
        
        Args:
            output_file (str): Kaydedilecek dosya yolu (varsayılan: orijinal dosya)
        """
        if output_file is None:
            output_file = self.vocab_file
        
        with open(output_file, 'w', encoding=self.encoding) as f:
            json.dump(self.vocab_categories, f, ensure_ascii=False, indent=2)
    
    def get_learning_stats(self):
        """
        Öğrenme istatistiklerini al
        
        Returns:
            dict: Vocab boyutu, kategori sayısı vb.
        """
        return {
            "toplam_token_sayısı": len(self.vocab),
            "kategori_sayısı": len(self.vocab_categories),
            "kategoriler": list(self.vocab_categories.keys()),
            "sonraki_token_id": self.next_token_id,
            "auto_learn_aktif": self.auto_learn
        }