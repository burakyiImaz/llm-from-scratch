import os
import json
from dataclasses import dataclass
from typing import Dict, Optional

from transformers import GenerationConfig

# 🔥 Custom model importları
from config import Config
from hf_model import CasualLM


@dataclass
class HFExporterConfig:
    model_path: str
    output_dir: str
    repo_id: Optional[str] = None
    model_max_length: int = 2048


class HFModelExporter:

    def __init__(self, config: HFExporterConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("Custom model yükleniyor...")

        # 🔥 AutoModel KULLANMIYORUZ
        self.model = CasualLM.from_pretrained(
            self.config.model_path,
            local_files_only=True
        )

        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                local_files_only=True
            )
        except Exception:
            print("Tokenizer bulunamadı. Tokenizer export edilmeyecek.")


    # ---------------------------------------------------
    # 1️⃣ Özel Token
    # ---------------------------------------------------
    def configure_special_tokens(self, special_tokens: Dict):

        if self.tokenizer is None:
            raise ValueError("Tokenizer yüklenmemiş.")

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print("Özel tokenler ayarlandı.")


    # ---------------------------------------------------
    # 2️⃣ Model Kaydet (HF native yöntem)
    # ---------------------------------------------------
    def save_model(self):

        # 🔥 HF native save_pretrained kullanıyoruz
        self.model.save_pretrained(
            self.config.output_dir,
            safe_serialization=True
        )

        print("Model safetensors olarak kaydedildi.")


    # ---------------------------------------------------
    # 3️⃣ Config Kaydet (model_type garanti)
    # ---------------------------------------------------
    def save_config(self):

        config_dict = self.model.config.to_dict()

        config_dict["model_type"] = "turkce_llm"

        if self.tokenizer:
            config_dict["vocab_size"] = len(self.tokenizer)
            config_dict["bos_token_id"] = self.tokenizer.bos_token_id
            config_dict["eos_token_id"] = self.tokenizer.eos_token_id
            config_dict["pad_token_id"] = self.tokenizer.pad_token_id

        with open(
            os.path.join(self.config.output_dir, "config.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print("config.json oluşturuldu.")


    # ---------------------------------------------------
    # 4️⃣ Tokenizer
    # ---------------------------------------------------
    def save_tokenizer(self):

        if self.tokenizer is None:
            return

        self.tokenizer.save_pretrained(self.config.output_dir)
        print("Tokenizer dosyaları oluşturuldu.")


    # ---------------------------------------------------
    # 5️⃣ Generation Config
    # ---------------------------------------------------
    def save_generation_config(self):

        if self.tokenizer is None:
            return

        gen_config = GenerationConfig(
            max_length=self.config.model_max_length,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        gen_config.save_pretrained(self.config.output_dir)
        print("generation_config.json oluşturuldu.")


    # ---------------------------------------------------
    # 6️⃣ README
    # ---------------------------------------------------
    def save_readme(self):

        model_name = self.config.repo_id or "local-model"

        content = f"""---
license: apache-2.0
language:
- tr
pipeline_tag: text-generation
library_name: transformers
---

# {model_name}

Türkçe causal language model.

## Model Özellikleri

- Architecture: turkce_llm
- Format: safetensors
- Max Length: {self.config.model_max_length}

## Kullanım

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Merhaba dünya"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))"""
    with open(
        os.path.join(self.config.output_dir, "README.md"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content)

    print("README.md oluşturuldu.")


# ---------------------------------------------------
# 7️⃣ FULL EXPORT
# ---------------------------------------------------
def export_all(self, special_tokens: Optional[Dict] = None):

    if special_tokens:
        self.configure_special_tokens(special_tokens)

    self.save_model()
    self.save_config()
    self.save_tokenizer()
    self.save_generation_config()
    self.save_readme()

    print("\n HuggingFace klasörü hazır!")
    print("Çıktı:", self.config.output_dir)

