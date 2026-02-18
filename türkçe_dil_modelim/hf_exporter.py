import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from safetensors.torch import save_file


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

        print("Model yükleniyor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path
        )


    def configure_special_tokens(self, special_tokens: Dict):

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print("Özel tokenler ayarlandı.")



    def save_model(self):

        state_dict = self.model.state_dict()
        save_file(
            state_dict,
            os.path.join(self.config.output_dir, "model.safetensors")
        )

        print("Model safetensors olarak kaydedildi.")



    def save_config(self):

        config = self.model.config.to_dict()

        config["vocab_size"] = len(self.tokenizer)
        config["bos_token_id"] = self.tokenizer.bos_token_id
        config["eos_token_id"] = self.tokenizer.eos_token_id
        config["pad_token_id"] = self.tokenizer.pad_token_id

        with open(
            os.path.join(self.config.output_dir, "config.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("config.json oluşturuldu.")
    


    def save_tokenizer(self):

        self.tokenizer.save_pretrained(self.config.output_dir)

        print("Tokenizer dosyaları oluşturuldu.")



    def save_generation_config(self):

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



    def save_readme(self):

        content = f"""# HuggingFace Model

Bu model causal language model olarak export edilmiştir.

## Vocab Size
{len(self.tokenizer)}

## Special Tokens

| Token | ID |
|-------|----|
| BOS | {self.tokenizer.bos_token_id} |
| EOS | {self.tokenizer.eos_token_id} |
| PAD | {self.tokenizer.pad_token_id} |

## Kullanım

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{self.config.repo_id or 'local_path'}")
model = AutoModelForCausalLM.from_pretrained("{self.config.repo_id or 'local_path'}")

inputs = tokenizer("Merhaba dünya", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0]))"""
        with open(
            os.path.join(self.config.output_dir, "README.md"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write(content)

        print("README.md oluşturuldu.")

    def export_all(self, special_tokens: Optional[Dict] = None):

        if special_tokens:
            self.configure_special_tokens(special_tokens)

        self.save_model()
        self.save_tokenizer()
        self.save_config()
        self.save_generation_config()
        self.save_readme()

        print("\n HuggingFace klasörü hazır!")
        print("Çıktı:", self.config.output_dir)




        

