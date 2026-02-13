import torch
from torch.utils.data import Dataset, DataLoader
from türkçe_dil_modelim.tokenizer import Tokenizer
from pathlib import Path


class TextDataset(Dataset):
    def __init__(self, token_ids, context_length, stride):
        super().__init__()

        self.inputs = []
        self.targets = []

        # sadece TAM context_length olan pencereleri al
        for i in range(0, len(token_ids) - context_length, stride):

            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]

            # güvenlik: eksik parça varsa alma
            if len(input_chunk) < context_length or len(target_chunk) < context_length:
                continue

            self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
            self.targets.append(torch.tensor(target_chunk, dtype=torch.long))
    
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_data_loader(token_ids: list, context_length:int, stride:int, batch_size:int, shuffle:bool=True):
        dataset= TextDataset(token_ids, context_length, stride)
        data_loader= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
def process_and_save_batches(file_path, tokenizer, context_length=32, stride=1, batch_size=64, save_dir="data_batches"):
    save_dir= Path(save_dir)
    save_dir.mkdir(exists_ok=True)

    all_token_ids= []

    with open(file_path,"r",encoding="utf") as f:
        for line_idx, line in enumerate(f):
              tokens= tokenizer.encode(line)
              all_token_ids.extend(tokens.tolist())

              if (line_idx + 1) % 1000 == 0:
                print(f"{line_idx+1} satır işlendi, vocab boyutu: {tokenizer.get_vocab_size()}")
    
    tokenizer.save_vocab()

    dataset = TextDataset(all_token_ids, context_length, stride)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3️⃣ Batch batch kaydet
    batch_files = []
    for batch_idx, (x, y) in enumerate(data_loader):
        batch_file = save_dir / f"batch_{batch_idx}.pt"
        torch.save({"input": x, "target": y}, batch_file)
        batch_files.append(batch_file)
    
    print(f"Tüm batchler kaydedildi: {len(batch_files)} dosya")
    return batch_files

