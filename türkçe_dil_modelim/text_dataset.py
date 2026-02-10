import torch
from torch.utils.data import Dataset

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



