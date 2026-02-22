import torch
from torch.utils.data import Dataset 

class TextDataset(Dataset):
    def __init__(self,token_ids, context_length, stride=1):
        super().__init__()

        if isinstance(token_ids, list):
            token_ids= torch.tensor(token_ids, dtype=torch.long)

        self.data= token_ids
        self.context_length= context_length
        self.stride= stride


        self