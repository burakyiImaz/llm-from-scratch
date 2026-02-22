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


        self.length= (len(self.data)-context_length)// stride

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        start= idx*self.stride
        end= start + self.context_length

        x= self.data[start:end]
        y= self.data[start+1:end+1]

        return x, y