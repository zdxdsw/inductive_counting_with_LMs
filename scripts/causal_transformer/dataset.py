from torch.utils.data import Dataset
import json, os, sys
import torch
from typing import List

class dataset(Dataset):
    def __init__(self,
                 data_path: str,
                 max_len: int,
                 vocab: List[str],
                 split: str
                 ):
        super().__init__()
        self.data = []
        with open(os.path.join(data_path, f"{split}.txt"), 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.data.append(json.loads(l))
        self.w2i = {w:i for i,w in enumerate(vocab)}
        self.max_len = max_len

    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, i): 
        i, l = self.data[i]
        input_id = [self.w2i[w] for w in i]
        input_id += [self.w2i['<pad>']] * (self.max_len - len(input_id))
        label = [self.w2i[w] for w in l]
        label + [-1] * (self.max_len - len(label))
        return {
            'input_id': torch.LongTensor(input_id),
            'label': torch.LongTensor(label)
        }