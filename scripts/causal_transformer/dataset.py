from torch.utils.data import Dataset
import json, os, sys, random
import torch
from typing import List

def sequences_collator(texts, w2i, max_len, posemb_shift=False):
    input_ids = []
    labels = []
    position_ids = []
    for t in texts:
        i, l = json.loads(t['text'])
        input_id = [w2i[w] for w in i]
        input_id += [w2i['<pad>']] * (max_len - len(input_id))
        label = [w2i[w] for w in l]
        label += [-1] * (max_len - len(label))
        shift_value = random.randint(0, max_len - len(i))
        position_id = list(range(shift_value, shift_value + len(i)))
        position_id += [0] * (max_len - len(position_id))

        input_ids.append(input_id)
        labels.append(label)
        position_ids.append(position_id)
        
    if not posemb_shift: position_ids = None
    else: position_ids = torch.LongTensor(position_ids)
    return {
        'input_id': torch.LongTensor(input_ids),
        'label': torch.LongTensor(labels),
        'position_id': position_ids,
    }



# class sequences_dataset(Dataset):
#     def __init__(self,
#                  data_path: str,
#                  max_len: int,
#                  vocab: List[str],
#                  split: str
#                  ):
#         super().__init__()
#         self.data = []
#         with open(os.path.join(data_path, f"{split}.txt"), 'r') as f:
#             lines = f.readlines()
#             for l in lines:
#                 self.data.append(json.loads(l))
#         self.w2i = {w:i for i,w in enumerate(vocab)}
#         self.max_len = max_len

    
#     def __len__(self): return len(self.data)
    
#     def __getitem__(self, idx): 
#         i, l = self.data[idx]
#         input_id = [self.w2i[w] for w in i]
#         input_id += [self.w2i['<pad>']] * (self.max_len - len(input_id))
#         label = [self.w2i[w] for w in l]
#         label + [-1] * (self.max_len - len(label))
#         #attention_mask = [1] * len(i) + [0] * len(self.max_len - len(i))
#         return {
#             'input_id': torch.LongTensor(input_id),
#             'label': torch.LongTensor(label),
#             #'attention_mask': torch.LongTensor(attention_mask),
#         }