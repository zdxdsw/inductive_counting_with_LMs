from torch.utils.data import Dataset
import json, os, sys, random
import torch
from typing import List

def construct_position_id(i, max_seq_len, max_position_embeddings, augmentation):
    effective_len_i = len([x for x in i if not x in ['<pad>', '<blk>']]) + 1
    
    if augmentation is None:  effective_position_id = list(range(effective_len_i))
    if augmentation == "shift":
        shift_value = random.randint(0, max_position_embeddings - effective_len_i)
        effective_position_id = list(range(shift_value, shift_value + effective_len_i))
    elif augmentation == "randomized":
        effective_position_id = sorted(random.sample(range(max_position_embeddings), effective_len_i))
    elif augmentation == "zooming":
        effective_position_id = [x/effective_len_i for x in range(1, effective_len_i+1)]

    position_id = []
    effective_p = 0
    for x in i:
        #try: 
        position_id.append(effective_position_id[effective_p])
        # except: 
        #     print(i)
        #     print(effective_len_i, effective_position_id)
        #     raise
        if not x in ['<pad>', '<blk>']: effective_p += 1
    
    position_id += [0] * (max_seq_len - len(position_id))
    #print(position_id)
    return position_id


def sequences_collator(texts, w2i, max_seq_len, max_position_embeddings, augmentation=None):
    input_ids = []
    labels = []
    position_ids = []
    attention_masks = []
    for t in texts:
        i, l = json.loads(t['text'])
        
        input_id = [w2i[w] for w in i]
        input_id += [w2i['<pad>']] * (max_seq_len - len(input_id))

        label = [w2i[w] if not w == '-1' else -1 for w in l]
        label += [-1] * (max_seq_len - len(label))

        position_ids.append(construct_position_id(i, max_seq_len, max_position_embeddings, augmentation))

        attention_mask = [1 if not w == '<pad>' else 0 for w in i]
        attention_mask += [0] * (max_seq_len - len(attention_mask))

        input_ids.append(input_id)
        labels.append(label)
        attention_masks.append(attention_mask)
        
    return {
        'input_id': torch.LongTensor(input_ids),
        'label': torch.LongTensor(labels),
        'position_id': torch.tensor(position_ids),
        'attention_mask': torch.LongTensor(attention_masks),
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