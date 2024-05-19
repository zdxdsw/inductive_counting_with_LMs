import torch, json
def sequences_collator(texts, w2i, max_seq_len, max_position_embeddings=None, augmentation=None):
    input_ids = []
    labels = []
    for t in texts:
        i, l = json.loads(t['text'])
        
        input_id = [w2i[w] for w in i]
        input_id += [w2i['<pad>']] * (max_seq_len - len(input_id))

        label = [w2i[w] if not w == '-1' else -1 for w in l]
        label += [-1] * (max_seq_len - len(label))

        input_ids.append(input_id)
        labels.append(label)
        
    return {
        'input_id': torch.LongTensor(input_ids),
        'label': torch.LongTensor(labels),
        'position_id': torch.LongTensor([[]]),
        'attention_mask': torch.tensor([[]]),
    }