
import json, torch
# def sequences_collator(texts, w2i, max_seq_len):
#     input_ids = []
#     labels = []
#     for t in texts:
#         i, l = json.loads(t['text'])
        
#         input_id = [w2i[w] for w in i]
#         input_id += [w2i['<pad>']] * (max_seq_len - len(input_id))

#         label = [w2i[w] if not w == '-1' else -1 for w in l]
#         label += [-1] * (max_seq_len - len(label))

#         input_ids.append(input_id)
#         labels.append(label)
        
#     return {
#         'input_id': torch.LongTensor(input_ids),
#         'label': torch.LongTensor(labels),
#     }

def convert_precision_config(p):
    if p == "fp32":
        accelerator_mix_precision = 'no'
        model_dtype = torch.float32
    elif p == "fp16":
        accelerator_mix_precision = 'fp16'
        model_dtype = torch.float16
    elif p == "bf16":
        accelerator_mix_precision = 'bf16'
        model_dtype = torch.bfloat16
    return (accelerator_mix_precision, model_dtype)