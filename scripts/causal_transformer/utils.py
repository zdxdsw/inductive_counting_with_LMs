import warnings, os, pytz, json
import numpy as np
from datetime import datetime
timezone = pytz.timezone('America/New_York') 

def get_acc(logits, labels, ignore_index, max_seen_len):
    pred = logits.argmax(dim=-1)

    counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo = 0, 0, 0, 0, 0, 0

    for _pred, _labels in zip(pred, labels):
        counting_pred, last_pred = _pred[_labels != ignore_index][:-1], _pred[_labels != ignore_index][-1].squeeze()
        counting_label, last_label = _labels[_labels != ignore_index][:-1], _labels[_labels != ignore_index][-1].squeeze()

        counting_correct += (counting_pred == counting_label).float().sum().item()
        counting_demo += counting_label.numel()
        last_correct += (last_pred == last_label).float().sum().item()
        last_demo += last_label.numel()

        unseen_len_correct += (_pred[_labels != ignore_index] == _labels[_labels != ignore_index])[max_seen_len:].float().sum().item()
        unseen_len_demo += _labels[_labels != -1][max_seen_len:].numel()

    return counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo


from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "#Params", "Param shape"])
    total_params = 0
    for name, parameter in model.named_parameters():
        #if not parameter.requires_grad: continue
        params = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, params, param_shape])
        total_params+=params
    print(table)
    print(f"Total Params: {total_params}")


def check_config(config):
    if not (config.absolute_posemb or config.rotary_posemb):
        warnings.warn("========== No positional embedding is used in the model. Essentially we're doing NoPE! ==========")
    if config.absolute_posemb_shift and config.absolute_posemb_rdmz:
        raise ValueError("========== You cannot use both shift and randomized augmentation for positional embeddings ==========")
    if config.rotary_posemb_shift and config.rotary_posemb_rdmz:
        raise ValueError("========== You cannot use both shift and randomized augmentation for positional embeddings ==========")

def trim_task(task):
    return task.replace("_addbigram", "").replace("_addtable", "")

def inference(model_to_eval, dataloader, criterion, device, max_seen_len, vocab):
    
    counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo, correct, demo = 0, 0, 0, 0, 0, 0, 0, 0
    losses = []
    testing_output = {}
    k = 0
    for i, batch in enumerate(dataloader):
        position_ids = None
        if batch['position_id'] is not None: position_ids = batch['position_id'].to(device)
        logits = model_to_eval(
            batch['input_id'].to(device),
            position_ids = position_ids,
            attention_mask = batch['attention_mask'].to(device),
        )
        
        batch['label'] = batch['label'].to(device)
        loss = criterion(
            logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
            batch['label'].view(-1), # 1, bs*seq_len
        )
        losses.append(loss.detach().item())
        _counting_correct, _counting_demo, _last_correct, _last_demo, _unseen_len_correct, _unseen_len_demo = get_acc(
            logits.detach().cpu(), 
            batch['label'].detach().cpu(), 
            ignore_index=-1,
            max_seen_len=max_seen_len
        )
        counting_correct += _counting_correct
        counting_demo += _counting_demo
        last_correct += _last_correct
        last_demo += _last_demo
        unseen_len_correct += _unseen_len_correct
        unseen_len_demo += _unseen_len_demo
        correct += (_counting_correct + _last_correct)
        demo += (_counting_demo + _last_demo)

        for input_id, gth_id, pred_id in zip(batch['input_id'], batch['label'], logits.argmax(dim=-1)):
            input_seq = [vocab[i] for i in input_id if vocab[i]!='<pad>']
            gth_seq = [vocab[gth_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1]
            pred_seq = [vocab[pred_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1][:len(gth_seq)]
            testing_output[k] = {
                "input": " ".join(input_seq),
                "gth": " ".join(gth_seq),
                "pred": " ".join(pred_seq),
            }
            k+=1

    avg_loss = round(np.mean(losses), 4)
    avg_acc = round(correct/demo, 4)
    avg_counting_acc = round(counting_correct/counting_demo, 4)
    avg_last_acc = round(last_correct/last_demo, 4)
    if unseen_len_demo == 0:
        avg_unseen_len_acc = -1
    else:
        avg_unseen_len_acc = round(unseen_len_correct/unseen_len_demo, 4)
    print(f"k = {k}, last_demo = {last_demo}")

    return avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc
    