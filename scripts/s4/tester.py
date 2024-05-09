import json, os, math, sys, random, re, pytz, argparse, warnings, time
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
import torch
from model import S4Model, LSTMModel, RNNModel
from config import *
sys.path.append("../")
from causal_transformer.utils import trim_task, get_acc
from s4_utils import sequences_collator

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

if os.path.exists('/data/yingshac/'): 
    os.environ['HF_HOME'] = '/data/yingshac/hf_cache'

parser = argparse.ArgumentParser()
parser.add_argument('--handle', type=str)
parser.add_argument('--load_from_epochs', type=str, default="all") # str of space separated ints
parser.add_argument('--test_files', type=str, default=None)
parser.add_argument('--loop', type=int, default=1) # number of times to loop through the test_dataloader
args = parser.parse_args()

""" ------------------------ Prepare Config ------------------------ """
config = Basic_Config()
default_config = Default_Config()
device="cuda" 
load_from_config = json.load(open(os.path.join(config.output_dir, args.handle, "config.json"), "r"))
config_keys = dir(config)
for k in config_keys:
    if k.startswith("__"): continue
    if k not in ["warmup_steps", "learning_rate", "num_epochs", "save_every_steps", "eval_every_steps", "logging_steps", "load_from_dir", "date", "data_path", "per_device_eval_batch_size"]:
        if k in load_from_config: setattr(config, k, load_from_config[k])
        else:
            setattr(config, k, default_config.__getattribute__(k))
            warnings.warn(f"Cannot find {k} in the resume_from_config. Set to {default_config.__getattribute__(k)} by default.")

model = eval(f"{config.model}Model")(config)
model = model.to(device)
model.eval()

if not config.task:
    if 'data_path' in dir(config): config.task = config.data_path.split("/")[-1]
    else: raise ValueError("Cannot find task from the config file: "+args.handle)
ckpt_dir = os.path.join(config.ckpt_dir, args.handle, "ckpts")
avail_ckpts = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[1]))
if args.load_from_epochs != "all":
    avail_ckpts = [ckpt for ckpt in avail_ckpts if int(ckpt.split("_")[0]) in [int(e) for e in args.load_from_epochs.split()]]

val_file = open(f"{config.eval_data_path}/{trim_task(config.task)}/val.txt", "r").readlines()
args.max_seen_len = max([len([x for x in json.loads(l)[0] if x != "<pad>"]) for l in val_file])
print(f"max_seen_len for {config.task} = {args.max_seen_len}")


""" -------------------- Prepare Reusable Variables -------------------- """
data_path = f"{config.eval_data_path}/{trim_task(config.task)}"
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

collator = partial(sequences_collator, 
                w2i={w:i for i,w in enumerate(config.vocab)}, 
                max_seq_len=config.max_seq_len,
                max_position_embeddings=config.max_seq_len,
                augmentation=None,
                )
if args.test_files is None:
    if "test_files" in load_from_config: args.test_files = load_from_config["test_files"]
    else: 
        args.test_files = "val ood_test"
        warnings.warn("No test_files specified, defaulting to 'val ood_test'")


""" ----------------------------- Testing ----------------------------- """
for load_from_pt in avail_ckpts:
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    last_date = ""
    for split in args.test_files.split():
        test_data = load_dataset(
                            "text", 
                            data_files={split: f"{data_path}/{split}.txt"})

        test_dataloader = DataLoader(test_data[split], shuffle=False, batch_size=config.per_device_train_batch_size, collate_fn=collator)

        counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo, correct, demo = 0, 0, 0, 0, 0, 0, 0, 0
        test_losses = []
        
        testing_output = {}

        _date = datetime.now(timezone).strftime("%m%d_%H%M%S") + "_" + load_from_pt.split("_")[0]
        if _date == last_date: 
            time.sleep(1)
            _date = datetime.now(timezone).strftime("%m%d_%H%M%S") + "_" + load_from_pt.split("_")[0]
        last_date = _date
        
        k = 0
        for loop in range(args.loop):
            for i, batch in enumerate(test_dataloader):
                
                logits = model(batch['input_id'].to(device))

                loss = criterion(
                    logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                    batch['label'].view(-1).to(device), # 1, bs*seq_len
                )
                test_losses.append(loss.detach().item())

                _counting_correct, _counting_demo, _last_correct, _last_demo, _unseen_len_correct, _unseen_len_demo = get_acc(
                    logits.detach().cpu(), 
                    batch['label'].detach().cpu(), 
                    ignore_index=-1,
                    max_seen_len=args.max_seen_len
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
                    input_seq = [config.vocab[i] for i in input_id if config.vocab[i]!='<pad>']
                    gth_seq = [config.vocab[gth_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1]
                    pred_seq = [config.vocab[pred_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1][:len(gth_seq)]
                    testing_output[k] = {
                        "input": " ".join(input_seq),
                        "gth": " ".join(gth_seq),
                        "pred": " ".join(pred_seq),
                    }
                    k+=1
            
        print(f""" {split} acc, load from {load_from_pt}
                | Test Loss: {round(np.mean(test_losses), 4)} 
                | Test Acc: {round(correct/demo, 4)} 
                | Test Counting Acc: {round(counting_correct/counting_demo, 4)} 
                | Test Last Acc: {round(last_correct/last_demo, 4)}
                | Test Unseen Len Acc: {round(unseen_len_correct/unseen_len_demo, 4) if unseen_len_demo != 0 else -1}
            """)

        save_dir = "test_samples" if "test" in split else "val_samples"
        os.makedirs(f"{config.output_dir}/{args.handle}/{save_dir}", exist_ok=True)
        json.dump({
                "test_data_file":  f"{data_path}/{split}.txt",
                "load_from": f"{args.handle}/{load_from_pt}",
                "test_acc": round(correct/demo, 4),
                "test_counting_acc": round(counting_correct/counting_demo, 4),
                "test_last_acc": round(last_correct/last_demo, 4),
                "test_unseen_len_acc": round(unseen_len_correct/unseen_len_demo, 4) if unseen_len_demo != 0 else -1,
                "test_loss": round(np.mean(test_losses), 4),
                "testing_output": testing_output,
            }, 
            open(f"{config.output_dir}/{args.handle}/{save_dir}/{_date}.json", "w"), indent=2)
    