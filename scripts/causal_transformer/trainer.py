import os, sys, json, random, io, pytz, argparse, pytz, re
import numpy as np
from tqdm import tqdm
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from config import *
from dataset import sequences_collator
from datasets import load_dataset
from model import Causal_Transformer
from functools import partial
from utils import *
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_constant_schedule_with_warmup
from collections import defaultdict, Counter
os.environ['HF_HOME'] = '/data/yingshac/hf_cache'


def Print(s):
   if not Accelerator().process_index:
      print(s)


parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default="debug")
parser.add_argument('--task', type=str)
args = parser.parse_args()


"""----------- Preparing Accelerator -----------"""

config = eval(f"{args.task}_Config")()
tmp = json.load(open(os.path.join(config.output_dir, args.date, "config.json"), "r"))
for k, v in tmp.items():
    setattr(config, k, v)
config.date = args.date
check_config(config)

# Fix all seeds to ensure reproducibility
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ddp_kwargs = DistributedDataParallelKwargs()
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    kwargs_handlers=[ddp_kwargs],
    project_dir=os.path.join(config.output_dir, config.date),
)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.init_trackers("tensorboard")
    os.makedirs(os.path.join(config.ckpt_dir, config.date, "ckpts"), exist_ok=True)


Print("------------- Preparing model -------------")

model = Causal_Transformer(config)
Print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
#Print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.warmup_steps * accelerator.num_processes / config.gradient_accumulation_steps,
)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)


Print("------------- Preparing data -------------")

if isinstance(config.data_path, str): config.data_path = [config.data_path]
text_datasets = [load_dataset(
                    "text", 
                    data_files={
                        "train": f"{path}/train.txt",
                        "validation": f"{path}/val.txt"
                        })
                    for path in config.data_path
                ]

train_data = concatenate_datasets([D['train'] for D in text_datasets])
val_data = concatenate_datasets([D['validation'] for D in text_datasets])

if config.absolute_posemb_shift or config.rotary_posemb_shift:
    augmentation = "shift"
elif config.absolute_posemb_rdmz or config.rotary_posemb_rdmz:
    augmentation = "randomized"
collator = partial(sequences_collator, 
                    w2i={w:i for i,w in enumerate(config.vocab)}, 
                    max_len=config.max_position_embeddings,
                    augmentation=augmentation,
                )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.per_device_train_batch_size, collate_fn=collator)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator)

Print(f"num train = {len(train_data)}")
Print(f"num val = {len(val_data)}")


"""------------ Prepare Initialization ------------"""

model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

global_step = 0
global_epoch = 0

# Resume from ckpt
if config.load_from_dir is not None:
    ckpt_dir = os.path.join(config.ckpt_dir, config.load_from_dir, "ckpts")
    load_from_pt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[1]))[-1]

    global_epoch = int(load_from_pt.split("_")[0]) + 1
    global_step = int(load_from_pt.split("_")[1]) + 1

    Print(f"resume from ckpt: {load_from_pt}\n\tepoch {global_epoch} step {global_step}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=accelerator.device)
    _model = accelerator.unwrap_model(model)
    _model.load_state_dict(state_dict, strict=True)

if config.init_from_ckpt is not None:
    init_from_ckpt = os.path.join(config.ckpt_dir, config.init_from_ckpt)
    accelerator.print(f"init from ckpt: {init_from_ckpt}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(init_from_ckpt, map_location=accelerator.device)
    _model = accelerator.unwrap_model(model)
    _model.load_state_dict(state_dict, strict=True)


Print(f"------------ Start job {config.date} ------------")

progress_bar = tqdm(total=(config.num_epochs-global_epoch)*len(train_dataloader), disable=not accelerator.is_local_main_process, mininterval=1)
for epoch in range(global_epoch, config.num_epochs):

    for step, batch in enumerate(train_dataloader):
        
        if global_step % config.eval_every_steps == 0: 
            
            with accelerator.autocast():
                counting_correct, counting_demo, last_correct, last_demo, correct, demo = 0, 0, 0, 0, 0, 0
                val_losses = []
                model_to_eval = accelerator.unwrap_model(model)
                model_to_eval.eval()
                for i, val_batch in enumerate(val_dataloader):
                    position_ids = None
                    if val_batch['position_id'] is not None: position_ids = val_batch['position_id'].to(accelerator.device)
                    logits = model_to_eval(
                        val_batch['input_id'].to(accelerator.device),
                        position_ids = position_ids,
                        attention_mask = val_batch['attention_mask'].to(accelerator.device),
                    )
                    loss = criterion(
                        logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                        val_batch['label'].view(-1), # 1, bs*seq_len
                    )
                    val_losses.append(loss.detach().item())
                    _counting_correct, _counting_demo, _last_correct, _last_demo = get_acc(logits.detach().cpu(), val_batch['label'].detach().cpu(), ignore_index=-1)
                    counting_correct += _counting_correct
                    counting_demo += _counting_demo
                    last_correct += _last_correct
                    last_demo += _last_demo
                    correct += (_counting_correct + _last_correct)
                    demo += (_counting_demo + _last_demo)
                Print(f"""Epoch {epoch} Step {global_step} 
                      | Val Loss: {round(np.mean(val_losses), 4)} 
                      | Val Acc: {round(correct/demo, 4)} 
                      | Val Counting Acc: {round(counting_correct/counting_demo, 4)} 
                      | Val Last Acc: {round(last_correct/last_demo, 4)}"""
                    )
                model_to_eval.train()
        
        accelerator.wait_for_everyone()
        with accelerator.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            optimizer.zero_grad()
            
            position_ids = None
            if batch['position_id'] is not None: position_ids = batch['position_id'].to(accelerator.device)
            
            logits = model(
                batch['input_id'].to(accelerator.device),
                position_ids = position_ids,
                attention_mask = batch['attention_mask'].to(accelerator.device),
            )

            loss = criterion(
                logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                batch['label'].view(-1),
            )

            accelerator.backward(loss / config.gradient_accumulation_steps)

            if (global_step+1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
            
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        accelerator.log(logs, step=global_step)
        global_step += 1

        if global_step % config.logging_steps == 0: progress_bar.set_postfix(**logs)
        progress_bar.update(1)
        
        # if accelerator.is_main_process and global_step % config.save_every_steps == 0:
        #     save_path = os.path.join(config.ckpt_dir, config.date, f"ckpts/{epoch}_{global_step}_transformer.pt")
        #     torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
        #break
    
    
    

    #if epoch == config.num_epochs - 1:
    with accelerator.autocast(): # validate at the end of every epoch
        counting_correct, counting_demo, last_correct, last_demo, correct, demo = 0, 0, 0, 0, 0, 0
        val_losses = []
        model_to_eval = accelerator.unwrap_model(model)
        model_to_eval.eval()
        for i, val_batch in enumerate(val_dataloader):

            position_ids = None
            if val_batch['position_id'] is not None: position_ids = val_batch['position_id'].to(accelerator.device)

            logits = model_to_eval(
                val_batch['input_id'].to(accelerator.device),
                position_ids = position_ids,
                attention_mask = val_batch['attention_mask'].to(accelerator.device),
            )
            
            loss = criterion(
                logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                val_batch['label'].view(-1),
            )
            val_losses.append(loss.detach().item())
            _counting_correct, _counting_demo, _last_correct, _last_demo = get_acc(logits.detach().cpu(), val_batch['label'].detach().cpu(), ignore_index=-1)
            counting_correct += _counting_correct
            counting_demo += _counting_demo
            last_correct += _last_correct
            last_demo += _last_demo
            correct += (_counting_correct + _last_correct)
            demo += (_counting_demo + _last_demo)
        Print(f"""Epoch {epoch} Step {global_step} 
                | Val Loss: {round(np.mean(val_losses), 4)} 
                | Val Acc: {round(correct/demo, 4)} 
                | Val Counting Acc: {round(counting_correct/counting_demo, 4)} 
                | Val Last Acc: {round(last_correct/last_demo, 4)}"""
            )
        model_to_eval.train()
    accelerator.wait_for_everyone()
    
    save_path = os.path.join(config.ckpt_dir, config.date, f"ckpts/{epoch}_{global_step}_transformer.pt")
    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)

accelerator.print(f"Finish!!! {config.date}")