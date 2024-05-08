import os, sys, json, random, io, pytz, argparse, pytz, re
import numpy as np
from tqdm import tqdm
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from functools import partial
from model import S4Model, LSTMModel
from config import *
from s4_utils import sequences_collator
sys.path.append("../")
from causal_transformer.config_taskspecific import *
from causal_transformer.utils import trim_task, inference
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_constant_schedule_with_warmup
from collections import defaultdict, Counter

if os.path.exists('/data/yingshac/'): 
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


Print(f"------------- Preparing {config.model} model -------------")

model = eval(f"{config.model}Model")(config)
Print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if (not hasattr(p, "_optim")) and (p.requires_grad)]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer, lr_scheduler = setup_optimizer(
    model, lr=config.learning_rate, weight_decay=config.weight_decay, epochs=config.num_epochs
)


Print("------------- Preparing data -------------")

tasks = [trim_task(args.task)] + config.aux_tasks

train_data = concatenate_datasets(
                    [load_dataset(
                            "text", 
                            data_files={"train": f"{config.train_data_path}/{task}/train.txt"}
                            )['train'] for task in tasks]
                )
val_data = load_dataset(
                    "text", 
                    data_files={"validation": f"{config.eval_data_path}/{trim_task(args.task)}/val.txt"}
                    )['validation']

args.max_seen_len = max([len([x for x in json.loads(l['text'])[0] if x != "<pad>"]) for l in val_data])
Print(f"max_seen_len for {args.task} = {args.max_seen_len}")

collator = partial(sequences_collator, 
                    w2i={w:i for i,w in enumerate(config.vocab)}, 
                    max_seq_len=config.max_seq_len,
                    max_position_embeddings=config.max_seq_len,
                    augmentation=None,
                )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.per_device_train_batch_size, collate_fn=collator)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator)

Print(f"num train = {len(train_data)}")
Print(f"num val = {len(val_data)}")

test_dataloaders = {}
for ood_test_file in config.test_files:
    test_data = load_dataset("text",  data_files={ood_test_file: f"{config.eval_data_path}/{trim_task(args.task)}/{ood_test_file}.txt"})
    test_dataloaders[ood_test_file] = DataLoader(test_data[ood_test_file], shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator)


"""------------ Prepare Initialization ------------"""

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
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
progress_bar = tqdm(total=(config.num_epochs-global_epoch)*len(train_dataloader), disable=not accelerator.is_local_main_process, mininterval=10)
for epoch in range(global_epoch, config.num_epochs):

    for step, batch in enumerate(train_dataloader):
        
        if global_step % config.eval_every_steps == 0: 
            if accelerator.is_main_process:
                with accelerator.autocast():
                    
                    model_to_eval = accelerator.unwrap_model(model)
                    model_to_eval.eval()

                    # Inference on validation set
                    avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = inference(
                        model_to_eval,
                        val_dataloader,
                        criterion,
                        accelerator.device,
                        max_seen_len=args.max_seen_len,
                        vocab=config.vocab,
                    )
                    Print(f"""Epoch {epoch} Step {global_step} 
                        | Val Loss: {avg_loss} 
                        | Val Acc: {avg_acc} 
                        | Val Counting Acc: {avg_counting_acc} 
                        | Val Last Acc: {avg_last_acc}
                        | Val Unseen Len Acc: {avg_unseen_len_acc}"""
                    )
                    
                    # Inference on testing sets
                    for ood_test_file, test_dataloader in test_dataloaders.items():
                        avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = inference(
                            model_to_eval,
                            test_dataloader,
                            criterion,
                            accelerator.device,
                            max_seen_len=args.max_seen_len,
                            vocab=config.vocab,
                        )
                        Print(f"""Epoch {epoch} Step {global_step} {ood_test_file}.txt
                            | Test Loss: {avg_loss}
                            | Test Acc: {avg_acc}
                            | Test Counting Acc: {avg_counting_acc}
                            | Test Last Acc: {avg_last_acc}
                            | Test Unseen Len Acc: {avg_unseen_len_acc}"""
                        )
                    
                    model_to_eval.train()
        
        accelerator.wait_for_everyone()
        with accelerator.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            optimizer.zero_grad()
                        
            logits = model(
                batch['input_id'].to(accelerator.device),
            )

            loss = criterion(
                logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                batch['label'].view(-1),
            )

            accelerator.backward(loss / config.gradient_accumulation_steps)

            if (global_step+1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                #lr_scheduler.step() # Currently disable lr_scheduler
            
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        accelerator.log(logs, step=global_step)
        global_step += 1

        if global_step % config.logging_steps == 0: progress_bar.set_postfix(**logs)
        progress_bar.update(1)
        
    

    with accelerator.autocast(): # validate at the end of every epoch
        if accelerator.is_main_process:
            model_to_eval = accelerator.unwrap_model(model)
            model_to_eval.eval()
            
            # Inference on validation set
            avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = inference(
                model_to_eval,
                val_dataloader,
                criterion,
                accelerator.device,
                max_seen_len=args.max_seen_len,
                vocab=config.vocab,
            )
            Print(f"""Epoch {epoch} Step {global_step} 
                    | Val Loss: {avg_loss} 
                    | Val Acc: {avg_acc} 
                    | Val Counting Acc: {avg_counting_acc} 
                    | Val Last Acc: {avg_last_acc}
                    | Val Unseen Len Acc: {avg_unseen_len_acc}"""
            )

            
            # Inference on testing sets
            for ood_test_file, test_dataloader in test_dataloaders.items():
                avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = inference(
                    model_to_eval,
                    test_dataloader,
                    criterion,
                    accelerator.device,
                    max_seen_len=args.max_seen_len,
                    vocab=config.vocab,
                )
                Print(f"""Epoch {epoch} Step {global_step} {ood_test_file}.txt
                    | Test Loss: {avg_loss}
                    | Test Acc: {avg_acc}
                    | Test Counting Acc: {avg_counting_acc}
                    | Test Last Acc: {avg_last_acc}
                    | Test Unseen Len Acc: {avg_unseen_len_acc}"""
                )

            model_to_eval.train()
    accelerator.wait_for_everyone()
    
    save_path = os.path.join(config.ckpt_dir, config.date, f"ckpts/{epoch}_{global_step}_{config.model}.pt")
    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)

if accelerator.is_main_process:
    Print("\n\n------------- Start Inference -------------")
    infr_command = "python tester.py --handle {} --test_files \"{}\"".format(config.date, " ".join(["val"]+config.test_files))
    os.system(infr_command)
accelerator.print(f"Finish!!! {config.date}")
