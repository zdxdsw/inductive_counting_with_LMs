import os, sys, json, random, io, pytz, argparse, pytz, re
import numpy as np
from tqdm import tqdm
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from config import *
from dataset import dataset
from model import Causal_Transformer
from functools import partial
from pprint import pprint
from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers.optimization import get_constant_schedule_with_warmup
import torch.nn.functional as F


def Print(s):
   if not Accelerator().process_index:
      print(s)


parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default="debug")
parser.add_argument('--task', type=str)
args = parser.parse_args()


Print("----------- Preparing Config and Accelerator -----------")

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

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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


Print("-------- Preparing model --------")

model = Causal_Transformer(config)
Print(f"Total parameters: {model.num_parameters()}")
#Print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")
optimizer = torch.optim.AdamW(model.unet.parameters(), lr=config.learning_rate)

lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes / config.gradient_accumulation_steps,
)


Print("------------ Preparing data ------------")

if isinstance(config.data_dir, str): config.data_dir = [config.data_dir]
train_data = concatenate_datasets([dataset(
    data_path = config.data_path,
    max_len = config.max_position_embeddings,
    vocab = config.vocab,
    split = "train"
)])

val_data = concatenate_datasets([dataset(
    data_path = config.data_path,
    max_len = config.max_position_embeddings,
    vocab = config.vocab,
    split = "val"
)])

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.train_batch_size)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=config.eval_batch_size)

Print(f"num train = {len(train_data)}")
Print(f"num val = {len(val_data)}")

# if config.load_from_dir is not None:
#     already_trained_epochs = int(config.load_from_dir.split("-")[-1]) // (len(train_data) / config.per_device_train_batch_size / config.gradient_accumulation_steps / 4)
#     Print(f"\nResume from {config.load_from_dir} Epoch {int(already_trained_epochs)}\n")


Print("------------ Prepare Initialization ------------")

model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

global_step = 0
global_epoch = 0

# Resume from ckpt
if config.load_from_dir is not None:
    raise NotImplementedError
    ckpt_dir = os.path.join(config.ckpt_dir, config.load_from_dir, "ckpts")
    load_from_pt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[0]))[-1]

    global_epoch = int(load_from_pt.split("_")[0]) + 1
    global_step = int(load_from_pt.split("_")[1]) + 1

    accelerator.print(f"resume from ckpt: {load_from_pt}\n\tepoch {global_epoch} step {global_step}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=accelerator.device)
    unet = accelerator.unwrap_model(model).unet
    unet.load_state_dict(state_dict, strict=False)

if config.init_from_ckpt is not None:
    raise NotImplementedError
    init_from_ckpt = os.path.join(config.ckpt_dir, config.init_from_ckpt)
    accelerator.print(f"init from ckpt: {init_from_ckpt}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(init_from_ckpt, map_location=accelerator.device)
    unet = accelerator.unwrap_model(model).unet
    unet.load_state_dict(state_dict, strict=False)


Print(f"------------ Start job {config.date} ------------")

progress_bar = tqdm(total=config.num_epochs-global_epoch, disable=not accelerator.is_local_main_process)
for epoch in range(global_epoch, config.num_epochs):

    for step, batch in enumerate(train_dataloader):
        
        if global_step % config.save_image_steps == 0: 
            texts = next(test_dataiter)['sentence']
            #print(sentence)
            with accelerator.autocast():
                model_to_eval = accelerator.unwrap_model(model)
                model_to_eval.eval()
                model_to_eval.inference(
                    f"{epoch}_{global_step}", 
                    texts, 
                    os.path.join(config.output_dir, config.date),
                    save = accelerator.is_local_main_process,
                    disable_pgbar = not accelerator.is_local_main_process
                )
                model_to_eval.train()
        
        accelerator.wait_for_everyone()
        with accelerator.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            optimizer.zero_grad()
            noise_pred, noise = model(
                batch['image'],
                batch['sentence']
            )
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss / config.gradient_accumulation_steps)

            if (global_step+1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
            

        
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1
        #break
    progress_bar.update(1)
    if accelerator.is_main_process and (epoch+1) % config.save_model_epochs == 0:
        save_path = os.path.join(config.ckpt_dir, config.date, f"ckpts/{epoch}_{global_step}_unet.pt")
        torch.save(accelerator.unwrap_model(model).unet.state_dict(), save_path)
    if epoch == config.num_epochs - 1:
        texts = next(test_dataiter)['sentence']
        with accelerator.autocast():
            model_to_eval = accelerator.unwrap_model(model)
            model_to_eval.eval()
            model_to_eval.inference(
                f"{epoch}_{global_step}", 
                texts, 
                os.path.join(config.output_dir, config.date),
                save = accelerator.is_local_main_process,
                disable_pgbar = not accelerator.is_local_main_process
            )
            model_to_eval.train()
    accelerator.wait_for_everyone()

accelerator.print(f"Finish!!! {config.date}")