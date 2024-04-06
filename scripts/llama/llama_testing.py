import os, sys, json, random, io, pytz, argparse, pytz, re, math
os.environ['HF_HOME'] = '/data/yingshac/hf_cache'
import numpy as np
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from datasets.features import Features, Value
from torch.utils.data import DataLoader
from config import FinetuningConfig
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig,
        PeftConfig,
        PeftModel,
        AutoPeftModelForCausalLM,
    )
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import *
from pprint import pprint
from functools import partial
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

# Fix all seeds to ensure reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_handle', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--specific_epoch', type=int)
parser.add_argument('--max_seq_length', type=int)
parser.add_argument('--split', type=str, default="validation")
parser.add_argument('--batch_size', type=int)

args = parser.parse_args()

def Print(s):
   if not Accelerator().process_index:
      print(s)

Print("-------- Preparing configs --------")
config = FinetuningConfig()
config.date = date
if args.ckpt_handle:
    ckpt_config = json.load(open(os.path.join(config.output_dir, args.ckpt_handle, "config.json"), "r"))
    for k, v in ckpt_config.items():
        setattr(config, k, v)
else:
    config.output_dir = os.path.join(config.output_dir, "zeroshot")
    config.date = ""

Print(f"config.date = {config.date}")

if args.data_dir is not None: config.data_dir = args.data_dir
if args.max_seq_length is not None: config.max_seq_length = args.max_seq_length
if args.batch_size is not None: config.per_device_eval_batch_size = args.batch_size

os.makedirs(os.path.join(config.output_dir, config.date, "test_samples"), exist_ok=True)


with open(config.prompt_path, "r") as f: prompt_template = "\n".join(f.readlines()).strip()

Print("-------- Preparing model and tokenizer --------")
if args.ckpt_handle is None:
    Print("          Zero-shot Inference          \n")
    model = LlamaForCausalLM.from_pretrained(config.model_name,
                                             load_in_8bit=config.bits==8,
                                                torch_dtype=torch.float16,
                                                device_map={"": Accelerator().process_index},
                                                )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # Turn-off an annoying warning msg
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'right'

else:
    all_ckpts = sorted(
            os.listdir(os.path.join(config.ckpt_dir, args.ckpt_handle, "ckpts")),
            key = lambda x: int(x.split("-")[-1])
        )
    steps_per_epoch = math.gcd(int(all_ckpts[0].split("-")[-1]), int(all_ckpts[1].split("-")[-1]))
    if args.specific_epoch is None:
        #tmp = "0"
        args.specific_ckpt = all_ckpts[-1]
        args.specific_epoch = int(all_ckpts[-1].split("-")[-1])
        #for f in os.listdir(os.path.join(config.output_dir, args.ckpt_handle, "ckpts")):
        #    if int(f.split("-")[-1]) > int(tmp.split("-")[-1]): tmp = f
        #args.specific_ckpt = tmp
    else:
        args.specific_ckpt = all_ckpts[args.specific_epoch-int(all_ckpts[0].split("-")[-1]) // steps_per_epoch]
    
    lora_adapter = os.path.join(config.ckpt_dir, args.ckpt_handle, "ckpts", args.specific_ckpt)
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_adapter, 
        torch_dtype=torch.float16, 
        load_in_8bit=config.bits==8,
        device_map={"": Accelerator().process_index}
    )
    Print(f"steps_per_epoch = {steps_per_epoch}")
    Print(f"Load LoRA-finetuned weights {args.ckpt_handle}/ckpts/{args.specific_ckpt}, ========== epoch {args.specific_epoch} ==========\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # Turn-off an annoying warning msg
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.padding_side = 'right'

model.resize_token_embeddings(len(tokenizer))


Print("-------- Preparing data --------")

val_data = load_dataset(config.data_dir, 
                        split=args.split, 
                        features=Features({
                            'input_str':Value("string"), 
                            "answer":Value("string")}),
                        )
Print(f"num val = {len(val_data)}")

i = random.choice(list(range(len(val_data))))
Print("\nexample data: ")
Print(preprocess(prompt_template, val_data[i]["input_str"], val_data[i]["answer"]))


# Print("-------- Preparing LoRA --------")

training_args = transformers.TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim=config.optim,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.learning_rate,
            report_to="none",
            logging_steps=config.logging_steps, # Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.
            #eval_steps=config.eval_steps,
            max_grad_norm=config.max_grad_norm,
            num_train_epochs=config.num_train_epochs,
            warmup_steps=config.warmup_steps,
            group_by_length=True,
            lr_scheduler_type="constant_with_warmup",
            do_eval=True,
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=config.eval_accumulation_steps,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
        )

response_template_with_context = "\n ### Answer:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer(
    response_template_with_context,
    add_special_tokens=False,
).input_ids[2:]
Print(f"response template tokens: {tokenizer.convert_ids_to_tokens(response_template_ids)}")
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


header = json.dumps({
    "data_dir": config.data_dir,
    "load_from_ckpt": args.specific_ckpt,
})
trainer = SFTTrainer(
    model=model,
    #train_dataset=train_data,
    eval_dataset=val_data,
    #peft_config=lora_config,
    formatting_func=partial(formatting_func, prompt_template=prompt_template),
    data_collator=collator,
    max_seq_length=config.max_seq_length,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, 
                            config=config, 
                            tokenizer=tokenizer,
                            save_to="test_samples",
                            header=header),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
# for name, module in trainer.model.named_modules():
#     if "norm" in name:
#         module = module.to(torch.float32)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
   trainer.evaluate()
