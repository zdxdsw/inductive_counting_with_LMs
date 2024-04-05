import os, sys, json, random, io, pytz, argparse, pytz, re
os.environ['HF_HOME'] = '/data/yingshac/hf_cache'
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from datasets.features import Features, Value
from torch.utils.data import DataLoader
from config import *
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig,
        AutoPeftModelForCausalLM,
        prepare_model_for_int8_training
    )
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from utils import *
#from utils import NiceSFTTrainer as SFTTrainer
from functools import partial
from pprint import pprint
#from datetime import datetime
#timezone = pytz.timezone('America/New_York') 
#date = datetime.now(timezone).strftime("%m%d_%H%M%S")

def Print(s):
   if not Accelerator().process_index:
      print(s)


parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default="debug")
parser.add_argument('--task', type=str)
args = parser.parse_args()

Print("----------- Preparing configs -----------")

config = eval(f"FinetuningConfig_{args.task}")()
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

with open(config.prompt_path, "r") as f: prompt_template = "\n".join(f.readlines()).strip()


Print("-------- Preparing model and tokenizer --------")

bnb_config = None
if config.bits is not None:
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.bits == 4,
            load_in_8bit=config.bits == 8,
            llm_int8_threshold=0.,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )

if config.load_from_dir is None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = LlamaForCausalLM.from_pretrained(config.model_name,
                                                load_in_8bit=config.bits==8,
                                                torch_dtype=torch.float16,
                                                device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}, # {"": Accelerator().process_index},
                                                #quantization_config = bnb_config,
                                                )

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # Turn-off an annoying warning msg
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'right'
    already_trained_epochs = 0
    resume_from_checkpoint = False
else: # resume from previous ckpt
    # lora_adapter = os.path.join(config.ckpt_dir, config.load_from_dir)
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     lora_adapter, 
    #     torch_dtype=torch.float16, 
    #     device_map={"": Accelerator().process_index}
    # )
    
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(lora_adapter)
    # except OSError:
    #     tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    #     tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # Turn-off an annoying warning msg
    #     tokenizer.add_special_tokens({"pad_token":"<pad>"})
    #     tokenizer.padding_side = 'right'
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = LlamaForCausalLM.from_pretrained(config.model_name,
                                                load_in_8bit=config.bits==8,
                                                torch_dtype=torch.float16,
                                                device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}, # {"": Accelerator().process_index},
                                                #quantization_config = bnb_config,
                                                )

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # Turn-off an annoying warning msg
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'right'
    resume_from_checkpoint = os.path.join(config.ckpt_dir, config.load_from_dir)

model.resize_token_embeddings(len(tokenizer))
if config.bits==8: model = prepare_model_for_int8_training(model)


Print("------------ Preparing data ------------")

def generate_and_tokenize_prompt(instance):
    prompt = preprocess(prompt_template, instance["input_str"], instance["answer"])
    tokenized_full_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

    user_prompt = preprocess(prompt_template, instance["input_str"], None)
    tokenized_user_prompt = tokenizer(
            user_prompt,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])-1

    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
    return tokenized_full_prompt

if isinstance(config.data_dir, str): config.data_dir = [config.data_dir]
train_data = concatenate_datasets([load_dataset(
                        ddir, 
                        split="train",
                        features=Features({
                            'input_str':Value("string"), 
                            "answer":Value("string")}),
                        ) for ddir in config.data_dir]).map(generate_and_tokenize_prompt)

#train_dl = DataLoader(train_data, batch_size=128, shuffle=True)

val_data = concatenate_datasets([load_dataset(
                        ddir, 
                        split="validation",
                        features=Features({
                            'input_str':Value("string"), 
                            "answer":Value("string")}),
                        ) for ddir in config.data_dir]).map(generate_and_tokenize_prompt)
#val_dl = DataLoader(val_data, batch_size=128, shuffle=False)

Print(f"num train = {len(train_data)}")
Print(f"num val = {len(val_data)}")

i = random.choice(list(range(len(val_data))))
Print("\ntask = {}".format(config.prompt_path.split("/")[-2]))
Print("\nexample data: ")
Print(preprocess(prompt_template, val_data[i]["input_str"], val_data[i]["answer"]))

if config.load_from_dir is not None:
    already_trained_epochs = int(config.load_from_dir.split("-")[-1]) // (len(train_data) / config.per_device_train_batch_size / config.gradient_accumulation_steps / 4)
    Print(f"\nResume from {config.load_from_dir} Epoch {int(already_trained_epochs)}\n")

Print("------------ Preparing LoRA ------------")

lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False})
model.config.use_cache = False # warning: use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False
model = get_peft_model(model, lora_config)

Print(f"Total parameters: {model.num_parameters()}")
Print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")


training_args = transformers.TrainingArguments(
            output_dir=os.path.join(config.ckpt_dir, config.date, "ckpts"),
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
            num_train_epochs=config.num_train_epochs, #-already_trained_epochs,
            warmup_steps=config.warmup_steps,
            group_by_length=True, # GOAT says: False is faster, but produces an odd training loss curve
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
collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ) #DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

trainer = transformers.Trainer( #SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    # peft_config=lora_config,
    # formatting_func=partial(formatting_func, prompt_template=prompt_template),
    data_collator=collator,
    # max_seq_length=config.max_seq_length,
    # tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, 
                            config=config, 
                            tokenizer=tokenizer,
                            save_to="eval_samples"),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    if config.zeroshot_eval: trainer.evaluate()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    #trainer.evaluate()

    # Print("Save final model")
    # if not Accelerator().process_index:
    #     trainer.save_model(f"{training_args.output_dir}/final")

