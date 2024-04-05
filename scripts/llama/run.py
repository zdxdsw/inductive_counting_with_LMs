import random, time, os, pytz, argparse, yaml, sys, json
from config import *

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="")
parser.add_argument('--accelerate_config', type=str, default="")
args = parser.parse_args()

print("-------- Preparing configs --------")
if len(args.task): config = eval(f"FinetuningConfig_{args.task}")()
else: config = FinetuningConfig()
config.date = date

if config.output_dir is not None:
  os.makedirs(os.path.join(config.ckpt_dir, config.date, "ckpts"), exist_ok=True)
  os.makedirs(os.path.join(config.output_dir, config.date, "eval_samples"), exist_ok=True)

  if config.load_from_dir is not None:
    ckpt_config = json.load(open(os.path.join(config.output_dir, config.load_from_dir, "config.json"), "r"))
    for k in ["lora_target_modules", "prompt_path", "data_dir", "max_seq_length"]:
        setattr(config, k, ckpt_config[k])
    # find the last ckpt to resume_from
    ckpt_dir = os.path.join(config.ckpt_dir, config.load_from_dir, "ckpts")
    specific_ckpt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[-1]))[-1]
    config.load_from_dir = f"{config.load_from_dir}/ckpts/{specific_ckpt}"
    config.zeroshot_eval = False # Since the previous ckpt will be loaded when trainer.train() is called, 
                                 # eval would perform on the un-finetuned model, which is unnecessary.
    args.task = config.prompt_path.split("/")[-2] ## auto infer task from the incomplete run

  # dump config
  C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
  with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
    json.dump(C, f, indent=2)

os.system("accelerate launch {} llama_finetune.py --date {} --task {} | tee {}".format(
    args.accelerate_config,
    config.date,
    args.task,
    os.path.join(config.output_dir, date, "terminal.txt"),
))