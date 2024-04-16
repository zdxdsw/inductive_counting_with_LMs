import random, time, os, pytz, argparse, yaml, sys, json
from config import *
from tqdm import trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="")
parser.add_argument('--sleep', type=int)
parser.add_argument('--accelerator', type=str, default="")

args = parser.parse_args()


"""------------- Preparing configs -------------"""
if len(args.task): config = eval(f"{args.task}_Config")()
else: config = Basic_Config()
config.date = date

if config.output_dir is not None:
  os.makedirs(os.path.join(config.ckpt_dir, config.date, "ckpts"), exist_ok=True)
  os.makedirs(os.path.join(config.output_dir, config.date), exist_ok=True)

  if config.load_from_dir is not None:
    resume_from_config = json.load(open(os.path.join(config.output_dir, config.load_from_dir, "config.json"), "r"))
    for k in resume_from_config:
      if k not in ["warmup_steps", "learning_rate", "num_epochs", "save_every_steps", "eval_every_steps", "logging_steps", "load_from_dir", "date"]:
        setattr(config, k, resume_from_config[k])
    args.task = resume_from_config['data_path'].split("/")[-1] ## auto infer task from the incomplete run

  # dump config
  config.task = args.task
  C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
  with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
    json.dump(C, f, indent=2)

"""------------- Sleep if needed -------------"""
if args.sleep is not None:
  for i in trange(args.sleep, desc="Sleeping"):
    time.sleep(60)

os.system("accelerate launch {} trainer.py --date {} --task {} | tee {}".format(
    args.accelerator,
    config.date,
    args.task,
    os.path.join(config.output_dir, date, "terminal.txt"),
))