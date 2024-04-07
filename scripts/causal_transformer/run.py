import random, time, os, pytz, argparse, yaml, sys, json
from config import *

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="")
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
      if k not in ["warmup_steps", "learning_rate", "num_epochs", "save_every_steps", "eval_every_steps", "logging_steps"]:
        setattr(config, k, resume_from_config[k])
      args.task = config.data_path.split("/")[-1] ## auto infer task from the incomplete run

  # dump config
  C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
  with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
    json.dump(C, f, indent=2)

os.system("accelerate launch trainer.py --date {} --task {} | tee {}".format(
    config.date,
    args.task,
    os.path.join(config.output_dir, date, "terminal.txt"),
))