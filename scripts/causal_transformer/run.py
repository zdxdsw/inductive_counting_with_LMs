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
  os.makedirs(os.path.join(config.output_dir, config.date, "eval_samples"), exist_ok=True)

  # dump config
  C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
  with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
    json.dump(C, f, indent=2)

os.system("accelerate launch trainer.py --date {} --task {} | tee {}".format(
    config.date,
    args.task,
    os.path.join(config.output_dir, date, "terminal.txt"),
))